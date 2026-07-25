from datetime import datetime, timezone
import math
import unittest

from bian.data.schema import AlertRecord, IncidentRecord, TopologyRecord
from bian.data.validators import (
    ValidationError,
    normalize_scores,
    validate_incident,
    validate_scores,
)
from bian.evaluation.hot_device import hot_device_ranking
from bian.evaluation.metrics import top_k_accuracy
from bian.methods.candidate_filter import top_p_candidates
from bian.methods.early_stop import score_entropy, should_early_stop
from bian.methods.rank_of_ranks import aggregate_rankings
from bian.methods.timeline import build_timeline
from bian.methods.topology import extract_candidate_subgraph


class CoreMethodTests(unittest.TestCase):
    candidates = ("A", "B", "C")

    def alert(self, alert_id, device, second=0):
        return AlertRecord(
            alert_id,
            (device,),
            "mock",
            datetime(2026, 1, 1, 0, 0, second, tzinfo=timezone.utc),
        )

    def test_scores_normalize_and_validate(self):
        scores = normalize_scores({"A": 3, "B": 1, "C": 0})
        validate_scores(scores, self.candidates)
        self.assertTrue(math.isclose(sum(scores.values()), 1.0))

    def test_invalid_scores_rejected(self):
        for scores in (
            {"A": math.inf, "B": 0, "C": 0},
            {"A": -1, "B": 1, "C": 1},
            {"A": 0.1, "B": 0.1, "C": 0.1},
        ):
            with self.assertRaises(ValidationError):
                validate_scores(scores, self.candidates)

    def test_candidate_filter_is_deterministic(self):
        scores = {"A": 0.8, "B": 0.15, "C": 0.05}
        self.assertEqual(top_p_candidates(scores, self.candidates, 0.7), ("A", "B"))

    def test_raw_and_normalized_entropy(self):
        scores = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
        raw = score_entropy(scores, self.candidates, "raw")
        normalized = score_entropy(scores, self.candidates, "normalized")
        self.assertAlmostEqual(raw, math.log(3))
        self.assertAlmostEqual(normalized, 1.0)
        self.assertTrue(should_early_stop({"A": 1.0, "B": 0.0, "C": 0.0}, self.candidates, 0.75))

    def test_rank_of_ranks(self):
        runs = [
            {"A": 0.6, "B": 0.3, "C": 0.1},
            {"A": 0.5, "B": 0.2, "C": 0.3},
            {"A": 0.4, "B": 0.35, "C": 0.25},
        ]
        ranking, averages = aggregate_rankings(runs, self.candidates)
        self.assertEqual(ranking[0], "A")
        self.assertEqual(averages["A"], 1.0)

    def test_timeline_is_strictly_sorted(self):
        alerts = (self.alert("z", "A", 1), self.alert("a", "B", 1), self.alert("x", "C", 0))
        timeline = build_timeline(alerts)
        self.assertTrue(all(left.timestamp < right.timestamp for left, right in zip(timeline, timeline[1:])))

    def test_topology_covers_candidates(self):
        topology = TopologyRecord(("A", "X", "B", "C"), (("A", "X"), ("X", "B"), ("B", "C")))
        reduced = extract_candidate_subgraph(topology, self.candidates)
        self.assertTrue(set(self.candidates) <= set(reduced.nodes))

    def test_hot_device_and_top_k(self):
        alerts = (self.alert("1", "B"), self.alert("2", "B"), self.alert("3", "A"))
        ranking = hot_device_ranking(alerts, self.candidates)
        self.assertEqual(ranking[0], "B")
        self.assertEqual(top_k_accuracy([ranking], [("B",)], 1), 1.0)

    def test_missing_ground_truth_rejected(self):
        with self.assertRaises(ValidationError):
            top_k_accuracy([self.candidates], [None], 1)

    def test_missing_required_incident_field_rejected(self):
        incident = IncidentRecord(
            "",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            self.candidates,
            (self.alert("1", "A"),),
            TopologyRecord(self.candidates, (("A", "B"), ("B", "C"))),
        )
        with self.assertRaises(ValidationError):
            validate_incident(incident)

    def test_unknown_topology_endpoint_rejected(self):
        incident = IncidentRecord(
            "bad-edge",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            self.candidates,
            (self.alert("1", "A"),),
            TopologyRecord(self.candidates, (("A", "UNKNOWN"),)),
        )
        with self.assertRaises(ValidationError):
            validate_incident(incident)

    def test_invalid_entropy_mode_rejected(self):
        scores = {"A": 0.8, "B": 0.1, "C": 0.1}
        with self.assertRaises(ValidationError):
            score_entropy(scores, self.candidates, "paper-unknown")


if __name__ == "__main__":
    unittest.main()
