"""Deterministic generic eight-region topology for the blind RCA dataset."""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any

from .validators import ValidationError


CANDIDATE_ROLES = (
    "br-1",
    "br-2",
    "cr-1",
    "cr-2",
    "fw",
    "traffic-vm",
    "service-1",
    "service-2",
    "service-3",
)
SWITCH_ROLES = ("sw-ext", "sw-core", "sw-mid", "sw-access")
REMOTE_AS = re.compile(r'remote_as="(65\d{3})"')


def inventory() -> list[dict[str, Any]]:
    return [
        {
            "node_id": f"region-{index}-{role}",
            "region_index": index,
            "region_id": f"region-{index}",
            "role": role,
            "candidate": True,
        }
        for index in range(1, 9)
        for role in CANDIDATE_ROLES
    ]


def discover_peer_regions(
    raw_root: Path, region_mapping: dict[Path, str]
) -> set[tuple[int, int]]:
    """Infer public inter-region links from BGP remote-AS labels."""
    links: set[tuple[int, int]] = set()
    for region_dir, region_id in region_mapping.items():
        region_index = int(region_id.split("-")[1])
        files = sorted((region_dir / "processed").glob("routing_metrics_*.csv"))
        if len(files) != 1:
            raise ValidationError(f"expected one routing file under {region_dir}")
        remote_regions: set[int] = set()
        with files[0].open(
            newline="", encoding="utf-8-sig", errors="replace"
        ) as handle:
            reader = csv.DictReader(handle)
            first_timestamp = None
            for row in reader:
                timestamp = row.get("timestamp")
                if first_timestamp is None:
                    first_timestamp = timestamp
                elif timestamp != first_timestamp and remote_regions:
                    break
                for value in REMOTE_AS.findall(row.get("label", "")):
                    remote_index = int(value) - 65000
                    if 1 <= remote_index <= 8 and remote_index != region_index:
                        remote_regions.add(remote_index)
        links.update(
            tuple(sorted((region_index, remote_index)))
            for remote_index in remote_regions
        )
    return links


def build_topology(inter_region_links: set[tuple[int, int]]) -> dict[str, Any]:
    nodes = inventory() + [
        {
            "node_id": f"region-{index}-{role}",
            "region_index": index,
            "region_id": f"region-{index}",
            "role": role,
            "candidate": False,
        }
        for index in range(1, 9)
        for role in SWITCH_ROLES
    ]
    edges: list[dict[str, Any]] = []

    def add(index: int, source: str, target: str, edge_type: str, protocol=None):
        edges.append(
            {
                "source": f"region-{index}-{source}",
                "target": f"region-{index}-{target}",
                "edge_type": edge_type,
                "protocol": protocol,
            }
        )

    for index in range(1, 9):
        add(index, "sw-ext", "br-1", "l2_attachment", "ebgp_transit")
        add(index, "sw-ext", "br-2", "l2_attachment", "ebgp_transit")
        add(index, "br-1", "sw-core", "l2_attachment", "ospfv3")
        add(index, "br-2", "sw-core", "l2_attachment", "ospfv3")
        add(index, "sw-core", "cr-1", "l2_attachment", "ospfv3")
        add(index, "sw-core", "cr-2", "l2_attachment", "ospfv3")
        add(index, "cr-1", "sw-mid", "l2_attachment", "ospfv3")
        add(index, "cr-2", "sw-mid", "l2_attachment", "ospfv3")
        add(index, "sw-mid", "fw", "l3_transit")
        add(index, "fw", "sw-access", "gateway_link")
        add(index, "sw-access", "traffic-vm", "access_link")
        for service in ("service-1", "service-2", "service-3"):
            add(index, "sw-access", service, "access_link")
    for left, right in sorted(inter_region_links):
        for br in ("br-1", "br-2"):
            edges.append(
                {
                    "source": f"region-{left}-{br}",
                    "target": f"region-{right}-{br}",
                    "edge_type": "inter_region",
                    "protocol": "ebgp",
                }
            )
    return {"directed": False, "nodes": nodes, "edges": edges}
