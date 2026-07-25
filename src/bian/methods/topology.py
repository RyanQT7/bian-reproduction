"""Candidate-covering topology reduction based on shortest paths."""

from __future__ import annotations

from collections import deque

from bian.data.schema import TopologyRecord
from bian.data.validators import ValidationError


def _adjacency(topology: TopologyRecord) -> dict[str, set[str]]:
    result = {node: set() for node in topology.nodes}
    for left, right in topology.edges:
        result[left].add(right)
        if not topology.directed:
            result[right].add(left)
    return result


def _shortest_path(graph: dict[str, set[str]], start: str, end: str) -> tuple[str, ...]:
    queue = deque([(start, (start,))])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in sorted(graph[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
    raise ValidationError(f"no topology path between {start} and {end}")


def extract_candidate_subgraph(
    topology: TopologyRecord, candidates: tuple[str, ...]
) -> TopologyRecord:
    if not candidates:
        raise ValidationError("candidates must not be empty")
    if not set(candidates) <= set(topology.nodes):
        raise ValidationError("all candidates must exist in topology")
    graph = _adjacency(topology)
    candidate_set = set(candidates)
    paths: list[tuple[str, ...]] = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            path = _shortest_path(graph, left, right)
            if any(node in candidate_set for node in path[1:-1]):
                continue
            paths.append(path)
    nodes = set(candidates)
    edges: set[tuple[str, str]] = set()
    for path in paths:
        nodes.update(path)
        for left, right in zip(path, path[1:]):
            edge = (left, right) if topology.directed or left <= right else (right, left)
            edges.add(edge)
    return TopologyRecord(
        nodes=tuple(sorted(nodes)),
        edges=tuple(sorted(edges)),
        device_groups={node: topology.device_groups[node] for node in nodes if node in topology.device_groups},
        directed=topology.directed,
    )
