"""General monitoring requirements for the public experiment taxonomy."""

PROFILES = {
    "bgp_route_flap": ("BR", "Repeated route announce/withdraw or route-count oscillation", "stable BGP session and route table"),
    "bgp_session_down": ("BR", "BGP neighbor/session transition to down and route loss", "session remains established"),
    "firewall_acl_drop": ("FW", "ACL deny/drop counters or matching firewall deny logs", "no matching rule hits"),
    "firewall_cpu_pressure": ("FW", "Sustained CPU saturation aligned with impaired forwarding", "normal CPU and no resource queueing"),
    "firewall_default_route_error": ("FW", "Default route disappearance or incorrect next hop", "valid unchanged default route"),
    "firewall_port_block": ("FW", "Port-specific blocking state and failed connections", "port permitted with successful connections"),
    "firewall_rate_limit": ("FW", "Rate-limit counters, shaping events, or throughput ceiling", "no policy limit hits"),
    "firewall_rule_order_error": ("FW", "Rule order change causing an earlier rule to shadow a later rule", "unchanged effective rule ordering"),
    "ospf6_cost_anomaly": ("CR/BR", "OSPFv3 interface cost change with path selection change", "stable cost and selected path"),
    "ospf6_neighbor_down": ("CR/BR", "OSPFv3 neighbor state transition away from Full", "neighbor remains Full"),
    "route_blackhole": ("BR/CR", "Route resolves to discard/null or forwarding has no viable next hop", "reachable next hop and forwarding"),
    "wrong_default_route": ("BR/CR", "Incorrect default route next hop or preference", "expected default route selected"),
    "wrong_static_route": ("BR/CR", "Incorrect static prefix, next hop, metric, or route replacement", "expected static route unchanged"),
}


def profile(fault_type: str) -> dict[str, str]:
    role, required, counter = PROFILES[fault_type]
    return {
        "definition": fault_type.replace("_", " "),
        "common_roles": role,
        "required_monitoring_evidence": required,
        "possible_counter_evidence": counter,
    }
