#!/usr/bin/env python3
"""Render one multi-page raw-metric PDF per mapped device."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "device"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _role(device_id: str) -> str:
    parts = device_id.split("-", 2)
    return parts[2] if len(parts) == 3 else device_id


def _load_ground_truth(main_path: Path, valid_devices: set[str],
                       localization_path: Path | None = None,
                       classification_path: Path | None = None,
                       incident_time_path: Path | None = None) -> tuple[list[dict], list[dict]]:
    main = {x["incident_id"]: x for x in _jsonl(main_path)}
    supplements = []
    for path in (localization_path, classification_path, incident_time_path):
        supplements.append({x["incident_id"]: x for x in _jsonl(path)} if path else {})
    aliases = {"br-a":"br-1", "br-b":"br-2",
               "service-vm-1":"service-1", "service-vm-2":"service-2",
               "service-vm-3":"service-3"}
    normalized, audit = [], []
    for incident_id, base in sorted(main.items()):
        record = dict(base)
        for supplement in supplements:
            for key, value in supplement.get(incident_id, {}).items():
                if key not in record or record[key] in (None, "", []):
                    record[key] = value
        roots = record.get("root_device_ids") or []
        raw_location = record.get("root_device_id_raw", "")
        mapped, basis = [], ""
        for root in roots:
            candidate = root
            if candidate not in valid_devices:
                match = re.fullmatch(r"(region-\d+)-(.+)", candidate)
                if match and match.group(2) in aliases:
                    candidate = f"{match.group(1)}-{aliases[match.group(2)]}"
            if candidate in valid_devices:
                mapped.append(candidate); basis = "exact_canonical_device_id_or_explicit_alias"
        mapped = sorted(set(mapped))
        status = "mapped" if mapped else "unmapped"
        reason = "" if mapped else (
            "no exact canonical root device in device_mapping_manifest; no fuzzy assignment")
        if "start_time" not in record or "end_time" not in record:
            raise ValueError(f"{incident_id} lacks start/end time after incident_id joins")
        start, end = pd.Timestamp(record["start_time"]), pd.Timestamp(record["end_time"])
        normalized.append({
            "incident_id": incident_id, "start_time": start.isoformat(),
            "end_time": end.isoformat(), "duration_minutes": (end-start).total_seconds()/60,
            "raw_location": raw_location or "|".join(roots),
            "normalized_device_id": "|".join(mapped),
            "fault_type": record.get("fault_type", ""), "mapping_status": status,
            "_mapped_devices": mapped})
        audit.append({
            "incident_id": incident_id, "raw_location": raw_location,
            "canonical_location": "|".join(roots), "normalized_location": "|".join(mapped),
            "final_device_id": "|".join(mapped), "mapping_status": status,
            "mapping_basis": basis, "unmapped_reason": reason})
    return normalized, audit


def _plot_metric(ax, data: pd.DataFrame, metric_type: str) -> None:
    """Draw every continuous segment separately so missing gaps remain disconnected."""
    for _, segment in data.groupby("continuous_segment_id", sort=True):
        segment = segment.sort_values("timestamp")
        if metric_type == "binary_state":
            ax.step(segment.timestamp, segment.raw_value, where="post", linewidth=.7)
        else:
            ax.plot(segment.timestamp, segment.raw_value, linewidth=.6)


def _overview(pdf: PdfPages, device: str, role: str, metric_count: int,
              point_count: int, data_start, data_end, start, end,
              detected: list[dict], persistent: list[dict],
              device_truth: list[dict], all_truth: list[dict],
              total_pages: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    grid = fig.add_gridspec(5, 1, height_ratios=[1.15, 1, 1, 1, 1])
    info = fig.add_subplot(grid[0]); info.axis("off")
    info.text(0, 1, "\n".join([
        f"Device: {device}", f"Role: {role}", f"Metrics: {metric_count}",
        f"Data range: {data_start} to {data_end}", f"Data points: {point_count:,}",
        f"Detected intervals: {len(detected)}",
        f"Persistent intervals (>35 min): {len(persistent)}",
        f"Ground-truth faults on device: {len(device_truth)}",
    ]), va="top", fontsize=11)
    axes = [fig.add_subplot(grid[i]) for i in range(1, 5)]
    for ax, rows, title, label in (
        (axes[0], detected, "MOMENT Detected Device Intervals", "Detected"),
        (axes[1], persistent, "Persistent device intervals", "Persistent"),
        (axes[2], device_truth,
         f"Ground-Truth Faults on This Device ({len(device_truth)})", "Ground truth"),
        (axes[3], all_truth,
         f"All Ground-Truth Fault Times ({len(all_truth)})", "All ground truth"),
    ):
        ax.set_xlim(start, end); ax.set_ylim(-.1, 1.15); ax.set_yticks([0, 1])
        ax.set_title(title); ax.grid(True, alpha=.25)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        if rows:
            for row in rows:
                left, right = pd.Timestamp(row["start_time"]), pd.Timestamp(row["end_time"])
                ax.plot([left, left, right, right], [0, 1, 1, 0], linewidth=1, label=label)
                if ax is axes[2]:
                    annotation = f"{row['incident_id']} {row.get('fault_type','')}".strip()
                    ax.annotate(annotation, (left, 1), xytext=(2, 2),
                                textcoords="offset points", fontsize=5, rotation=20)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:1], labels[:1], loc="upper right")
        else:
            ax.plot([start, end], [0, 0], linewidth=.7)
            empty = ("No ground-truth faults on this device"
                     if ax is axes[2] else f"No {label.lower()} intervals")
            ax.text(.5, .55, empty, transform=ax.transAxes,
                    ha="center", va="center")
    fig.text(.5, .015, f"Page 1 / {total_pages}", ha="center")
    fig.tight_layout(rect=[0, .03, 1, 1]); pdf.savefig(fig); plt.close(fig)


def render(args) -> tuple[list[dict], list[dict]]:
    data_root, manifest_root, output = map(Path, (
        args.data_root, args.manifest_root, args.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    series_path = Path(args.series_parquet) if args.series_parquet else (
        data_root.parents[1] / "moment_inputs" / f"{data_root.name}_no_traffic_v2" / "series.parquet")
    metrics = pd.read_csv(manifest_root / "device_metric_manifest.csv")
    mappings = pd.read_csv(manifest_root / "device_mapping_manifest.csv")
    traffic = metrics.source_file.str.match(r"(?:^|.*/)traffic_[^/]*\.csv$", na=False)
    metrics = metrics.loc[~traffic].copy()
    mapping_traffic = mappings.source_file.str.match(r"(?:^|.*/)traffic_[^/]*\.csv$", na=False)
    if mapping_traffic.any():
        raise ValueError("traffic source found in device mapping manifest")
    detected_all = _jsonl(manifest_root / "device_intervals.jsonl")
    persistent_all = [
        x for x in _jsonl(manifest_root / "persistent_long_intervals.jsonl")
        if x.get("level") == "device"]
    gt_main = getattr(args, "ground_truth_file", None)
    all_truth, mapping_audit = ([], [])
    if gt_main:
        all_truth, mapping_audit = _load_ground_truth(
            Path(gt_main), set(mappings.device_id.unique()),
            Path(args.localization_ground_truth_file)
            if getattr(args, "localization_ground_truth_file", None) else None,
            Path(args.classification_ground_truth_file)
            if getattr(args, "classification_ground_truth_file", None) else None,
            Path(args.incident_time_file)
            if getattr(args, "incident_time_file", None) else None)
        normalized_fields = ["incident_id","start_time","end_time","duration_minutes",
                             "raw_location","normalized_device_id","fault_type","mapping_status"]
        with (output / "ground_truth_intervals_normalized.csv").open(
                "w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=normalized_fields); writer.writeheader()
            writer.writerows([{k:v for k,v in x.items() if not k.startswith("_")}
                              for x in all_truth])
        with (output / "ground_truth_mapping_audit.csv").open(
                "w", newline="", encoding="utf-8") as f:
            fields=list(mapping_audit[0]) if mapping_audit else [
                "incident_id","raw_location","canonical_location","normalized_location",
                "final_device_id","mapping_status","mapping_basis","unmapped_reason"]
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            writer.writerows(mapping_audit)
    start, end = pd.Timestamp(args.start_time), pd.Timestamp(args.end_time)
    index_rows, errors = [], []
    for device in sorted(metrics.device_id.unique()):
        try:
            device_metrics = metrics[metrics.device_id == device].sort_values(
                ["base_metric_family", "metric", "source_file"])
            data = pd.read_parquet(
                series_path,
                columns=["timestamp","device_id","metric","raw_value",
                         "continuous_segment_id","metric_type"],
                filters=[("device_id", "==", device)])
            data.timestamp = pd.to_datetime(data.timestamp, utc=True)
            data = data[(data.timestamp >= start) & (data.timestamp < end)]
            data = data.groupby(
                ["timestamp","device_id","metric","continuous_segment_id","metric_type"],
                as_index=False).raw_value.mean()
            plotted = [
                row for row in device_metrics.itertuples(index=False)
                if row.metric in set(data.loc[data.raw_value.notna(), "metric"])]
            page_count = 1 + math.ceil(len(plotted) / args.plots_per_page)
            pdf_path = output / f"{_safe_name(device)}.pdf"
            detected = [x for x in detected_all if x.get("device_id") == device]
            persistent = [x for x in persistent_all if x.get("device_id") == device]
            device_truth = [x for x in all_truth if device in x["_mapped_devices"]]
            with PdfPages(pdf_path) as pdf:
                _overview(pdf, device, _role(device), len(plotted), len(data),
                          data.timestamp.min(), data.timestamp.max(), start, end,
                          detected, persistent, device_truth, all_truth, page_count)
                for offset in range(0, len(plotted), args.plots_per_page):
                    page_number = 2 + offset // args.plots_per_page
                    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=True)
                    for ax, row in zip(axes.flat, plotted[offset:offset+args.plots_per_page]):
                        subset = data[data.metric == row.metric]
                        _plot_metric(ax, subset, row.metric_type)
                        title = f"{row.metric}\nfamily={row.base_metric_family}; type={row.metric_type}"
                        ax.set_title(textwrap.fill(title, 95), fontsize=7)
                        ax.set_xlim(start, end); ax.grid(True, alpha=.25)
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(
                            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                    for ax in axes.flat[len(plotted[offset:offset+args.plots_per_page]):]:
                        ax.axis("off")
                    fig.text(.5, .012, f"Page {page_number} / {page_count}", ha="center")
                    fig.tight_layout(rect=[0, .025, 1, 1]); pdf.savefig(fig); plt.close(fig)
            index_rows.append({
                "device_id": device, "device_role": _role(device),
                "pdf_path": str(pdf_path.resolve()), "metric_count": len(plotted),
                "page_count": page_count,
                "start_time": start.isoformat(), "end_time": end.isoformat(),
                "detected_interval_count": len(detected),
                "persistent_interval_count": len(persistent),
                "ground_truth_fault_count_on_device": len(device_truth),
                "all_ground_truth_fault_count": len(all_truth),
                "ground_truth_mapping_status": (
                    "mapped_faults_present" if device_truth else "no_mapped_fault_on_device")})
            del data, device_metrics, plotted
        except Exception as exc:
            plt.close("all")
            errors.append({"device_id": device, "error_type": type(exc).__name__,
                           "error_message": str(exc)})
    fields = ["device_id","device_role","pdf_path","metric_count","page_count","start_time",
              "end_time","detected_interval_count","persistent_interval_count",
              "ground_truth_fault_count_on_device","all_ground_truth_fault_count",
              "ground_truth_mapping_status"]
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(index_rows)
    with (output / "plot_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["device_id","error_type","error_message"])
        writer.writeheader(); writer.writerows(errors)
    return index_rows, errors


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",required=True)
    p.add_argument("--manifest-root",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--start-time",required=True)
    p.add_argument("--end-time",required=True)
    p.add_argument("--plots-per-page",type=int,default=6)
    p.add_argument("--series-parquet")
    p.add_argument("--ground-truth-file")
    p.add_argument("--localization-ground-truth-file")
    p.add_argument("--classification-ground-truth-file")
    p.add_argument("--incident-time-file",
                   help="Optional incident_id-keyed time source when evaluation GT lacks times")
    args=p.parse_args()
    if not 1 <= args.plots_per_page <= 6:
        p.error("--plots-per-page must be between 1 and 6")
    rows,errors=render(args)
    print(json.dumps({"pdfs":len(rows),"pages":sum(x["page_count"] for x in rows),
                      "metrics":sum(x["metric_count"] for x in rows),
                      "errors":len(errors)}))

if __name__=="__main__":
    main()
