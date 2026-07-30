# Device metric PDFs

`scripts/plot_device_metrics_pdf.py` renders one raw-metric PDF per mapped device. It
uses the no-traffic device/metric manifests, preserves continuous-segment gaps, draws
binary states as steps, and includes device anomaly timelines only on the overview page.
For post-hoc review it can join evaluation, localization, classification, and incident-time
records strictly by `incident_id`, adding per-device and all-network ground-truth bands.
