"""Escaped, standalone reports built from one immutable analysis artifact."""

import html
import json
import re

REPORT_SECTIONS = {
    "overview": "Dataset overview",
    "quality": "Quality and schema",
    "profiles": "Column statistics and distributions",
    "correlations": "Numerical correlations",
    "associations": "Categorical associations",
    "outliers": "Potential anomalies",
    "insights": "Evidence-backed findings",
}


def display(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return html.escape(str(value))


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join("<th>" + display(h) + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join("<td>" + display(v) + "</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def html_report(r, sections=None):
    output = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Prism analytical report</title>
    <style>body{font:14px system-ui;max-width:1100px;margin:40px auto;color:#17253b;padding:20px}h1{color:#087e8b}h2{border-bottom:1px solid #ccd;padding-top:20px}table{border-collapse:collapse;width:100%;font-size:11px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:8px;overflow-wrap:anywhere}th{background:#edf4f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.finding{border-left:3px solid #087e8b;padding:10px;margin:15px 0}svg{max-width:100%}@media print{body{margin:0}tr{break-inside:avoid}h2,h3{break-after:avoid}}</style>
    <h1>Prism · Analytical report</h1><p>Exploratory evidence, not causal proof. Potential anomalies are investigation candidates, not automatic errors.</p>"""
    ]
    output.append(
        "<h2>Provenance</h2>"
        + table(
            ["Field", "Value"],
            [(k, r[k]) for k in ("run_id", "fingerprint", "created_at", "algorithm_version")],
        )
    )
    output.append(
        "<h2>Dataset overview</h2>"
        + table(["Metric", "Value"], [(k, v) for k, v in r["overview"].items() if not isinstance(v, dict)])
    )
    output.append(
        "<h2>Quality and schema</h2>"
        + table(
            ["Column", "Physical type", "Semantic type", "Missing %", "Distinct", "Invalid", "Nonfinite"],
            [
                (
                    p["name"],
                    p["physical_type"],
                    p["semantic_type"],
                    p["missing_percentage"],
                    p["distinct_count"],
                    p.get("invalid_count", 0),
                    p.get("nonfinite_count", 0),
                )
                for p in r["profiles"]
            ],
        )
    )
    output.append("<h2>Column statistics and distributions</h2>")
    for p in r["profiles"]:
        output.append("<h3>" + display(p["name"]) + "</h3>")
        output.append(table(["Statistic", "Value"], list(p["statistics"].items())))
        d = p["distribution"]
        if d.get("counts"):
            counts = d["counts"]
            maximum = max(counts) or 1
            width = 600 / len(counts)
            bars = "".join(
                f'<rect x="{i * width:.2f}" y="{150 - v / maximum * 140:.2f}" width="{max(0.1, width - 1):.2f}" height="{v / maximum * 140:.2f}" fill="#087e8b"/>'
                for i, v in enumerate(counts)
            )
            output.append('<svg role="img" aria-label="Histogram" viewBox="0 0 600 155">' + bars + "</svg>")
            output.append(
                table(["Bin lower", "Bin upper", "Count"], list(zip(d["edges"][:-1], d["edges"][1:], counts)))
            )
        else:
            output.append(
                table(
                    ["Value (top 20)", "Count", "Fraction"],
                    [(v["value"], v["count"], v["relative_frequency"]) for v in p["top_values"]],
                )
            )
            output.append("<p>Other records: " + display(p["other_count"]) + "</p>")
        if p.get("temporal"):
            output.append("<pre>" + display(json.dumps(p["temporal"], indent=2)) + "</pre>")
    output.append(
        "<h2>Numerical correlations</h2>"
        + table(
            ["X", "Y", r["config"]["correlation"], "Valid pairs"],
            [(p["x"], p["y"], p["coefficient"], p["n"]) for p in r["correlations"]],
        )
    )
    output.append(
        "<h2>Categorical associations</h2>"
        + table(
            ["X", "Y", "Cramér’s V", "Pairs"],
            [(p["x"], p["y"], p["cramers_v"], p["n"]) for p in r["associations"]],
        )
    )
    output.append(
        "<h2>Potential anomalies</h2>"
        + table(
            ["Column", "Method", "Threshold", "Lower", "Upper", "Count"],
            [
                (
                    p["name"],
                    p["outliers"]["method"],
                    p["outliers"]["threshold"],
                    p["outliers"]["lower"],
                    p["outliers"]["upper"],
                    p["outliers"]["count"],
                )
                for p in r["profiles"]
                if p["outliers"]
            ],
        )
    )
    output.append("<h2>Evidence-backed findings</h2>")
    for i in r["insights"]:
        output.append(
            '<div class="finding"><strong>'
            + display(i["title"])
            + "</strong><p>"
            + display(i["recommendation"])
            + "</p><p>"
            + display(i["evidence"]["metric"])
            + ": "
            + display(i["evidence"]["value"])
            + "</p></div>"
        )
    output.append(
        "<h2>Analysis configuration</h2><pre>"
        + display(json.dumps(r["config"], indent=2))
        + "</pre><h2>Methods and limitations</h2><pre>"
        + display(json.dumps(r["methodology"], indent=2))
        + "</pre></html>"
    )
    document = "".join(output)
    if sections is None:
        return document
    chosen = {REPORT_SECTIONS[key] for key in sections}
    # Split only trusted fixed headings; all uploaded content is already escaped.
    blocks = re.split(r"(?=<h2>)", document)
    return "".join(
        block
        for block in blocks
        if not (match := re.match(r"<h2>([^<]+)</h2>", block))
        or match[1] not in REPORT_SECTIONS.values()
        or match[1] in chosen
    )
