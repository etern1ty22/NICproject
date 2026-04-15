from __future__ import annotations

import ast
import csv
import json
import statistics
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable


def build_final_analysis(
    *,
    greedy_csv: str | Path,
    aco_csv: str | Path,
    sweep_csv: str | Path,
    ablation_csv: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    greedy_path = Path(greedy_csv)
    aco_path = Path(aco_csv)
    sweep_path = Path(sweep_csv)
    ablation_path = Path(ablation_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    greedy_rows = _read_run_records(greedy_path)
    aco_rows = _read_run_records(aco_path)
    sweep_rows = _read_run_records(sweep_path)
    ablation_rows = _read_run_records(ablation_path)

    greedy_summary = _summary_stats(greedy_rows)
    aco_summary = _summary_stats(aco_rows)
    comparison_rows = [
        _with_label("greedy_solver", greedy_summary),
        _with_label("aco_solver", aco_summary),
    ]

    multiseed_rows = [_with_label("aco_final_multiseed", aco_summary)]
    sweep_summary_rows = _summarize_sweep_rows(sweep_rows)
    ablation_summary_rows = _summarize_ablation_rows(ablation_rows)

    comparison_csv = output_path / "greedy_vs_aco_summary.csv"
    multiseed_csv = output_path / "final_multiseed_summary.csv"
    sweep_csv_out = output_path / "parameter_sweep_summary.csv"
    ablation_csv_out = output_path / "local_search_ablation_summary.csv"
    summary_md = output_path / "final_analysis_summary.md"
    manifest_json = output_path / "analysis_manifest.json"

    _write_csv(comparison_csv, comparison_rows)
    _write_csv(multiseed_csv, multiseed_rows)
    _write_csv(sweep_csv_out, sweep_summary_rows)
    _write_csv(ablation_csv_out, ablation_summary_rows)

    greedy_vs_aco_svg = output_path / "greedy_vs_aco_distance.svg"
    sweep_top_svg = output_path / "parameter_sweep_top5_distance.svg"
    ablation_svg = output_path / "local_search_ablation_distance.svg"

    _write_horizontal_bar_chart(
        greedy_vs_aco_svg,
        title="Greedy vs ACO Mean Distance",
        rows=comparison_rows,
        label_key="label",
        value_key="mean_distance",
    )
    _write_horizontal_bar_chart(
        sweep_top_svg,
        title="Top 5 Sweep Settings by Mean Distance",
        rows=sweep_summary_rows[:5],
        label_key="setting",
        value_key="mean_distance",
    )
    _write_horizontal_bar_chart(
        ablation_svg,
        title="Local Search Ablation Mean Distance",
        rows=ablation_summary_rows,
        label_key="operators",
        value_key="mean_distance",
    )

    delta_distance = aco_summary["mean_distance"] - greedy_summary["mean_distance"]
    delta_runtime = aco_summary["mean_runtime_s"] - greedy_summary["mean_runtime_s"]
    best_sweep = sweep_summary_rows[0] if sweep_summary_rows else None
    best_ablation = ablation_summary_rows[0] if ablation_summary_rows else None
    summary_md.write_text(
        _render_summary_markdown(
            greedy_summary=greedy_summary,
            aco_summary=aco_summary,
            delta_distance=delta_distance,
            delta_runtime=delta_runtime,
            best_sweep=best_sweep,
            best_ablation=best_ablation,
            sources={
                "greedy_csv": greedy_path,
                "aco_csv": aco_path,
                "sweep_csv": sweep_path,
                "ablation_csv": ablation_path,
            },
        ),
        encoding="utf-8",
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "greedy_csv": str(greedy_path),
            "aco_csv": str(aco_path),
            "sweep_csv": str(sweep_path),
            "ablation_csv": str(ablation_path),
        },
        "outputs": {
            "greedy_vs_aco_summary": str(comparison_csv),
            "final_multiseed_summary": str(multiseed_csv),
            "parameter_sweep_summary": str(sweep_csv_out),
            "local_search_ablation_summary": str(ablation_csv_out),
            "final_analysis_summary": str(summary_md),
            "greedy_vs_aco_distance_svg": str(greedy_vs_aco_svg),
            "parameter_sweep_top5_distance_svg": str(sweep_top_svg),
            "local_search_ablation_distance_svg": str(ablation_svg),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "comparison_csv": comparison_csv,
        "multiseed_csv": multiseed_csv,
        "sweep_csv": sweep_csv_out,
        "ablation_csv": ablation_csv_out,
        "summary_md": summary_md,
        "manifest_json": manifest_json,
        "greedy_vs_aco_svg": greedy_vs_aco_svg,
        "sweep_top_svg": sweep_top_svg,
        "ablation_svg": ablation_svg,
    }


def _read_run_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment artifact: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_rows.append(
            {
                **row,
                "seed": int(row["seed"]),
                "feasible": str(row["feasible"]).lower() == "true",
                "vehicles_used": int(float(row["vehicles_used"])),
                "distance": float(row["distance"]),
                "official_cost": float(row["official_cost"]),
                "runtime_s": float(row["runtime_s"]),
                "params": _parse_params(row.get("params", "")),
            }
        )
    return parsed_rows


def _parse_params(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one run record is required for analysis.")

    distances = [row["distance"] for row in rows]
    vehicles = [row["vehicles_used"] for row in rows]
    runtimes = [row["runtime_s"] for row in rows]
    feasible_count = sum(1 for row in rows if row["feasible"])

    return {
        "records": len(rows),
        "feasible_count": feasible_count,
        "feasible_rate": round(feasible_count / len(rows), 6),
        "mean_vehicles_used": round(statistics.mean(vehicles), 6),
        "mean_distance": round(statistics.mean(distances), 6),
        "best_distance": round(min(distances), 6),
        "worst_distance": round(max(distances), 6),
        "distance_std": round(statistics.pstdev(distances), 6) if len(distances) > 1 else 0.0,
        "mean_runtime_s": round(statistics.mean(runtimes), 6),
        "total_runtime_s": round(sum(runtimes), 6),
    }


def _with_label(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"label": label, **payload}


def _summarize_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        params = row["params"]
        key = (float(params["alpha"]), float(params["beta"]), float(params["rho"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (alpha, beta, rho), group in groups.items():
        summary = _summary_stats(group)
        summary_rows.append(
            {
                "setting": f"alpha={alpha:g}, beta={beta:g}, rho={rho:g}",
                "alpha": alpha,
                "beta": beta,
                "rho": rho,
                **summary,
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: (
            row["mean_vehicles_used"],
            row["mean_distance"],
            row["mean_runtime_s"],
        ),
    )


def _summarize_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        params = row["params"]
        operators = tuple(str(operator) for operator in params.get("local_search_operators", ()))
        groups.setdefault(operators, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for operators, group in groups.items():
        summary = _summary_stats(group)
        summary_rows.append(
            {
                "operators": _format_operators_label(operators),
                "operators_count": len(operators),
                **summary,
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: (
            row["mean_vehicles_used"],
            row["mean_distance"],
            row["mean_runtime_s"],
        ),
    )


def _format_operators_label(operators: Iterable[str]) -> str:
    normalized = tuple(operators)
    if not normalized:
        return "none"
    return "+".join(normalized)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty analysis table: {path}")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_horizontal_bar_chart(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
) -> None:
    if not rows:
        return

    width = 960
    row_height = 64
    header_height = 80
    footer_height = 30
    height = header_height + (row_height * len(rows)) + footer_height
    label_x = 20
    chart_left = 340
    chart_width = 520
    value_x = chart_left + chart_width + 20
    max_value = max(float(row[value_key]) for row in rows) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8" />',
        f'<text x="20" y="38" font-size="28" font-family="Arial, sans-serif" fill="#1f2937">{escape(title)}</text>',
        f'<text x="20" y="64" font-size="14" font-family="Arial, sans-serif" fill="#6b7280">Lower is better</text>',
    ]

    for index, row in enumerate(rows):
        y = header_height + (index * row_height)
        label = escape(str(row[label_key]))
        value = float(row[value_key])
        bar_width = 0 if max_value == 0 else (value / max_value) * chart_width
        parts.extend(
            [
                f'<text x="{label_x}" y="{y + 28}" font-size="15" font-family="Arial, sans-serif" fill="#111827">{label}</text>',
                f'<rect x="{chart_left}" y="{y + 8}" width="{chart_width}" height="24" rx="12" fill="#e5e7eb" />',
                f'<rect x="{chart_left}" y="{y + 8}" width="{bar_width:.2f}" height="24" rx="12" fill="#d97706" />',
                f'<text x="{value_x}" y="{y + 26}" font-size="14" font-family="Arial, sans-serif" fill="#111827">{value:.3f}</text>',
            ]
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _render_summary_markdown(
    *,
    greedy_summary: dict[str, Any],
    aco_summary: dict[str, Any],
    delta_distance: float,
    delta_runtime: float,
    best_sweep: dict[str, Any] | None,
    best_ablation: dict[str, Any] | None,
    sources: dict[str, Path],
) -> str:
    lines = [
        "# Final Analysis Summary",
        "",
        f"Generated from stable experiment artifacts on {datetime.now(timezone.utc).date().isoformat()} UTC.",
        "",
        "## Sources",
        "",
        f"- Greedy baseline: `{sources['greedy_csv']}`",
        f"- ACO final multiseed: `{sources['aco_csv']}`",
        f"- Parameter sweep: `{sources['sweep_csv']}`",
        f"- Local-search ablation: `{sources['ablation_csv']}`",
        "",
        "## Greedy vs ACO",
        "",
        f"- Greedy mean distance: `{greedy_summary['mean_distance']}` across `{greedy_summary['records']}` run(s).",
        f"- ACO mean distance: `{aco_summary['mean_distance']}` across `{aco_summary['records']}` run(s).",
        f"- Distance delta (`ACO - Greedy`): `{round(delta_distance, 6)}`.",
        f"- Runtime delta (`ACO - Greedy`, mean seconds): `{round(delta_runtime, 6)}`.",
        "",
        "## Final ACO Multiseed",
        "",
        f"- Feasible runs: `{aco_summary['feasible_count']}/{aco_summary['records']}`.",
        f"- Mean vehicles used: `{aco_summary['mean_vehicles_used']}`.",
        f"- Best distance: `{aco_summary['best_distance']}`.",
        f"- Worst distance: `{aco_summary['worst_distance']}`.",
        f"- Distance std: `{aco_summary['distance_std']}`.",
        "",
        "## Best Sweep Setting",
        "",
    ]

    if best_sweep is None:
        lines.append("- No sweep records were available.")
    else:
        lines.extend(
            [
                f"- Setting: `{best_sweep['setting']}`.",
                f"- Mean distance: `{best_sweep['mean_distance']}`.",
                f"- Mean vehicles used: `{best_sweep['mean_vehicles_used']}`.",
                f"- Mean runtime: `{best_sweep['mean_runtime_s']}` seconds.",
            ]
        )

    lines.extend(["", "## Best Local-Search Ablation", ""])
    if best_ablation is None:
        lines.append("- No ablation records were available.")
    else:
        lines.extend(
            [
                f"- Operators: `{best_ablation['operators']}`.",
                f"- Mean distance: `{best_ablation['mean_distance']}`.",
                f"- Mean vehicles used: `{best_ablation['mean_vehicles_used']}`.",
                f"- Mean runtime: `{best_ablation['mean_runtime_s']}` seconds.",
            ]
        )

    lines.append("")
    return "\n".join(lines)
