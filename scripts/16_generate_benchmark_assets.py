from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reportlab.graphics import renderPDF, renderPM, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "final_assets"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"


PALETTE = {
    "ink": colors.HexColor("#1F2937"),
    "muted": colors.HexColor("#6B7280"),
    "grid": colors.HexColor("#E5E7EB"),
    "bg": colors.HexColor("#FCFCFD"),
    "blue": colors.HexColor("#3B82F6"),
    "teal": colors.HexColor("#0F766E"),
    "green": colors.HexColor("#059669"),
    "amber": colors.HexColor("#D97706"),
    "coral": colors.HexColor("#E76F51"),
    "rose": colors.HexColor("#BE123C"),
    "graybar": colors.HexColor("#C7D2DA"),
    "grayfill": colors.HexColor("#E5E7EB"),
    "slate": colors.HexColor("#64748B"),
    "gold": colors.HexColor("#C2410C"),
}

ERROR_COLORS = {
    "clean_hit": PALETTE["green"],
    "generation_issue": PALETTE["amber"],
    "ranking_issue": PALETTE["blue"],
    "true_miss": PALETTE["rose"],
    "same_page_false_negative": PALETTE["slate"],
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    return payload["summary"]["overall"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _pct(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value) * 100.0


def _format_metric(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


def _write_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_table(
    path: Path,
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
) -> None:
    colspec = "l" * len(headers)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{_escape_latex(label)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(_escape_latex(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_escape_latex(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_drawing(drawing: Drawing, stem: str) -> list[str]:
    files: list[str] = []
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    renderSVG.drawToFile(drawing, str(svg_path))
    renderPDF.drawToFile(drawing, str(pdf_path))
    try:
        renderPM.drawToFile(drawing, str(png_path), fmt="PNG")
        files.extend([str(svg_path), str(pdf_path), str(png_path)])
    except Exception:
        files.extend([str(svg_path), str(pdf_path)])
    return files


def _blend_with_white(color: colors.Color, strength: float) -> colors.Color:
    strength = max(0.0, min(1.0, strength))
    r = 1.0 - (1.0 - color.red) * strength
    g = 1.0 - (1.0 - color.green) * strength
    b = 1.0 - (1.0 - color.blue) * strength
    return colors.Color(r, g, b)


def _add_title(d: Drawing, title: str, subtitle: str = "") -> None:
    height = float(d.height)
    d.add(String(28, height - 24, title, fontName="Times-Bold", fontSize=14.5, fillColor=PALETTE["ink"]))
    if subtitle:
        d.add(String(28, height - 39, subtitle, fontName="Times-Roman", fontSize=8.2, fillColor=PALETTE["muted"]))


def _draw_axes(
    d: Drawing,
    x0: float,
    y0: float,
    width: float,
    height: float,
    max_value: float,
    tick_step: float,
    x_label: str = "",
    y_label: str = "",
) -> None:
    d.add(Line(x0, y0, x0, y0 + height, strokeColor=PALETTE["ink"], strokeWidth=0.9))
    d.add(Line(x0, y0, x0 + width, y0, strokeColor=PALETTE["ink"], strokeWidth=0.9))
    tick = 0.0
    while tick <= max_value + 1e-6:
        y = y0 + (tick / max_value) * height if max_value > 0 else y0
        d.add(Line(x0, y, x0 + width, y, strokeColor=PALETTE["grid"], strokeWidth=0.7))
        d.add(String(x0 - 8, y - 3, f"{tick:.0f}", fontName="Times-Roman", fontSize=7.8, textAnchor="end", fillColor=PALETTE["muted"]))
        tick += tick_step
    if x_label:
        d.add(String(x0 + width / 2, y0 - 30, x_label, fontName="Times-Bold", fontSize=9, textAnchor="middle", fillColor=PALETTE["ink"]))
    if y_label:
        d.add(String(18, y0 + height / 2, y_label, fontName="Times-Bold", fontSize=9, fillColor=PALETTE["ink"], angle=90))


def _add_legend_wrapped(
    d: Drawing,
    items: list[tuple[str, colors.Color]],
    x: float,
    y: float,
    max_width: float,
    row_gap: float = 15,
) -> None:
    cursor_x = x
    cursor_y = y
    for label, color in items:
        item_width = 18 + len(label) * 4.7 + 14
        if cursor_x > x and cursor_x + item_width > x + max_width:
            cursor_x = x
            cursor_y += row_gap
        d.add(Rect(cursor_x, cursor_y, 10, 10, fillColor=color, strokeColor=color))
        d.add(String(cursor_x + 15, cursor_y + 1.5, label, fontName="Times-Roman", fontSize=8, fillColor=PALETTE["ink"]))
        cursor_x += item_width


def _doc_page_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
    doc_id = str(metadata.get("ucsf_document_id", ""))
    page_no = str(metadata.get("ucsf_document_page_no", ""))
    return f"{doc_id}|{page_no}"


def _question_types(sample: dict[str, Any]) -> list[str]:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
    raw_value = metadata.get("question_types", [])
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if raw_value in (None, ""):
        return []
    return [str(raw_value)]


def _strict_flags(citations: list[dict[str, Any]], sample_id: str) -> list[bool]:
    prefix = f"docvqa/val/{sample_id}"
    return [str(citation.get("source_ref") or citation.get("source") or "").startswith(prefix) for citation in citations]


def _sample_id_from_source_ref(source_ref: str) -> str:
    if not source_ref.startswith("docvqa/val/"):
        return ""
    return source_ref.split("/", 2)[2].split("#", 1)[0]


def _docpage_flags(
    citations: list[dict[str, Any]],
    sample_to_docpage: dict[str, str],
    expected_docpage: str,
) -> list[bool]:
    flags: list[bool] = []
    for citation in citations:
        source_ref = str(citation.get("source_ref") or citation.get("source") or "")
        sample_id = _sample_id_from_source_ref(source_ref)
        flags.append(sample_to_docpage.get(sample_id, "") == expected_docpage)
    return flags


def _first_true_rank(flags: list[bool]) -> int | None:
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return idx
    return None


def analyze_error_run(samples_path: Path, rag_run_path: Path, limit: int = 100) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_rows = _read_jsonl(samples_path)[:limit]
    rag_rows = _read_jsonl(rag_run_path)
    sample_by_id = {str(row["id"]): row for row in sample_rows}
    sample_to_docpage = {str(row["id"]): _doc_page_key(row) for row in sample_rows}

    analyzed_rows: list[dict[str, Any]] = []
    for rag_row in rag_rows:
        sample_id = str(rag_row["sample_id"])
        sample_row = sample_by_id.get(sample_id)
        if sample_row is None:
            continue

        citations = rag_row.get("citations", [])
        metrics = rag_row.get("metrics", {})
        expected_docpage = sample_to_docpage[sample_id]

        sample_strict_flags = _strict_flags(citations, sample_id)
        sample_docpage_flags = _docpage_flags(citations, sample_to_docpage, expected_docpage)

        strict_hit = any(sample_strict_flags)
        strict_top1 = bool(sample_strict_flags[0]) if sample_strict_flags else False
        docpage_hit = any(sample_docpage_flags)
        docpage_top1 = bool(sample_docpage_flags[0]) if sample_docpage_flags else False
        strict_rank = _first_true_rank(sample_strict_flags)
        docpage_rank = _first_true_rank(sample_docpage_flags)

        exact_match = float(metrics.get("exact_match", 0.0))
        answer_contains = float(metrics.get("answer_contains", 0.0))
        token_f1 = float(metrics.get("token_f1", 0.0))
        anls = float(metrics.get("anls", 0.0))

        tags: list[str] = []
        if not strict_hit and docpage_hit:
            tags.append("same_page_false_negative")
        if strict_hit and not strict_top1:
            tags.append("strict_ranking_issue")
        if docpage_hit and not docpage_top1:
            tags.append("docpage_ranking_issue")
        if strict_hit and exact_match < 1.0:
            tags.append("generation_after_strict_hit")
        if docpage_hit and exact_match < 1.0:
            tags.append("generation_after_docpage_hit")
        if exact_match < 1.0 and answer_contains >= 1.0:
            tags.append("normalization_sensitive")
        if not docpage_hit:
            tags.append("true_retrieval_miss")

        if not strict_hit and docpage_hit:
            primary_category = "same_page_false_negative"
        elif not docpage_hit:
            primary_category = "true_miss"
        elif strict_hit and not strict_top1:
            primary_category = "ranking_issue"
        elif strict_hit and exact_match < 1.0:
            primary_category = "generation_issue"
        else:
            primary_category = "clean_hit"

        analyzed_rows.append(
            {
                "sample_id": sample_id,
                "question_types": "|".join(_question_types(sample_row)),
                "strict_hit": int(strict_hit),
                "strict_top1": int(strict_top1),
                "strict_first_relevant_rank": strict_rank or "",
                "docpage_hit": int(docpage_hit),
                "docpage_top1": int(docpage_top1),
                "docpage_first_relevant_rank": docpage_rank or "",
                "exact_match": exact_match,
                "answer_contains": answer_contains,
                "token_f1": token_f1,
                "anls": anls,
                "primary_category": primary_category,
                "tags": "|".join(tags),
            }
        )

    category_counts = Counter(str(row["primary_category"]) for row in analyzed_rows)
    error_category_counts = Counter(str(row["primary_category"]) for row in analyzed_rows if str(row["primary_category"]) != "clean_hit")
    tag_counts = Counter()
    by_question_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in analyzed_rows:
        for tag in filter(None, str(row["tags"]).split("|")):
            tag_counts[tag] += 1
        sample_types = [item for item in str(row["question_types"]).split("|") if item] or ["unknown"]
        for sample_type in sample_types:
            by_question_type[sample_type][str(row["primary_category"])] += 1

    summary = {
        "num_samples": len(analyzed_rows),
        "category_counts": dict(category_counts),
        "error_category_counts": dict(error_category_counts),
        "tag_counts": dict(tag_counts),
        "by_question_type": {key: dict(counter) for key, counter in sorted(by_question_type.items())},
    }
    return analyzed_rows, summary


def build_repair_chart(repair_stats: dict[str, int]) -> list[str]:
    width, height = 540, 300
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PALETTE["bg"], strokeColor=PALETTE["bg"]))
    _add_title(d, "Benchmark Repair", "Duplicate-page contamination is removed in the fixed benchmark.")

    x0, y0, plot_w, plot_h = 76, 72, 408, 172
    _draw_axes(d, x0, y0, plot_w, plot_h, 100.0, 20.0, y_label="Questions")
    _add_legend_wrapped(
        d,
        [("Unique pages", PALETTE["blue"]), ("Duplicate-page questions", PALETTE["grayfill"])],
        x0,
        24,
        plot_w,
    )

    categories = [
        ("Original", repair_stats["original_unique"], repair_stats["original_duplicate"]),
        ("Unique-docpage-100", repair_stats["repaired_unique"], repair_stats["repaired_duplicate"]),
    ]
    bar_w = 96
    gap = 120
    start_x = x0 + 54
    for idx, (label, unique_count, duplicate_count) in enumerate(categories):
        x = start_x + idx * (bar_w + gap)
        unique_h = (unique_count / 100.0) * plot_h
        duplicate_h = (duplicate_count / 100.0) * plot_h
        d.add(Rect(x, y0, bar_w, unique_h, fillColor=PALETTE["blue"], strokeColor=PALETTE["blue"]))
        d.add(Rect(x, y0 + unique_h, bar_w, duplicate_h, fillColor=PALETTE["grayfill"], strokeColor=PALETTE["grayfill"]))
        d.add(String(x + bar_w / 2, y0 - 18, label, fontName="Times-Bold", fontSize=8.4, textAnchor="middle", fillColor=PALETTE["ink"]))
        d.add(String(x + bar_w / 2, y0 + unique_h + duplicate_h + 6, f"{unique_count}/{unique_count + duplicate_count}", fontName="Times-Roman", fontSize=8, textAnchor="middle", fillColor=PALETTE["muted"]))
    return _save_drawing(d, "fig_benchmark_repair")


def build_grouped_bar_chart(
    stem: str,
    title: str,
    series: list[dict[str, Any]],
    metrics: list[tuple[str, str]],
    legend_items: list[tuple[str, colors.Color]],
    show_value_labels: bool = False,
    label_series_start: int = 0,
) -> list[str]:
    width, height = 780, 356
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PALETTE["bg"], strokeColor=PALETTE["bg"]))
    _add_title(d, title)

    x0, y0, plot_w, plot_h = 78, 90, 648, 198
    _draw_axes(d, x0, y0, plot_w, plot_h, 100.0, 20.0, y_label="Score (%)")
    _add_legend_wrapped(d, legend_items, x0, 24, plot_w)

    cluster_gap = plot_w / max(1, len(metrics))
    bar_w = min(18.0, (cluster_gap - 26) / max(1, len(series)))
    for metric_idx, (metric_key, metric_label) in enumerate(metrics):
        center = x0 + cluster_gap * (metric_idx + 0.5)
        cluster_width = len(series) * bar_w + (len(series) - 1) * 6
        cluster_start = center - cluster_width / 2
        for series_idx, row in enumerate(series):
            x = cluster_start + series_idx * (bar_w + 6)
            value = _pct(row[metric_key])
            h = (value / 100.0) * plot_h
            visible_h = h if h > 0 else 0.9
            d.add(Rect(x, y0, bar_w, visible_h, fillColor=row["color"], strokeColor=row["color"]))
            if show_value_labels and series_idx >= label_series_start:
                y_label = y0 + visible_h + 5 + (series_idx - label_series_start) * 8
                d.add(
                    String(
                        x + bar_w / 2,
                        y_label,
                        f"{value:.1f}",
                        fontName="Times-Roman",
                        fontSize=7.2,
                        textAnchor="middle",
                        fillColor=PALETTE["muted"],
                    )
                )
        d.add(String(center, y0 - 18, metric_label, fontName="Times-Bold", fontSize=8.5, textAnchor="middle", fillColor=PALETTE["ink"]))
    return _save_drawing(d, stem)


def build_tradeoff_scatter(rows: list[dict[str, Any]]) -> list[str]:
    width, height = 640, 340
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PALETTE["bg"], strokeColor=PALETTE["bg"]))
    _add_title(d, "Grounding-Quality Frontier")

    x0, y0, plot_w, plot_h = 80, 78, 504, 196
    x_min, x_max = 85.0, 91.5
    y_min, y_max = 87.0, 97.0
    d.add(Line(x0, y0, x0, y0 + plot_h, strokeColor=PALETTE["ink"], strokeWidth=0.9))
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=PALETTE["ink"], strokeWidth=0.9))
    for x_tick in [85, 87, 89, 91]:
        x = x0 + ((x_tick - x_min) / (x_max - x_min)) * plot_w
        d.add(Line(x, y0, x, y0 + plot_h, strokeColor=PALETTE["grid"], strokeWidth=0.7))
        d.add(String(x, y0 - 16, f"{x_tick:.0f}", fontName="Times-Roman", fontSize=7.8, textAnchor="middle", fillColor=PALETTE["muted"]))
    for y_tick in [88, 90, 92, 94, 96]:
        y = y0 + ((y_tick - y_min) / (y_max - y_min)) * plot_h
        d.add(Line(x0, y, x0 + plot_w, y, strokeColor=PALETTE["grid"], strokeWidth=0.7))
        d.add(String(x0 - 8, y - 3, f"{y_tick:.0f}", fontName="Times-Roman", fontSize=7.8, textAnchor="end", fillColor=PALETTE["muted"]))
    d.add(String(x0 + plot_w / 2, y0 - 30, "Citation@1 (%)", fontName="Times-Bold", fontSize=9, textAnchor="middle", fillColor=PALETTE["ink"]))
    d.add(String(18, y0 + plot_h / 2, "ANLS (%)", fontName="Times-Bold", fontSize=9, fillColor=PALETTE["ink"], angle=90))

    label_offsets = {
        "Base": (8, -10),
        "Old": (8, 8),
        "QIA": (-24, 9),
        "QIA+VA": (8, 8),
    }
    for row in rows:
        x = x0 + ((_pct(row["citation_accuracy"]) - x_min) / (x_max - x_min)) * plot_w
        y = y0 + ((_pct(row["anls"]) - y_min) / (y_max - y_min)) * plot_h
        radius = max(5.0, 4.5 + (_pct(row["exact_match"]) - 75.0) * 0.20)
        d.add(Circle(x, y, radius, fillColor=row["color"], strokeColor=colors.white, strokeWidth=1.0))
        dx, dy = label_offsets.get(str(row["label"]), (8, 8))
        d.add(String(x + dx, y + dy, str(row["label"]), fontName="Times-Roman", fontSize=8, fillColor=PALETTE["ink"]))
    return _save_drawing(d, "fig_grounding_vs_quality_tradeoff")


def build_question_type_chart(question_type_counts: dict[str, int]) -> list[str]:
    items = sorted(question_type_counts.items(), key=lambda item: (-item[1], item[0]))
    width, height = 620, 360
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PALETTE["bg"], strokeColor=PALETTE["bg"]))
    _add_title(d, "Appendix: Question-Type Mix", "Unique-docpage-100 manifest composition.")

    x0, y0, plot_w, plot_h = 156, 52, 398, 248
    max_value = float(max(question_type_counts.values())) if question_type_counts else 1.0
    d.add(Line(x0, y0, x0, y0 + plot_h, strokeColor=PALETTE["ink"], strokeWidth=0.9))
    d.add(Line(x0, y0, x0 + plot_w, y0, strokeColor=PALETTE["ink"], strokeWidth=0.9))
    for tick in range(0, int(max_value) + 1, 2):
        x = x0 + (tick / max_value) * plot_w if max_value > 0 else x0
        d.add(Line(x, y0, x, y0 + plot_h, strokeColor=PALETTE["grid"], strokeWidth=0.7))
        d.add(String(x, y0 - 15, str(tick), fontName="Times-Roman", fontSize=7.8, textAnchor="middle", fillColor=PALETTE["muted"]))
    row_gap = plot_h / max(1, len(items))
    bar_h = min(18.0, row_gap * 0.62)
    for idx, (label, count) in enumerate(items):
        y = y0 + plot_h - (idx + 0.8) * row_gap
        w = (count / max_value) * plot_w if max_value > 0 else 0
        d.add(String(x0 - 10, y + 3, label, fontName="Times-Roman", fontSize=8.3, textAnchor="end", fillColor=PALETTE["ink"]))
        d.add(Rect(x0, y, w, bar_h, fillColor=PALETTE["teal"], strokeColor=PALETTE["teal"]))
        d.add(String(x0 + w + 8, y + 3, str(count), fontName="Times-Roman", fontSize=8, fillColor=PALETTE["muted"]))
    d.add(String(x0 + plot_w / 2, 18, "Count", fontName="Times-Bold", fontSize=9, textAnchor="middle", fillColor=PALETTE["ink"]))
    return _save_drawing(d, "fig_appendix_question_type_mix")


def build_error_by_type_chart(by_question_type: dict[str, dict[str, int]]) -> list[str]:
    items: list[tuple[str, dict[str, int], int]] = []
    for label, counts in by_question_type.items():
        total = sum(int(value) for value in counts.values())
        items.append((label, counts, total))
    items.sort(key=lambda item: (-item[2], item[0]))

    categories = [
        ("support", "Support"),
        ("clean_hit", "Clean"),
        ("generation_issue", "Gen"),
        ("ranking_issue", "Rank"),
        ("true_miss", "Miss"),
    ]
    width, height = 720, 412
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PALETTE["bg"], strokeColor=PALETTE["bg"]))
    _add_title(d, "Error Stratification by Question Type")

    x_name = 34
    x0, y0 = 168, 46
    cell_w = 94
    cell_h = 28
    header_h = 24
    total_w = cell_w * len(categories)
    total_h = header_h + cell_h * len(items)

    d.add(String(x0, y0 + total_h + 12, "Cell color indicates row-normalized share; number indicates count.", fontName="Times-Roman", fontSize=8, fillColor=PALETTE["muted"]))

    for col_idx, (_, header) in enumerate(categories):
        x = x0 + col_idx * cell_w
        d.add(Rect(x, y0 + total_h - header_h, cell_w, header_h, fillColor=PALETTE["grayfill"], strokeColor=colors.white, strokeWidth=1))
        d.add(String(x + cell_w / 2, y0 + total_h - header_h + 7, header, fontName="Times-Bold", fontSize=8.5, textAnchor="middle", fillColor=PALETTE["ink"]))

    for row_idx, (label, counts, total) in enumerate(items):
        y = y0 + total_h - header_h - (row_idx + 1) * cell_h
        d.add(String(x0 - 12, y + 9, label, fontName="Times-Roman", fontSize=8.4, textAnchor="end", fillColor=PALETTE["ink"]))
        for col_idx, (category, _) in enumerate(categories):
            x = x0 + col_idx * cell_w
            if category == "support":
                fill = PALETTE["grayfill"]
                text = str(total)
                text_color = PALETTE["ink"]
            else:
                value = int(counts.get(category, 0))
                frac = (value / total) if total > 0 else 0.0
                fill = _blend_with_white(ERROR_COLORS[category], 0.18 + 0.82 * frac)
                text = str(value)
                text_color = PALETTE["ink"] if frac < 0.62 else colors.white
            d.add(Rect(x, y, cell_w, cell_h, fillColor=fill, strokeColor=colors.white, strokeWidth=1))
            d.add(String(x + cell_w / 2, y + 9, text, fontName="Times-Roman", fontSize=8.3, textAnchor="middle", fillColor=text_color))

    d.add(Rect(x0, y0, total_w, total_h, fillColor=None, strokeColor=PALETTE["grid"], strokeWidth=0.8))
    return _save_drawing(d, "fig_appendix_error_by_question_type")


def build_error_overview_chart(category_counts: dict[str, int]) -> list[str]:
    categories = [
        ("clean_hit", "Clean"),
        ("generation_issue", "Generation"),
        ("ranking_issue", "Ranking"),
        ("true_miss", "True miss"),
    ]
    width, height = 560, 320
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=PALETTE["bg"], strokeColor=PALETTE["bg"]))
    _add_title(d, "Appendix: Final Error Overview", "Final gated visual-assist mainline, n=100.")

    x0, y0, plot_w, plot_h = 72, 68, 430, 180
    _draw_axes(d, x0, y0, plot_w, plot_h, 80.0, 20.0, y_label="Count")
    _add_legend_wrapped(
        d,
        [(label, ERROR_COLORS[key]) for key, label in categories],
        x0,
        24,
        plot_w,
    )
    bar_w = 64
    gap = 34
    start_x = x0 + 18
    for idx, (key, label) in enumerate(categories):
        x = start_x + idx * (bar_w + gap)
        value = int(category_counts.get(key, 0))
        h = (value / 80.0) * plot_h
        d.add(Rect(x, y0, bar_w, h, fillColor=ERROR_COLORS[key], strokeColor=ERROR_COLORS[key]))
        d.add(String(x + bar_w / 2, y0 - 18, label, fontName="Times-Bold", fontSize=8.4, textAnchor="middle", fillColor=PALETTE["ink"]))
        d.add(String(x + bar_w / 2, y0 + h + 6, str(value), fontName="Times-Roman", fontSize=8, textAnchor="middle", fillColor=PALETTE["muted"]))
    return _save_drawing(d, "fig_appendix_error_overview")


def build_tables(
    data: dict[str, dict[str, Any]],
    manifest_summary: dict[str, Any],
    current_error_summary: dict[str, Any],
) -> list[str]:
    files: list[str] = []

    main_rows = [
        {
            "line": "promptfix_v3",
            "benchmark": "original val-100",
            "type": "retrieval",
            "hit@5": _format_metric(data["orig_promptfix_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["orig_promptfix_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["orig_promptfix_retr"]["citation_accuracy"]),
            "EM": "-",
            "ANLS": "-",
            "Token-F1": "-",
        },
        {
            "line": "exp5_densefusion_w045",
            "benchmark": "original val-100",
            "type": "retrieval",
            "hit@5": _format_metric(data["orig_exp5_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["orig_exp5_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["orig_exp5_retr"]["citation_accuracy"]),
            "EM": "-",
            "ANLS": "-",
            "Token-F1": "-",
        },
        {
            "line": "stronger_qia_v6",
            "benchmark": "unique-docpage-100 rebuilt",
            "type": "retrieval",
            "hit@5": _format_metric(data["repaired_qia_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["repaired_qia_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["repaired_qia_retr"]["citation_accuracy"]),
            "EM": "-",
            "ANLS": "-",
            "Token-F1": "-",
        },
        {
            "line": "stronger_qia_base",
            "benchmark": "unique-docpage-100 rebuilt",
            "type": "rag",
            "hit@5": _format_metric(data["repaired_qia_rag_base"]["hit_at_k"]),
            "precision@5": _format_metric(data["repaired_qia_rag_base"]["precision_at_k"]),
            "citation@1": _format_metric(data["repaired_qia_rag_base"]["citation_accuracy"]),
            "EM": _format_metric(data["repaired_qia_rag_base"]["exact_match"]),
            "ANLS": _format_metric(data["repaired_qia_rag_base"]["anls"], digits=6),
            "Token-F1": _format_metric(data["repaired_qia_rag_base"]["token_f1"], digits=6),
        },
        {
            "line": "stronger_qia_visualassist_gated_v2",
            "benchmark": "unique-docpage-100 rebuilt",
            "type": "rag",
            "hit@5": _format_metric(data["repaired_qia_rag_visual_gated"]["hit_at_k"]),
            "precision@5": _format_metric(data["repaired_qia_rag_visual_gated"]["precision_at_k"]),
            "citation@1": _format_metric(data["repaired_qia_rag_visual_gated"]["citation_accuracy"]),
            "EM": _format_metric(data["repaired_qia_rag_visual_gated"]["exact_match"]),
            "ANLS": _format_metric(data["repaired_qia_rag_visual_gated"]["anls"], digits=6),
            "Token-F1": _format_metric(data["repaired_qia_rag_visual_gated"]["token_f1"], digits=6),
        },
    ]
    fieldnames = list(main_rows[0].keys())
    csv_path = OUTPUT_DIR / "table_closeout_main_results.csv"
    _write_csv(csv_path, main_rows, fieldnames)
    files.append(str(csv_path))
    md_rows = [[str(row[key]) for key in fieldnames] for row in main_rows]
    md_path = OUTPUT_DIR / "table_closeout_main_results.md"
    _write_markdown_table(md_path, fieldnames, md_rows)
    files.append(str(md_path))
    tex_path = OUTPUT_DIR / "table_closeout_main_results.tex"
    _write_latex_table(
        tex_path,
        fieldnames,
        md_rows,
        caption="Closeout benchmark summary across historical and repaired benchmark lines.",
        label="tab:closeout-main-results",
    )
    files.append(str(tex_path))

    retrieval_rows = [
        {
            "line": "promptfix_v3",
            "benchmark": "original val-100",
            "hit@5": _format_metric(data["orig_promptfix_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["orig_promptfix_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["orig_promptfix_retr"]["citation_accuracy"]),
        },
        {
            "line": "exp5_densefusion_w045",
            "benchmark": "original val-100",
            "hit@5": _format_metric(data["orig_exp5_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["orig_exp5_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["orig_exp5_retr"]["citation_accuracy"]),
        },
        {
            "line": "stronger_rebuilt_top1v2",
            "benchmark": "unique-docpage-100 rebuilt",
            "hit@5": _format_metric(data["repaired_top1v2_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["repaired_top1v2_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["repaired_top1v2_retr"]["citation_accuracy"]),
        },
        {
            "line": "stronger_qia_v6",
            "benchmark": "unique-docpage-100 rebuilt",
            "hit@5": _format_metric(data["repaired_qia_retr"]["hit_at_k"]),
            "precision@5": _format_metric(data["repaired_qia_retr"]["precision_at_k"]),
            "citation@1": _format_metric(data["repaired_qia_retr"]["citation_accuracy"]),
        },
    ]
    retrieval_fields = list(retrieval_rows[0].keys())
    r_csv = OUTPUT_DIR / "table_retrieval_progression.csv"
    _write_csv(r_csv, retrieval_rows, retrieval_fields)
    files.append(str(r_csv))
    r_tex = OUTPUT_DIR / "table_retrieval_progression.tex"
    _write_latex_table(
        r_tex,
        retrieval_fields,
        [[str(row[key]) for key in retrieval_fields] for row in retrieval_rows],
        caption="Retrieval progression from the historical benchmark to the repaired benchmark mainline.",
        label="tab:retrieval-progression",
    )
    files.append(str(r_tex))

    appendix_rows: list[dict[str, Any]] = []
    for question_type, counts in sorted(current_error_summary["by_question_type"].items()):
        support = sum(int(value) for value in counts.values())
        appendix_rows.append(
            {
                "question_type": question_type,
                "primary_count": str(manifest_summary["question_type_counts"].get(question_type, 0)),
                "multi_label_support": str(support),
                "clean_hit": str(counts.get("clean_hit", 0)),
                "generation_issue": str(counts.get("generation_issue", 0)),
                "ranking_issue": str(counts.get("ranking_issue", 0)),
                "true_miss": str(counts.get("true_miss", 0)),
            }
        )
    appendix_fields = list(appendix_rows[0].keys())
    appendix_csv = OUTPUT_DIR / "table_appendix_error_by_type.csv"
    _write_csv(appendix_csv, appendix_rows, appendix_fields)
    files.append(str(appendix_csv))
    appendix_tex = OUTPUT_DIR / "table_appendix_error_by_type.tex"
    _write_latex_table(
        appendix_tex,
        appendix_fields,
        [[str(row[key]) for key in appendix_fields] for row in appendix_rows],
        caption="Question-type composition and error breakdown for the final gated visual-assist mainline.",
        label="tab:appendix-error-by-type",
    )
    files.append(str(appendix_tex))
    return files


def _write_readme(manifest: dict[str, Any]) -> str:
    retrieval = manifest["closeout_mainline"]["retrieval"]
    rag = manifest["closeout_mainline"]["rag"]
    lines = [
        "# Final Benchmark Assets",
        "",
        "This directory contains publication-style figures and tables generated from the current benchmark mainline.",
        "",
        "## Reproducibility",
        "",
        "Run the following command from the project root:",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe scripts/16_generate_benchmark_assets.py",
        "```",
        "",
        "## Current Closeout Mainline",
        "",
        f"- Retrieval: `{retrieval['run']}`",
        f"  - Hit@5 = {retrieval['metrics']['hit_at_k']:.3f}, Precision@5 = {retrieval['metrics']['precision_at_k']:.3f}, Citation@1 = {retrieval['metrics']['citation_accuracy']:.3f}",
        f"- RAG: `{rag['run']}`",
        f"  - Hit@5 = {rag['metrics']['hit_at_k']:.3f}, Precision@5 = {rag['metrics']['precision_at_k']:.3f}, Citation@1 = {rag['metrics']['citation_accuracy']:.3f}, EM = {rag['metrics']['exact_match']:.3f}, ANLS = {rag['metrics']['anls']:.4f}, Token-F1 = {rag['metrics']['token_f1']:.5f}",
        "",
        "## Main Figures",
        "",
        "- `fig_benchmark_repair`: benchmark repair justification.",
        "- `fig_retrieval_progression`: retrieval progression across major lines.",
        "- `fig_rag_closeout_comparison`: end-to-end RAG comparison.",
        "- `fig_grounding_vs_quality_tradeoff`: model-selection frontier.",
        "",
        "## Appendix Figures",
        "",
        "- `fig_appendix_question_type_mix`: question-type composition of the fixed manifest.",
        "- `fig_appendix_error_overview`: final error distribution under the mainline.",
        "- `fig_appendix_error_by_question_type`: per-type error stratification.",
        "",
        "## Tables",
        "",
        "- `table_closeout_main_results.*`: final closeout table.",
        "- `table_retrieval_progression.*`: retrieval-only progression table.",
        "- `table_appendix_error_by_type.*`: appendix table for question types and error counts.",
        "  - `primary_count` is the manifest's primary-type count; `multi_label_support` is the support used in the error stratification table.",
        "",
        "## Benchmark Repair Summary",
        "",
        f"- Original val-100: {manifest['repair_stats']['original_total']} questions, {manifest['repair_stats']['original_unique']} unique doc-pages, {manifest['repair_stats']['original_duplicate']} duplicate-page questions.",
        f"- Repaired unique-docpage-100: {manifest['repair_stats']['repaired_unique']} unique doc-pages, {manifest['repair_stats']['repaired_duplicate']} duplicate-page questions.",
        "",
        "## Final Error Snapshot",
        "",
        f"- Clean hits: {manifest['current_error_summary']['category_counts'].get('clean_hit', 0)}",
        f"- Generation issues: {manifest['current_error_summary']['category_counts'].get('generation_issue', 0)}",
        f"- Ranking issues: {manifest['current_error_summary']['category_counts'].get('ranking_issue', 0)}",
        f"- True misses: {manifest['current_error_summary']['category_counts'].get('true_miss', 0)}",
        "",
    ]
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return str(readme_path)


def build_assets() -> dict[str, Any]:
    _ensure_output_dir()

    orig_promptfix_retr = _load_summary(EVAL_DIR / "docvqa_val_100_retrieval_promptfix_v3.summary.json")
    orig_exp5_retr = _load_summary(EVAL_DIR / "docvqa_val_100_retrieval_exp5_densefusion_w045.summary.json")
    orig_promptfix_rag = _load_summary(EVAL_DIR / "docvqa_val_100_rag_promptfix_v3.summary.json")
    orig_exp5_rag = _load_summary(EVAL_DIR / "docvqa_val_100_rag_exp5_densefusion_w045.summary.json")
    repaired_top1v2 = _load_summary(EVAL_DIR / "docvqa_val_unique_docpage_100_retrieval_stronger_rebuilt_top1v2.summary.json")
    repaired_qia = _load_summary(EVAL_DIR / "docvqa_val_unique_docpage_100_retrieval_stronger_qia_v6.summary.json")
    repaired_rag_base = _load_summary(EVAL_DIR / "docvqa_val_unique_docpage_100_rag_stronger_qia_base.summary.json")
    repaired_rag_visual_gated = _load_summary(EVAL_DIR / "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_gated_v2.summary.json")
    manifest_summary = _load_json(EVAL_DIR / "docvqa_val_unique_docpage_100.manifest.summary.json")

    current_error_rows, current_error_summary = analyze_error_run(
        EVAL_DIR / "docvqa_val_unique_docpage_100.manifest.jsonl",
        EVAL_DIR / "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_gated_v2.jsonl",
        limit=100,
    )
    current_error_json = OUTPUT_DIR / "appendix_current_error_summary.json"
    current_error_json.write_text(json.dumps(current_error_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rows = []
    with (PROJECT_ROOT / "data" / "processed" / "docvqa" / "val.jsonl").open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= 100:
                break
            rows.append(json.loads(line))
    original_total = len(rows)
    original_unique = len(
        {
            f"{(row.get('metadata') or {}).get('ucsf_document_id', '')}|{(row.get('metadata') or {}).get('ucsf_document_page_no', '')}"
            for row in rows
        }
    )
    repair_stats = {
        "original_total": original_total,
        "original_unique": original_unique,
        "original_duplicate": original_total - original_unique,
        "repaired_unique": 100,
        "repaired_duplicate": 0,
    }

    data = {
        "orig_promptfix_retr": orig_promptfix_retr,
        "orig_exp5_retr": orig_exp5_retr,
        "orig_promptfix_rag": orig_promptfix_rag,
        "orig_exp5_rag": orig_exp5_rag,
        "repaired_top1v2_retr": repaired_top1v2,
        "repaired_qia_retr": repaired_qia,
        "repaired_qia_rag_base": repaired_rag_base,
        "repaired_qia_rag_visual_gated": repaired_rag_visual_gated,
    }

    produced_files: list[str] = []
    produced_files.extend(build_repair_chart(repair_stats))
    produced_files.extend(
        build_grouped_bar_chart(
            stem="fig_retrieval_progression",
            title="Retrieval Progression",
            series=[
                {"color": PALETTE["graybar"], **orig_promptfix_retr},
                {"color": PALETTE["amber"], **orig_exp5_retr},
                {"color": PALETTE["teal"], **repaired_top1v2},
                {"color": PALETTE["coral"], **repaired_qia},
            ],
            metrics=[("hit_at_k", "Hit@5"), ("precision_at_k", "Precision@5"), ("citation_accuracy", "Citation@1")],
            legend_items=[
                ("PromptFix", PALETTE["graybar"]),
                ("Old VF", PALETTE["amber"]),
                ("Top1v2", PALETTE["teal"]),
                ("Final retrieval", PALETTE["coral"]),
            ],
            show_value_labels=True,
            label_series_start=0,
        )
    )
    produced_files.extend(
        build_grouped_bar_chart(
            stem="fig_rag_closeout_comparison",
            title="RAG Closeout Comparison",
            series=[
                {"color": PALETTE["graybar"], **orig_promptfix_rag},
                {"color": PALETTE["amber"], **orig_exp5_rag},
                {"color": PALETTE["teal"], **repaired_rag_base},
                {"color": PALETTE["coral"], **repaired_rag_visual_gated},
            ],
            metrics=[
                ("citation_accuracy", "Citation@1"),
                ("exact_match", "EM"),
                ("anls", "ANLS"),
                ("answer_contains", "Contain"),
                ("token_f1", "Token-F1"),
            ],
            legend_items=[
                ("PromptFix", PALETTE["graybar"]),
                ("Old VF", PALETTE["amber"]),
                ("QIA base", PALETTE["teal"]),
                ("QIA + visual assist", PALETTE["coral"]),
            ],
            show_value_labels=True,
            label_series_start=0,
        )
    )
    produced_files.extend(
        build_tradeoff_scatter(
            rows=[
                {"label": "Base", "color": PALETTE["graybar"], **orig_promptfix_rag},
                {"label": "Old", "color": PALETTE["amber"], **orig_exp5_rag},
                {"label": "QIA", "color": PALETTE["teal"], **repaired_rag_base},
                {"label": "QIA+VA", "color": PALETTE["coral"], **repaired_rag_visual_gated},
            ]
        )
    )
    produced_files.extend(build_question_type_chart(manifest_summary["question_type_counts"]))
    produced_files.extend(build_error_overview_chart(current_error_summary["category_counts"]))
    produced_files.extend(build_error_by_type_chart(current_error_summary["by_question_type"]))
    produced_files.extend(build_tables(data, manifest_summary, current_error_summary))
    produced_files.append(str(current_error_json))

    manifest = {
        "output_dir": str(OUTPUT_DIR),
        "figures_and_tables": produced_files,
        "repair_stats": repair_stats,
        "manifest_summary": manifest_summary,
        "current_error_summary": current_error_summary,
        "current_error_rows": len(current_error_rows),
        "closeout_mainline": {
            "retrieval": {
                "run": "docvqa_val_unique_docpage_100_retrieval_stronger_qia_v6",
                "metrics": {
                    "hit_at_k": repaired_qia["hit_at_k"],
                    "precision_at_k": repaired_qia["precision_at_k"],
                    "citation_accuracy": repaired_qia["citation_accuracy"],
                },
            },
            "rag": {
                "run": "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_gated_v2",
                "metrics": {
                    "hit_at_k": repaired_rag_visual_gated["hit_at_k"],
                    "precision_at_k": repaired_rag_visual_gated["precision_at_k"],
                    "citation_accuracy": repaired_rag_visual_gated["citation_accuracy"],
                    "exact_match": repaired_rag_visual_gated["exact_match"],
                    "anls": repaired_rag_visual_gated["anls"],
                    "answer_contains": repaired_rag_visual_gated["answer_contains"],
                    "token_f1": repaired_rag_visual_gated["token_f1"],
                },
            },
        },
    }
    manifest_path = OUTPUT_DIR / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    produced_files.append(str(manifest_path))
    produced_files.append(_write_readme(manifest))
    manifest["figures_and_tables"] = produced_files
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_assets()
    print(f"[benchmark-assets] output_dir={manifest['output_dir']}")
    print(f"[benchmark-assets] files={len(manifest['figures_and_tables'])}")
    for path in manifest["figures_and_tables"]:
        print(path)


if __name__ == "__main__":
    main()
