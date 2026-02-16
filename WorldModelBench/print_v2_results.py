"""Print per-dimension v2 results from existing eval outputs."""
import json
import sys
from pathlib import Path
import numpy as np

MAX_Q = 5
DIMENSION_SUBS = {
    "Physical Laws": ["Newton", "Mass", "Fluid", "Penetration", "Gravity"],
    "Common Sense": ["Aesthetics", "Temporal"],
}


def _parse_score(text: str) -> float:
    """Extract score from 'Score: X' pattern, return 0 on failure."""
    try:
        return min(float(text.split(":")[-1].strip(" .")), MAX_Q)
    except ValueError:
        return 0.0


def load_results(result_dir: Path) -> dict:
    """Load results.json, compute scores from preds/accs if needed."""
    with open(result_dir / "results.json") as f:
        data = json.load(f)

    if "detail" in data:
        return data

    preds = data.get("preds", {})
    if not preds:
        return data

    instr_scores = []
    pl_scores = {n: [] for n in DIMENSION_SUBS["Physical Laws"]}
    cs_scores = {n: [] for n in DIMENSION_SUBS["Common Sense"]}

    for vid, evals in preds.items():
        for p in evals.get("instruction", []):
            instr_scores.append(_parse_score(p))

        for sub_names, key, scores_dict in [
            (DIMENSION_SUBS["Physical Laws"], "physical_laws", pl_scores),
            (DIMENSION_SUBS["Common Sense"], "common_sense", cs_scores),
        ]:
            sub_preds = evals.get(key, [])
            for i, name in enumerate(sub_names):
                if i < len(sub_preds):
                    scores_dict[name].append(_parse_score(sub_preds[i]))

    return {
        "model_name": data.get("model_name", "unknown"),
        "num_videos": len(preds),
        "instr_scores": instr_scores,
        "pl_scores": pl_scores,
        "cs_scores": cs_scores,
    }


def _format_dimension_rows(dim_name: str, sub_scores: dict) -> tuple[list[str], float]:
    """Format table rows for a multi-sub dimension. Returns (lines, subtotal)."""
    lines = []
    subtotal = 0.0
    for i, (name, scores) in enumerate(sub_scores.items()):
        m = np.mean(scores) if scores else 0
        subtotal += m
        prefix = dim_name if i == 0 else ""
        lines.append(f"| {prefix} | {name} | {m:.2f} | / {MAX_Q} | {m/MAX_Q*100:.1f}% |")
    dim_max = len(sub_scores) * MAX_Q
    lines.append(f"| | **Subtotal** | **{subtotal:.2f}** | **/ {dim_max}** | **{subtotal/dim_max*100:.1f}%** |")
    return lines, subtotal


def format_results(data: dict) -> str:
    """Format results as markdown table."""
    model = data["model_name"]
    n = data["num_videos"]
    instr = data["instr_scores"]

    lines = [f"### {model} ({n} videos)\n"]
    lines.append("| Dimension | Sub-category | Mean | / Max | Pct |")
    lines.append("|---|---|---:|---:|---:|")

    # Instruction
    instr_mean = np.mean(instr) if instr else 0
    lines.append(f"| Instruction Following | | {instr_mean:.2f} | / {MAX_Q} | {instr_mean/MAX_Q*100:.1f}% |")

    # Physical Laws and Common Sense
    dim_totals = {"Instruction": instr_mean}
    for dim_name, scores_key in [("Physical Laws", "pl_scores"), ("Common Sense", "cs_scores")]:
        dim_lines, subtotal = _format_dimension_rows(dim_name, data[scores_key])
        lines.extend(dim_lines)
        dim_totals[dim_name] = subtotal

    # Total
    raw_total = sum(dim_totals.values())
    raw_max = MAX_Q + sum(len(subs) * MAX_Q for subs in DIMENSION_SUBS.values())  # 5 + 25 + 10 = 40
    normalized = raw_total / raw_max * 10
    lines.append(f"| **Total** | | **{raw_total:.2f}** | **/ {raw_max}** | **{raw_total/raw_max*100:.1f}%** |")
    lines.append(f"| **Normalized** | | **{normalized:.2f}** | **/ 10** | |")
    lines.append("")

    return "\n".join(lines)


def main():
    eval_dir = Path(__file__).parent / "evaloutputs"
    dirs = sorted(eval_dir.glob("eval_v2_*"))

    if not dirs:
        print("No eval_v2 results found.")
        sys.exit(1)

    all_md = ["# WorldModelBench v2 Evaluation Results (0-5 scale)\n"]

    for d in dirs:
        print(f"Processing {d.name}...")
        data = load_results(d)
        md = format_results(data)
        all_md.append(md)
        print(md)

    return "\n".join(all_md)


if __name__ == "__main__":
    md_content = main()

    # Save to experiment-results
    out_path = Path(__file__).parent.parent / "Phyrefine" / "docs" / "experiment-results" / "worldmodelbench_v2.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content)
    print(f"\nSaved to {out_path}")
