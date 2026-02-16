"""Print per-dimension v2 results from existing eval outputs."""
import json
import sys
from pathlib import Path
import numpy as np

MAX_Q = 5
SUB_NAMES_PL = ["Newton", "Mass", "Fluid", "Penetration", "Gravity"]
SUB_NAMES_CS = ["Aesthetics", "Temporal"]


def load_results(result_dir: Path) -> dict:
    """Load results.json, compute scores from preds/accs if needed."""
    with open(result_dir / "results.json") as f:
        data = json.load(f)

    # If already has detailed sub-scores, return as-is
    if "detail" in data:
        return data

    # Compute from preds
    preds = data.get("preds", {})
    if not preds:
        return data

    instr_scores, pl_scores, cs_scores = [], {n: [] for n in SUB_NAMES_PL}, {n: [] for n in SUB_NAMES_CS}

    for vid, evals in preds.items():
        # instruction
        for p in evals.get("instruction", []):
            try:
                s = float(p.split(":")[-1].strip(" ."))
                instr_scores.append(min(s, MAX_Q))
            except ValueError:
                instr_scores.append(0)

        # physical_laws (5 sub-questions interleaved)
        pl_preds = evals.get("physical_laws", [])
        for i, name in enumerate(SUB_NAMES_PL):
            if i < len(pl_preds):
                try:
                    s = float(pl_preds[i].split(":")[-1].strip(" ."))
                    pl_scores[name].append(min(s, MAX_Q))
                except ValueError:
                    pl_scores[name].append(0)

        # common_sense (2 sub-questions interleaved)
        cs_preds = evals.get("common_sense", [])
        for i, name in enumerate(SUB_NAMES_CS):
            if i < len(cs_preds):
                try:
                    s = float(cs_preds[i].split(":")[-1].strip(" ."))
                    cs_scores[name].append(min(s, MAX_Q))
                except ValueError:
                    cs_scores[name].append(0)

    return {
        "model_name": data.get("model_name", "unknown"),
        "num_videos": len(preds),
        "instr_scores": instr_scores,
        "pl_scores": pl_scores,
        "cs_scores": cs_scores,
    }


def format_results(data: dict) -> str:
    """Format results as markdown table."""
    model = data["model_name"]
    n = data["num_videos"]
    instr = data["instr_scores"]
    pl = data["pl_scores"]
    cs = data["cs_scores"]

    lines = [f"### {model} ({n} videos)\n"]
    lines.append("| Dimension | Sub-category | Mean | / Max | Pct |")
    lines.append("|---|---|---:|---:|---:|")

    # Instruction
    instr_mean = np.mean(instr) if instr else 0
    lines.append(f"| Instruction Following | | {instr_mean:.2f} | / {MAX_Q} | {instr_mean/MAX_Q*100:.1f}% |")

    # Physical Laws
    pl_total = 0
    for i, name in enumerate(SUB_NAMES_PL):
        m = np.mean(pl[name]) if pl[name] else 0
        pl_total += m
        prefix = "Physical Laws" if i == 0 else ""
        lines.append(f"| {prefix} | {name} | {m:.2f} | / {MAX_Q} | {m/MAX_Q*100:.1f}% |")
    pl_max = len(SUB_NAMES_PL) * MAX_Q
    lines.append(f"| | **Subtotal** | **{pl_total:.2f}** | **/ {pl_max}** | **{pl_total/pl_max*100:.1f}%** |")

    # Common Sense
    cs_total = 0
    for i, name in enumerate(SUB_NAMES_CS):
        m = np.mean(cs[name]) if cs[name] else 0
        cs_total += m
        prefix = "Common Sense" if i == 0 else ""
        lines.append(f"| {prefix} | {name} | {m:.2f} | / {MAX_Q} | {m/MAX_Q*100:.1f}% |")
    cs_max = len(SUB_NAMES_CS) * MAX_Q
    lines.append(f"| | **Subtotal** | **{cs_total:.2f}** | **/ {cs_max}** | **{cs_total/cs_max*100:.1f}%** |")

    # Total
    raw_total = instr_mean + pl_total + cs_total
    raw_max = MAX_Q + pl_max + cs_max  # 5 + 25 + 10 = 40
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
