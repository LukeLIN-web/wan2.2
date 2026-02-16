"""WorldModelBench v2: 0-5 Likert scale evaluation for all dimensions.

Changes from v1:
- All dimensions use 0-5 scoring (was: instruction 0-3, physics/common_sense binary)
- Total raw score: 40 (1×5 + 5×5 + 2×5), normalized to 10-point scale
- Compatible with v1 judge models (VILA) and stronger VLMs
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os
import re

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import numpy as np
from mmengine import load, dump
from collections import defaultdict
from tqdm import tqdm


MAX_SCORE_PER_QUESTION = 5
TOTAL_RAW_MAX = 40  # 1×5 + 5×5 + 2×5
NORMALIZED_MAX = 10.0


class EvaluationType(Enum):
    INSTRUCTION = "instruction"
    PHYSICAL_LAWS = "physical_laws"
    COMMON_SENSE = "common_sense"


# --- Prompt Templates ---

PROMPT_INSTRUCTION = """
Evaluate if this video follows the instruction: '{instruction}'.

Scoring criteria (0-5):
- 0: Completely irrelevant — the video content has nothing to do with the instruction.
- 1: Related scene but wrong action (e.g., instructed to stop but the car accelerates).
- 2: Correct direction but incomplete execution (e.g., instructed to stop but only slows down).
- 3: Mostly follows instruction but with noticeable flaws (e.g., stops but at wrong location).
- 4: Follows instruction well with only minor deviations.
- 5: Precisely executes the instruction — action, object, and timing are all correct.

Let's analyze step-by-step and conclude with 'Score: [0-5]'.
""".strip()

PROMPT_PHYSICAL_LAWS = """
Evaluate the video for '{physical_laws}'.

Scoring criteria (0-5):
- 0: Severe violation — physics completely breaks down (objects pass through each other, vanish, or float without gravity).
- 1: Multiple obvious violations (e.g., no collision response AND unrealistic trajectory).
- 2: One obvious violation, rest is mostly reasonable.
- 3: No obvious violation, but noticeable unnaturalness (e.g., unrealistic speed or acceleration).
- 4: Minor unnaturalness only detectable upon close inspection.
- 5: Fully physically correct — no violations whatsoever.

Let's analyze step-by-step and conclude with 'Score: [0-5]'.
""".strip()

PROMPT_AESTHETICS = """
Evaluate the visual quality (aesthetics) of this video.

Scoring criteria (0-5):
- 0: Severely distorted — frames are unrecognizable.
- 1: Large-area artifacts, heavy blur, or severe color distortion.
- 2: Multiple localized artifacts or obvious quality issues.
- 3: Acceptable overall but with visible flaws (e.g., blurry edges, local distortion).
- 4: Good quality with only minor imperfections.
- 5: High quality — visually close to a real video.

Let's analyze step-by-step and conclude with 'Score: [0-5]'.
""".strip()

PROMPT_TEMPORAL = """
Evaluate the temporal consistency of this video.

Scoring criteria (0-5):
- 0: Completely incoherent — frames are unrelated to each other.
- 1: Severe flickering or drastic shape/appearance jumps between frames.
- 2: Multiple noticeable discontinuities (e.g., objects suddenly appear or vanish).
- 3: Generally coherent but with visible jitter or local inconsistencies.
- 4: Mostly smooth, only occasional minor inconsistencies.
- 5: Perfectly smooth — temporal consistency indistinguishable from a real video.

Let's analyze step-by-step and conclude with 'Score: [0-5]'.
""".strip()


def get_prompt_templates() -> Dict[str, str]:
    return {
        EvaluationType.INSTRUCTION.value: PROMPT_INSTRUCTION,
        EvaluationType.PHYSICAL_LAWS.value: PROMPT_PHYSICAL_LAWS,
        # common_sense has per-question templates, handled separately
    }


def get_question_pool() -> Dict[str, Optional[List[str]]]:
    return {
        EvaluationType.INSTRUCTION.value: None,
        EvaluationType.PHYSICAL_LAWS.value: [
            "Newton's Law compliance: Do objects move only when acted upon by external forces?",
            "Mass and deformation: Do solid objects maintain consistent shape and mass?",
            "Fluid dynamics: Do liquids flow in a physically natural manner?",
            "Non-penetration: Do objects avoid unnaturally passing through each other?",
            "Gravity: Do objects behave consistently with gravity?",
        ],
        EvaluationType.COMMON_SENSE.value: [
            PROMPT_AESTHETICS,
            PROMPT_TEMPORAL,
        ],
    }


@dataclass
class EvaluationConfig:
    PROMPT_TEMPLATES: Dict[str, str] = field(default_factory=get_prompt_templates)
    QUESTION_POOL: Dict[str, Optional[List[str]]] = field(default_factory=get_question_pool)


# --- Score Parsing ---

_SCORE_RE = re.compile(r"Score:\s*(\d(?:\.\d+)?)", re.IGNORECASE)


def parse_score(pred: str) -> float:
    """Extract 'Score: X' from judge output. Returns 0 on failure."""
    m = _SCORE_RE.search(pred)
    if m:
        return min(float(m.group(1)), MAX_SCORE_PER_QUESTION)
    # fallback: try last number in the text
    nums = re.findall(r"\b(\d(?:\.\d+)?)\b", pred)
    if nums:
        val = float(nums[-1])
        if 0 <= val <= MAX_SCORE_PER_QUESTION:
            return val
    return 0.0


def _build_prompts(eval_type: EvaluationType, config: 'EvaluationConfig', video_item: dict) -> list[str]:
    """Build the list of prompts for a given evaluation type and video."""
    template = config.PROMPT_TEMPLATES.get(eval_type.value)
    questions = config.QUESTION_POOL[eval_type.value]

    if eval_type == EvaluationType.INSTRUCTION:
        return [template.format(instruction=video_item["text_instruction"])]
    if eval_type == EvaluationType.PHYSICAL_LAWS:
        return [template.format(physical_laws=q.lower()) for q in questions]
    if eval_type == EvaluationType.COMMON_SENSE:
        # Each question is already a full prompt
        return list(questions)
    return []


# --- Results Display ---

class ResultsPrinter:
    def __init__(self):
        self.console = Console()

    def print_results(self, accs: Dict[str, list], num_videos: int) -> Dict[str, float]:
        sub_names = {
            EvaluationType.PHYSICAL_LAWS.value: ["Newton", "Mass", "Fluid", "Penetration", "Gravity"],
            EvaluationType.COMMON_SENSE.value: ["Aesthetics", "Temporal"],
        }

        table = Table(title="Evaluation Results (0-5 scale)", show_header=True, header_style="bold magenta")
        table.add_column("Dimension", style="cyan")
        table.add_column("Sub-category", style="cyan")
        table.add_column("Mean", justify="right", style="yellow")
        table.add_column("/ Max", justify="right", style="dim")

        dim_scores = {}  # per-dimension raw scores

        for eval_type in EvaluationType:
            dim_name = eval_type.value.replace("_", " ").title()
            scores = accs[eval_type.value]

            if eval_type.value in sub_names:
                names = sub_names[eval_type.value]
                num_sub = len(names)
                dim_max = num_sub * MAX_SCORE_PER_QUESTION
                dim_sum = 0.0
                for i, name in enumerate(names):
                    sub_mean = np.mean(scores[i::num_sub])
                    dim_sum += sub_mean
                    dim_scores[f"{dim_name}/{name}"] = sub_mean
                    table.add_row(
                        dim_name if i == 0 else "",
                        name, f"{sub_mean:.2f}", f"/ {MAX_SCORE_PER_QUESTION}",
                    )
                table.add_row("", "[bold]Subtotal[/bold]", f"[bold]{dim_sum:.2f}[/bold]",
                              f"/ {dim_max}")
                dim_scores[dim_name] = dim_sum
            else:
                mean = np.mean(scores)
                dim_scores[dim_name] = mean
                table.add_row(dim_name, "", f"{mean:.2f}", f"/ {MAX_SCORE_PER_QUESTION}")

        self.console.print(table)

        # Per-dimension summary
        raw_total = sum(dim_scores[k] for k in ["Instruction", "Physical Laws", "Common Sense"])
        summary_table = Table(title="Per-Dimension Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Dimension", style="cyan")
        summary_table.add_column("Score", justify="right", style="yellow")
        summary_table.add_column("Max", justify="right", style="dim")
        summary_table.add_column("Pct", justify="right", style="green")

        dim_max_map = {"Instruction": 5, "Physical Laws": 25, "Common Sense": 10}
        for dim, max_val in dim_max_map.items():
            val = dim_scores[dim]
            summary_table.add_row(dim, f"{val:.2f}", str(max_val), f"{val / max_val * 100:.1f}%")
        summary_table.add_row(
            "[bold]Total[/bold]", f"[bold]{raw_total:.2f}[/bold]", f"[bold]{TOTAL_RAW_MAX}[/bold]",
            f"[bold]{raw_total / TOTAL_RAW_MAX * 100:.1f}%[/bold]",
        )
        normalized = raw_total / TOTAL_RAW_MAX * NORMALIZED_MAX
        summary_table.add_row(
            "[bold green]Normalized[/bold green]", f"[bold green]{normalized:.2f}[/bold green]",
            f"[bold green]{NORMALIZED_MAX:.0f}[/bold green]", "",
        )
        self.console.print(summary_table)

        return dim_scores


# --- Evaluator ---

class WorldModelEvaluator:
    def __init__(self, judge_path: str, video_dir: str, config: EvaluationConfig):
        self.judge = self._load_judge(judge_path)
        self.video_dir = Path(video_dir)
        self.config = config
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _load_judge(judge_path: str):
        import llava
        return llava.load(judge_path)

    def _load_video(self, video_name: str):
        video_path = self.video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            self.logger.warning(f"Video not found: {video_path}")
            return None
        import llava
        return llava.Video(str(video_path))

    def evaluate_video(self, video, prompt: str, cot: bool = True) -> str:
        if not cot:
            prompt = prompt.replace(
                "Let's analyze step-by-step and conclude with", "Answer with"
            )
        return self.judge.generate_content([video, prompt])


# --- Main ---

class RichLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.console = Console()

    def emit(self, record):
        try:
            msg = self.format(record)
            style = "bold red" if record.levelno >= logging.WARNING else "blue"
            self.console.print(f"[{style}]{msg}[/{style}]")
        except Exception:
            self.handleError(record)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="WorldModelBench v2 (0-5 scale)")
    parser.add_argument("--judge", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="worldmodelbench_v2_results")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichLogHandler()])

    config = EvaluationConfig()
    evaluator = WorldModelEvaluator(args.judge, args.video_dir, config)
    printer = ResultsPrinter()
    console = printer.console

    console.print("[bold]Loading validation set...[/bold]")
    validation_set = load("./worldmodelbench.json")
    if args.end > 0:
        validation_set = validation_set[args.start:args.end]
    elif args.start > 0:
        validation_set = validation_set[args.start:]

    save_path = f"{args.save_name}_cot.json" if args.cot else f"{args.save_name}.json"

    if os.path.exists(save_path):
        console.print("[bold yellow]Loading existing results...[/bold yellow]")
        results = load(save_path)
        preds, accs = results["preds"], results["accs"]
    else:
        console.print("[bold green]Starting v2 evaluation (0-5 scale)...[/bold green]")
        preds = {}
        accs = defaultdict(list)

        with Progress(
            "[progress.description]{task.description}", BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn(),
            console=console,
        ) as progress:
            video_task = progress.add_task("Processing videos", total=len(validation_set))

            for v_i in tqdm(validation_set, total=len(validation_set)):
                video_name = Path(v_i["first_frame"]).stem
                video = evaluator._load_video(video_name)
                if not video:
                    progress.advance(video_task)
                    continue

                eval_task = progress.add_task(f"Evaluating {video_name}", total=len(EvaluationType))

                for eval_type in EvaluationType:
                    prompts = _build_prompts(eval_type, config, v_i)
                    preds_i = []
                    for prompt in prompts:
                        pred = evaluator.evaluate_video(video, prompt, args.cot)
                        preds_i.append(pred)
                        accs[eval_type.value].append(parse_score(pred))

                    preds.setdefault(video_name, {})[eval_type.value] = preds_i
                    progress.advance(eval_task)

                progress.remove_task(eval_task)
                progress.advance(video_task)

        if not args.no_save:
            results = {"model_name": args.model_name, "version": "v2", "preds": preds, "accs": accs}
            dump(results, save_path, indent=4)
            console.print(f"[green]Results saved to: {save_path}[/green]")

    console.print("\n[bold]Final Evaluation Results (v2)[/bold]")
    printer.print_results(accs, len(preds))


if __name__ == "__main__":
    main()
