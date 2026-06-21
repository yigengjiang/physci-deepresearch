"""Standalone PhySciBench scorer entrypoint (`python -m eval.score`).

Loads `physcibench.json` + `predictions.jsonl`, joins by string `id`, judges
each sample asynchronously, computes metrics, and writes `metrics.json` with
EXACTLY these top-level keys (plan §10):

    overall, category_metrics, type_metrics,
    e2b_enabled, files_present, skipped,
    num_predictions, num_matched, num_missing

There are NO subject_metrics / level_metrics / calibration keys.
"""

import argparse
import asyncio
import json
import logging
import pathlib

from . import sandbox
from .config import ScorerConfig
from .data import join_by_id, load_benchmark, load_predictions
from .judge import PhySciBenchJudge

logger = logging.getLogger("eval.score")


def _files_present(data_dir: pathlib.Path) -> bool:
    files_dir = data_dir / "files"
    return files_dir.is_dir() and any(files_dir.iterdir())


async def run_scoring(
    *,
    predictions_path: str | pathlib.Path,
    benchmark_path: str | pathlib.Path,
    data_dir: str | pathlib.Path,
    dataset: str = "PhySciBench",
    judge_concurrency: int = 8,
    limit: int | None = None,
    judge: PhySciBenchJudge | None = None,
) -> dict:
    """Load, join, judge, and aggregate. Returns the metrics dict.

    `judge` may be injected (tests pass a stubbed judge); otherwise a real
    `PhySciBenchJudge` is built from the environment-derived config.
    """
    config = ScorerConfig.from_env(dataset=dataset, data_dir=data_dir)
    if judge is None:
        judge = PhySciBenchJudge(config)

    samples = load_benchmark(benchmark_path)
    predictions = load_predictions(predictions_path)
    join_counts = join_by_id(samples, predictions)

    if limit is not None:
        samples = samples[:limit]

    logger.info(
        "Loaded %d benchmark samples, %d predictions; matched=%d missing=%d unmatched=%d",
        len(samples),
        join_counts["num_predictions"],
        join_counts["num_matched"],
        join_counts["num_missing"],
        len(join_counts["unmatched_prediction_ids"]),
    )
    if join_counts["unmatched_prediction_ids"]:
        logger.warning("Prediction ids not in benchmark: %s", join_counts["unmatched_prediction_ids"])
    if join_counts["missing_ids"]:
        logger.warning("Benchmark ids missing a prediction: %s", join_counts["missing_ids"])

    semaphore = asyncio.Semaphore(judge_concurrency)

    async def _judge(sample):
        async with semaphore:
            return await judge.judge_one(sample)

    judged = await asyncio.gather(*[_judge(s) for s in samples])

    metrics = judge.calculate_metrics(judged)
    overall = {
        "Accuracy (%)": metrics["Accuracy (%)"],
        "Average Score": metrics["Average Score"],
        "Details": metrics["Details"],
    }

    skipped = {
        "code_generation_e2b_unavailable": getattr(judge, "skipped_code_generation", 0),
    }

    result = {
        "overall": overall,
        "category_metrics": metrics["category_metrics"],
        "type_metrics": metrics["type_metrics"],
        "e2b_enabled": sandbox.e2b_available(),
        "files_present": _files_present(pathlib.Path(data_dir)),
        "skipped": skipped,
        "num_predictions": join_counts["num_predictions"],
        "num_matched": join_counts["num_matched"],
        "num_missing": join_counts["num_missing"],
    }
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhySciBench standalone scorer")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl ({id, response} per line)")
    parser.add_argument("--out", default="metrics.json", help="Output metrics path (default: metrics.json)")
    parser.add_argument("--data-dir", default="PhySciBench", help="Data directory holding physcibench.json and files/ (default: PhySciBench)")
    parser.add_argument("--benchmark", default=None, help="Path to physcibench.json (default: <data-dir>/physcibench.json)")
    parser.add_argument("--dataset", default="PhySciBench", help="Benchmark name (default: PhySciBench)")
    parser.add_argument("--judge-concurrency", type=int, default=8, help="Max concurrent judge calls (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N samples")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s - %(message)s")
    args = _parse_args(argv)

    data_dir = pathlib.Path(args.data_dir)
    benchmark_path = pathlib.Path(args.benchmark) if args.benchmark else data_dir / "physcibench.json"

    result = asyncio.run(
        run_scoring(
            predictions_path=args.predictions,
            benchmark_path=benchmark_path,
            data_dir=data_dir,
            dataset=args.dataset,
            judge_concurrency=args.judge_concurrency,
            limit=args.limit,
        )
    )

    out_path = pathlib.Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %s | overall=%s e2b=%s files=%s skipped=%s",
        out_path,
        result["overall"]["Average Score"],
        result["e2b_enabled"],
        result["files_present"],
        result["skipped"],
    )


if __name__ == "__main__":
    main()
