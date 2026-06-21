"""Metric aggregation for the standalone scorer.

Ported from `core/eval/processer/utils.py:MetricsUtils` (origin/submit). The kept
methods — `calculate_overall_metrics`, `calculate_category_metrics`,
`calculate_type_metrics` — have VERBATIM bodies; the ONLY edit is the type-hint
import (`EvaluationSample` -> the local `Sample`).

DROPPED for Release 1 (invalid on public data, see plan §4a / C1):
  - `calculate_subject_metrics` (public JSON has no `meta`; all 200 -> "Physics")
  - `calculate_calibration` (confidence never populated on the long-form/structured paths)
  - `calculate_level_metrics` (all 200 `level` values are None -> single None bucket)
"""

from .data import Sample


class MetricsUtils:
    @staticmethod
    def calculate_overall_metrics(samples: list[Sample]) -> dict:
        """calculate overall metrics"""
        invalid_count = 0
        total_score = 0.0
        valid_count = 0
        for item in samples:
            if item.judged_response == "invalid":
                invalid_count += 1
            else:
                total_score += item.score or 0.0
                valid_count += 1
        total = len(samples)
        correct_count = sum(1 for item in samples if item.correct)
        incorrect_count = total - correct_count - invalid_count
        avg_score = round(total_score / valid_count, 4) if valid_count > 0 else 0.0
        return {
            "Accuracy (%)": round(correct_count / total * 100, 2),
            "Average Score": avg_score,
            "Details": {
                "correct": correct_count,
                "wrong": incorrect_count,
                "unknown": invalid_count,
                "total": total,
            },
        }

    @staticmethod
    def calculate_category_metrics(samples: list[Sample]) -> dict:
        """Calculate metrics grouped by category."""
        category_bin = {}
        for item in samples:
            category = item.category or "unknown"
            if category not in category_bin:
                category_bin[category] = {"correct": 0, "wrong": 0, "unknown": 0, "total_score": 0.0, "count": 0}
            if item.judged_response == "invalid":
                category_bin[category]["unknown"] += 1
                continue
            category_bin[category]["count"] += 1
            category_bin[category]["total_score"] += item.score or 0.0
            if item.correct:
                category_bin[category]["correct"] += 1
            else:
                category_bin[category]["wrong"] += 1

        # Calculate accuracy and average score for each category
        for _, counts in category_bin.items():
            total = counts["correct"] + counts["wrong"]
            if total > 0:
                counts["accuracy"] = round(counts["correct"] / total * 100, 4)
            else:
                counts["accuracy"] = 0.0
            if counts["count"] > 0:
                counts["avg_score"] = round(counts["total_score"] / counts["count"], 4)
            else:
                counts["avg_score"] = 0.0

        return {
            "category_metrics": category_bin,
        }

    @staticmethod
    def calculate_type_metrics(samples: list[Sample]) -> dict:
        """Calculate metrics grouped by type."""
        type_bin = {}
        for item in samples:
            sample_type = item.type or "unknown"
            if sample_type not in type_bin:
                type_bin[sample_type] = {"correct": 0, "wrong": 0, "unknown": 0, "total_score": 0.0, "count": 0}
            if item.judged_response == "invalid":
                type_bin[sample_type]["unknown"] += 1
                continue
            type_bin[sample_type]["count"] += 1
            type_bin[sample_type]["total_score"] += item.score or 0.0
            if item.correct:
                type_bin[sample_type]["correct"] += 1
            else:
                type_bin[sample_type]["wrong"] += 1

        # Calculate accuracy and average score for each type
        for _, counts in type_bin.items():
            total = counts["correct"] + counts["wrong"]
            if total > 0:
                counts["accuracy"] = round(counts["correct"] / total * 100, 4)
            else:
                counts["accuracy"] = 0.0
            if counts["count"] > 0:
                counts["avg_score"] = round(counts["total_score"] / counts["count"], 4)
            else:
                counts["avg_score"] = 0.0

        return {
            "type_metrics": type_bin,
        }
