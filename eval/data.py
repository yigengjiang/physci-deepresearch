"""Data model + loaders for the standalone PhySciBench scorer.

`Sample` is a PLAIN `pydantic.BaseModel` (NOT SQLModel) carrying every field the
ported judging code reads or writes via `.update()`. `.update()` mirrors
`core/db/dr_basemodel.py:DRBaseModel.update` and is `hasattr`-guarded, so any
field a ported method writes MUST be declared here or the value is silently
dropped — hence the exhaustive field list below.

Loaders read the raw `physcibench.json` (list of 200 records) and a
`predictions.jsonl` (`{"id", "response"}` per line), join by the string `id`
(`physci-NNN`). The `category` typo `atomic-anwser` -> `atomic-answer` is
normalized ON LOAD ONLY; the source JSON is never mutated.
"""

import json
import pathlib

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """A single PhySciBench evaluation sample (question + ground truth + judging fields)."""

    # --- Loaded from physcibench.json ---
    id: str
    raw_question: str = ""  # <- JSON "question"
    correct_answer: str = ""  # <- JSON "answer"
    category: str | None = None
    type: str | None = None
    files: list = Field(default_factory=list)
    rubrics: list = Field(default_factory=list)
    meta: dict | None = None  # None on public data
    source: str | None = None  # absent on public data; declared for completeness

    # --- Written by the ported judging code via .update() ---
    response: str | None = None
    score: float | None = None
    correct: bool | None = None
    judged_response: str | None = None
    reasoning: str | None = None
    extracted_final_answer: str | None = None
    confidence: int | None = None
    augmented_question: str | None = None
    dataset_index: int | None = None

    # ---- DRBaseModel-compatible shim (mirrors core/db/dr_basemodel.py) ----
    def update(self, **kwargs) -> None:
        """Update fields with the given keyword arguments (hasattr-guarded)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get(self, key, default=None):
        """Get the value of the specified key, or return default if not found."""
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data: dict) -> "Sample":
        return cls(**data)

    def as_dict(self) -> dict:
        # only contain fields that are not None
        return {k: v for k, v in self.model_dump().items() if v is not None}


def _normalize_category(category: str | None) -> str | None:
    """Normalize the cosmetic `atomic-anwser` typo to `atomic-answer` (load only)."""
    if category == "atomic-anwser":
        return "atomic-answer"
    return category


def load_benchmark(path: str | pathlib.Path) -> list[Sample]:
    """Load `physcibench.json` (a list of records) into `Sample` objects.

    Maps JSON `question`->`raw_question`, `answer`->`correct_answer`, reads
    `id, category, type, files, rubrics` directly, defaults
    `dataset_index` to the enumeration index, and normalizes the `category` typo.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    samples: list[Sample] = []
    for idx, rec in enumerate(records):
        samples.append(
            Sample(
                id=str(rec["id"]),
                raw_question=rec.get("question", "") or "",
                correct_answer=rec.get("answer", "") or "",
                category=_normalize_category(rec.get("category")),
                type=rec.get("type"),
                files=rec.get("files") or [],
                rubrics=rec.get("rubrics") or [],
                meta=rec.get("meta"),
                source=rec.get("source"),
                dataset_index=idx,
            )
        )
    return samples


def load_predictions(path: str | pathlib.Path) -> dict[str, str]:
    """Load `predictions.jsonl` (`{"id", "response"}` per line) into an id->response dict.

    The last occurrence of a duplicate id wins.
    """
    predictions: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            predictions[str(obj["id"])] = obj.get("response", "")
    return predictions


def join_by_id(samples: list[Sample], predictions: dict[str, str]) -> dict:
    """Attach `response` onto samples by string `id`; return join counts.

    Returns a dict with:
      - num_predictions: number of distinct prediction ids supplied
      - num_matched: benchmark samples that received a prediction
      - num_missing: benchmark samples with NO prediction (list of ids)
      - unmatched_prediction_ids: prediction ids not present in the benchmark
    """
    benchmark_ids = {s.id for s in samples}
    matched = 0
    missing: list[str] = []
    for s in samples:
        if s.id in predictions:
            s.update(response=predictions[s.id])
            matched += 1
        else:
            missing.append(s.id)
    unmatched = sorted(pid for pid in predictions if pid not in benchmark_ids)
    return {
        "num_predictions": len(predictions),
        "num_matched": matched,
        "num_missing": len(missing),
        "missing_ids": missing,
        "unmatched_prediction_ids": unmatched,
    }
