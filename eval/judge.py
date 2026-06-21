"""PhySciBench LLM-as-judge scoring (standalone, clean-room).

Ported from `core/eval/processer/physci_bench.py` (origin/submit). The scoring
methods' BODIES are copied CHARACTER-FOR-CHARACTER. The ONLY edits are:
  (i) import lines — `core.config`/`core.utils`/`..data`/base processors are
      replaced by the local `JudgeClient`, `Sample`, `extract_*`, `load_prompts`,
      `MetricsUtils`, and a stdlib `logging` logger;
  (ii) `__init__` builds a local `JudgeClient` from the local `ScorerConfig`;
  (iii) `_get_file_prompt` resolves files against the CONFIGURED data dir
       (`config.data_dir / "files"`) instead of `Path(__file__).parent x4`;
  (iv) `_execute_code` delegates to `eval.sandbox.execute_code` for graceful
       e2b degradation (the extraction+run logic itself is preserved verbatim
       inside `eval/sandbox.py`); when the sandbox is unavailable the sample is
       annotated and counted as skipped;
  (v) `calculate_metrics` calls only the kept MetricsUtils functions
      (overall + category + type) — the source `calculate_level_metrics` /
      `calculate_subject_metrics` calls are removed (plan §4b / C1).

`_parse_judge_response` is the 3-group `physci_bench.py` variant (verbatim),
NOT the base_llm_processor.py 4-group/confidence variant.
"""

import csv
import io
import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from . import sandbox
from ._extract import extract_csv_re, extract_json_re
from ._openai_client import JudgeClient
from .config import ScorerConfig
from .data import Sample
from .metrics import MetricsUtils
from .prompts import load_prompts

logger = logging.getLogger(__name__)

# Load judge prompts from the vendored YAML (retains the top-level `default` key).
JUDGE_PROMPTS = load_prompts()


class CriterionEvaluation(BaseModel):
    """Model for evaluating a single criterion"""

    criterion: str = Field(description="Criterion identifier (e.g., criterion1, criterion2, ...)")
    criterion_name: str = Field(description="The name of the criterion from the rubric")
    reasoning: str = Field(
        description="Explain why the response meets or does not meet this criterion when compared to the correct answer. "
        "Focus on what is present or missing in the response relative to the standard set by the correct answer."
    )
    score: Literal[0, 1] = Field(description="Score: 1 if the response fully meets the criterion, 0 otherwise")


class JudgmentResponse(BaseModel):
    """Model for the overall judgment response containing all criterion evaluations"""

    evaluations: list[CriterionEvaluation] = Field(description="List of criterion evaluations")


class PhySciBenchJudge:
    """Standalone clone of `PhySciBenchProcesser` (judging methods only)."""

    name: str = "PhySciBench"

    def __init__(self, config: ScorerConfig) -> None:
        self.config = config
        self.judge_client = JudgeClient(**config.judge_model.model_provider.model_dump())
        # number of code-generation samples skipped because e2b was unavailable
        self.skipped_code_generation = 0

    def preprocess_one(self, sample: Sample) -> Sample:
        augmented_question = sample.raw_question + self._get_file_prompt(sample.files)
        sample.update(
            augmented_question=augmented_question,
        )
        return sample

    async def judge_one(self, sample: Sample) -> Sample:
        """Judge a single sample based on data type"""
        response = sample.response
        correct_answer = sample.correct_answer
        rubrics = sample.rubrics
        data_type = sample.type
        question = sample.raw_question

        if not response:
            sample.update(judged_response="No response provided", correct=False, score=0.0)
            return sample

        if not correct_answer:
            sample.update(judged_response="No correct answer provided", correct=False, score=0.0)
            return sample

        try:
            if data_type == "code-generation":
                # Execute both response and correct answer code
                response_result = await self._execute_code(response)
                correct_result = await self._execute_code(correct_answer)

                # Graceful degradation: skip + annotate when e2b is unavailable.
                if response_result == sandbox.SANDBOX_UNAVAILABLE or correct_result == sandbox.SANDBOX_UNAVAILABLE:
                    self.skipped_code_generation += 1
                    sample.update(
                        judged_response="invalid",
                        correct=False,
                        score=0.0,
                    )
                    return sample

                if rubrics:
                    await self._eval_long_form_answer(sample, question, correct_result, rubrics, response_result)
                else:
                    await self._eval_atomic_answer(sample, question, correct_result, response_result)

            elif data_type in ["multimodal-qa", "long-context-qa", "experimental-design", "scientific-reasoning"]:
                if rubrics:
                    await self._eval_long_form_answer(sample, question, correct_answer, rubrics, response)
                else:
                    await self._eval_atomic_answer(sample, question, correct_answer, response)

            elif data_type == "structured-information-extraction":
                await self._eval_structured_data(sample, correct_answer, response)

            else:
                logger.warning(f"Unknown data type: {data_type}. Using atomic answer evaluation.")
            #await self._eval_atomic_answer(sample, question, correct_answer, response)

        except Exception as e:
            logger.error(f"Error judging sample {sample.dataset_index}: {e}")
            sample.update(judged_response=f"Error during evaluation: {str(e)}", correct=False, score=0.0)

        return sample

    async def _execute_code(self, code: str) -> str:
        """Execute Python code using e2b sandbox and return the output.

        Delegates to `eval.sandbox.execute_code`, which holds the source
        extraction + run logic verbatim and returns `SANDBOX_UNAVAILABLE` when
        e2b is not configured (graceful degradation).
        """
        return await sandbox.execute_code(code)

    async def _eval_long_form_answer(
        self, sample: Sample, question: str, correct_answer: str, rubrics: list, response: str
    ) -> None:
        """Evaluate long-form answer using LLM with rubrics"""
        # Load template from YAML
        judge_template = JUDGE_PROMPTS.get("PhySciBench_LongForm", JUDGE_PROMPTS["default"])

        # Convert rubrics list to JSON string for prompt if needed
        if isinstance(rubrics, list):
            rubrics_str = json.dumps(rubrics, indent=2)
        else:
            rubrics_str = str(rubrics)

        judge_prompt = judge_template.format(
            question=question, response=response, correct_answer=correct_answer, rubric=rubrics_str
        )

        try:
            # Use structured output with Pydantic model
            completion = await self.judge_client.beta.chat.completions.parse(
                model=self.config.judge_model.model_provider.model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format=JudgmentResponse,
                **self.config.judge_model.model_params.model_dump(),
            )

            judgment = completion.choices[0].message.parsed

            # Parse rubrics to extract weights
            if isinstance(rubrics, str):
                try:
                    rubrics_list = json.loads(rubrics)
                except json.JSONDecodeError:
                    rubrics_list = []
            else:
                rubrics_list = rubrics if rubrics else []

            # Create a mapping from criterion identifier to weight
            weight_map = {}
            for rubric_item in rubrics_list:
                # Find the criterion key (e.g., "criterion1", "criterion2")
                for key, value in rubric_item.items():
                    if key.startswith("criterion") and not key.endswith("_name"):
                        # Use this key as criterion identifier
                        criterion_id = key
                        weight = rubric_item.get("weight", 1.0 / len(rubrics_list) if rubrics_list else 1.0)
                        weight_map[criterion_id] = weight
                        break

            # Calculate weighted score
            weighted_score = 0.0
            total_weight = 0.0
            unweighted_total = 0
            max_score = len(judgment.evaluations)

            for eval_result in judgment.evaluations:
                criterion_id = eval_result.criterion
                weight = weight_map.get(criterion_id, 1.0 / max_score if max_score > 0 else 1.0)
                weighted_score += eval_result.score * weight
                total_weight += weight
                unweighted_total += eval_result.score

            # Normalize the weighted score if total weight is not 1.0
            if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
                final_score = weighted_score / total_weight
            else:
                final_score = weighted_score

            # Format judgment response
            judgment_text = f"Weighted Score: {final_score:.4f} ({final_score:.2%})\n"
            judgment_text += f"Unweighted Score: {unweighted_total}/{max_score}\n\n"

            for i, eval_result in enumerate(judgment.evaluations, 1):
                criterion_id = eval_result.criterion
                weight = weight_map.get(criterion_id, 1.0 / max_score if max_score > 0 else 1.0)
                judgment_text += f"Criterion {i}: {eval_result.criterion}\n"
                judgment_text += f"Name: {eval_result.criterion_name}\n"
                judgment_text += f"Score: {eval_result.score}\n"
                judgment_text += f"Weight: {weight}\n"
                judgment_text += f"Weighted Contribution: {eval_result.score * weight:.4f}\n"
                judgment_text += f"Reasoning: {eval_result.reasoning}\n\n"

            # Update sample with results
            sample.update(
                judged_response=judgment_text,
                score=final_score,
                correct=(final_score >= 0.8),
                reasoning=f"Weighted score: {final_score:.4f} (unweighted: {unweighted_total}/{max_score})",
            )

        except Exception as e:
            logger.error(f"Error in long-form answer evaluation: {e}")
            sample.update(judged_response=f"Evaluation error: {str(e)}", correct=False, score=0.0)

    async def _eval_atomic_answer(
        self, sample: Sample, question: str, correct_answer: str, response: str
    ) -> None:
        """Evaluate atomic answer using LLM (similar to base_llm_processor)"""
        # Check for exact match first
        if self._extract_exact_answer(response) == correct_answer:
            sample.update(judged_response="Exact match", correct=True, score=1.0)
            return

        # Load template from YAML
        judge_template = JUDGE_PROMPTS.get("PhySciBench_Atomic", JUDGE_PROMPTS["default"])
        judge_prompt = judge_template.format(question=question, response=response, correct_answer=correct_answer)

        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": judge_prompt}]

            content = await self.judge_client.query_one(
                messages=messages, **self.config.judge_model.model_params.model_dump()
            )

            parsed_content = self._parse_judge_response(content)
            # Add score based on correctness: 1.0 if correct, 0.0 if incorrect
            parsed_content["score"] = 1.0 if parsed_content.get("correct", False) else 0.0
            sample.judged_response = content
            sample.update(**parsed_content)

        except Exception as e:
            logger.error(f"Error in atomic answer evaluation: {e}")
            sample.update(judged_response=f"Evaluation error: {str(e)}", correct=False, score=0.0)

    async def _eval_structured_data(self, sample: Sample, correct_answer: str, response: str) -> None:
        """Evaluate structured data (JSON or CSV)

        First determines data type by parsing correct_answer, then evaluates response.
        Handles markdown-wrapped responses (e.g., ```json ... ```) using regex extraction.
        """
        scores = None
        data_format = None

        # Step 1: Determine data type by parsing correct_answer
        is_json = False
        try:
            json.loads(correct_answer)
            is_json = True
            data_format = "JSON"
        except json.JSONDecodeError:
            # Assume CSV if not JSON
            is_json = False
            data_format = "CSV"

        # Step 2: Parse response based on determined data type
        try:
            if is_json:
                # Try to parse response as JSON
                response_data = response
                try:
                    json.loads(response)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code block
                    extracted = extract_json_re(response)
                    if extracted:
                        response_data = extracted
                        logger.info("Extracted JSON from markdown code block")
                    else:
                        # Last attempt: try to find JSON-like content
                        logger.warning("Could not extract JSON from response, using original")

                scores = self._evaluate_json(correct_answer, response_data)

            else:  # CSV
                # Try to parse response as CSV
                response_data = response
                try:
                    list(csv.reader(io.StringIO(response)))
                except csv.Error:
                    # Try to extract CSV from markdown code block
                    extracted = extract_csv_re(response)
                    if extracted:
                        response_data = extracted
                        logger.info("Extracted CSV from markdown code block")
                    else:
                        logger.warning("Could not extract CSV from response, using original")

                scores = self._evaluate_csv(correct_answer, response_data)

        except Exception as e:
            logger.error(f"Failed to evaluate {data_format}: {e}")
            sample.update(
                judged_response=f"Invalid {data_format} format. Error: {str(e)}",
                correct=False,
                score=0.0
            )
            return

        if scores:
            # Update sample with evaluation results
            final_score = scores["final_score"]  # This is already 0-1 scale
            is_correct = (final_score >= 0.8)  # Correct only if score is greater than or equal to 0.8

            judgment_text = f"{data_format} Evaluation Results:\n"
            judgment_text += f"Syntax Score: {scores['s_s (syntax)']}\n"
            judgment_text += f"Key Match Score: {scores['s_k (key_match)']}\n"
            judgment_text += f"Value Match Score: {scores['s_v (value_match)']}\n"
            judgment_text += f"Final Score: {final_score:.4f} ({final_score:.2%})\n"

            sample.update(
                judged_response=judgment_text,
                score=final_score,
                correct=is_correct,
                reasoning=f"{data_format} structure evaluation: {final_score:.2%} match",
            )

    def _evaluate_json(self, ground_truth_str: str, prediction_str: str) -> dict:
        """Evaluate JSON structure (from test_struct_json.py)"""

        def _flatten_json(data, prefix=""):
            """Recursively flatten JSON into dot-notation keys"""
            flat_map = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    flat_map.update(_flatten_json(value, new_prefix))
            elif isinstance(data, list):
                if not data:
                    flat_map[prefix] = []
                for i, item in enumerate(data):
                    new_prefix = f"{prefix}[{i}]"
                    flat_map.update(_flatten_json(item, new_prefix))
            else:
                flat_map[prefix] = data
            return flat_map

        # Calculate s_s (Syntax Score)
        try:
            pred_data = json.loads(prediction_str)
            s_s = 1.0
        except json.JSONDecodeError:
            return {"s_s (syntax)": 0.0, "s_k (key_match)": 0.0, "s_v (value_match)": 0.0, "final_score": 0.0}

        gt_data = json.loads(ground_truth_str)

        # Flatten both JSON objects
        gt_flat = _flatten_json(gt_data)
        pred_flat = _flatten_json(pred_data)

        if not gt_flat:
            is_correct = not pred_flat
            score = 1.0 if is_correct else 0.0
            return {"s_s (syntax)": 1.0, "s_k (key_match)": score, "s_v (value_match)": score, "final_score": score}

        # Calculate s_k (Key Match Score) and s_v (Value Match Score)
        total_keys_in_gt = len(gt_flat)
        matched_keys = 0
        matched_values = 0

        for key, gt_value in gt_flat.items():
            if key in pred_flat:
                matched_keys += 1
                if pred_flat[key] == gt_value:
                    matched_values += 1

        s_k = matched_keys / total_keys_in_gt if total_keys_in_gt > 0 else 1.0
        s_v = matched_values / total_keys_in_gt if total_keys_in_gt > 0 else 1.0

        # Calculate final weighted score
        final_score = 0.2 * s_s + 0.4 * s_k + 0.4 * s_v

        return {
            "s_s (syntax)": s_s,
            "s_k (key_match)": round(s_k, 4),
            "s_v (value_match)": round(s_v, 4),
            "final_score": round(final_score, 4),
        }

    def _evaluate_csv(self, ground_truth_csv_str: str, prediction_csv_str: str) -> dict:
        """Evaluate CSV structure (from test_struct_csv.py)"""
        # Calculate s_s (Syntax Score)
        try:
            pred_lines = list(csv.reader(io.StringIO(prediction_csv_str)))
            s_s = 1.0
            if not pred_lines:
                return {"s_s (syntax)": 1.0, "s_k (key_match)": 0.0, "s_v (value_match)": 0.0, "final_score": 0.2}
        except csv.Error:
            return {"s_s (syntax)": 0.0, "s_k (key_match)": 0.0, "s_v (value_match)": 0.0, "final_score": 0.0}

        # Parse data and extract headers/rows
        gt_lines = list(csv.reader(io.StringIO(ground_truth_csv_str)))

        if not gt_lines:
            is_correct = not pred_lines
            score = 1.0 if is_correct else 0.0
            return {"s_s (syntax)": 1.0, "s_k (key_match)": score, "s_v (value_match)": score, "final_score": score}

        gt_headers = gt_lines[0]
        pred_headers = pred_lines[0]
        gt_rows = gt_lines[1:]
        pred_rows = pred_lines[1:]

        # Calculate s_k (Header Match Score)
        gt_header_set = set(gt_headers)
        pred_header_set = set(pred_headers)
        matched_header_count = len(gt_header_set.intersection(pred_header_set))
        s_k = matched_header_count / len(gt_header_set) if gt_header_set else 1.0

        # Calculate s_v (Value Match Score)
        try:
            pred_header_map = {header: i for i, header in enumerate(pred_headers)}
        except TypeError:
            pred_header_map = {}

        total_cells_in_gt = len(gt_headers) * len(gt_rows)
        matched_cells = 0

        if total_cells_in_gt > 0:
            for r, gt_row in enumerate(gt_rows):
                if r < len(pred_rows):
                    pred_row = pred_rows[r]
                    for c, gt_header in enumerate(gt_headers):
                        gt_cell_value = gt_row[c]
                        if gt_header in pred_header_map:
                            pred_col_index = pred_header_map[gt_header]
                            if pred_col_index < len(pred_row):
                                pred_cell_value = pred_row[pred_col_index]
                                if gt_cell_value == pred_cell_value:
                                    matched_cells += 1
            s_v = matched_cells / total_cells_in_gt
        else:
            s_v = 1.0 if not pred_rows else 0.0

        # Calculate final weighted score
        final_score = 0.2 * s_s + 0.4 * s_k + 0.4 * s_v

        return {
            "s_s (syntax)": s_s,
            "s_k (key_match)": round(s_k, 4),
            "s_v (value_match)": round(s_v, 4),
            "final_score": round(final_score, 4),
        }

    def _parse_judge_response(self, response: str) -> dict:
        """Parse the judge response into a structured format (from base_llm_processor.py)"""
        pattern = re.compile(
            r"(?=.*?extracted_final_answer:\s*(?P<extracted_final_answer>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?reasoning:\s*(?P<reasoning>.*?)(?=\n\s*\w+:|$))?"
            r"(?=.*?correct:\s*(?P<correct>.*?)(?=\n\s*\w+:|$))?",
            re.DOTALL,
        )
        # Remove bold formatting
        response = response.replace("**", "")
        # Search for pattern
        match = pattern.search(response)
        if not match:
            raise ValueError("Invalid judge response format.")

        return {
            "extracted_final_answer": match.group("extracted_final_answer").strip()
            if match.group("extracted_final_answer")
            else "",
            "reasoning": match.group("reasoning").strip() if match.group("reasoning") else "",
            "correct": match.group("correct").strip().lower() == "yes" if match.group("correct") else False,
        }

    def _extract_exact_answer(self, response: str) -> str:
        """Extract the exact answer from the response"""
        return response.strip() if response else ""

    def calculate_metrics(self, samples: list[Sample]) -> dict:
        """Calculate metrics from the judged data"""
        return {
            **MetricsUtils.calculate_overall_metrics(samples),
            **MetricsUtils.calculate_category_metrics(samples),
            **MetricsUtils.calculate_type_metrics(samples),
        }

    def _get_file_prompt(self, files: list) -> str:
        if not files:
            return ""
        files = [
            self.config.data_dir / "files" / file
            for file in files
        ]
        files = [file.as_posix() for file in files]

        return self._format_available_files(files)

    def _format_available_files(self, files: list) -> str:
        local_files = [file for file in files if not file.startswith(("http://", "https://"))]

        if not local_files:
            return ""

        header = "\nTo solve the task above, you will have to use these attached files."
        file_list_str = "\n".join(f"- {file}" for file in local_files)
        note = "Note: If you need sub-agent to read the files, you must provide the **original complete file path**."
        return f"{header}\n{file_list_str}\n{note}"
