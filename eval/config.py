"""Minimal local configuration for the standalone scorer.

No agent/DB config. Judge-model settings come from the environment, canonical
`JUDGE_LLM_*` taking precedence, then generic `OPENAI_*`, then legacy
`UTU_LLM_*`. Exposes a structure compatible with the ported judging code:
`config.judge_model.model_provider.{type,model,base_url,api_key}`,
`config.judge_model.model_params.{temperature,top_p}`,
`config.data.{dataset,question_field,gt_field,files_field}`, and
`config.data_dir`.
"""

import os
import pathlib

from pydantic import BaseModel, Field


def _env(*names: str, default: str | None = None) -> str | None:
    """Return the first set environment variable among `names` (precedence order)."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


class ModelProvider(BaseModel):
    """Provider connection settings passed straight into `JudgeClient(**...)`."""

    type: str = "chat.completions"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class ModelParams(BaseModel):
    """Sampling params forwarded to the judge create/parse calls."""

    temperature: float = 0.0
    top_p: float = 1.0


class JudgeModel(BaseModel):
    model_provider: ModelProvider = Field(default_factory=ModelProvider)
    model_params: ModelParams = Field(default_factory=ModelParams)


class DataConfig(BaseModel):
    dataset: str = "PhySciBench"
    question_field: str = "question"
    gt_field: str = "answer"
    files_field: str = "files"


class ScorerConfig(BaseModel):
    judge_model: JudgeModel = Field(default_factory=JudgeModel)
    data: DataConfig = Field(default_factory=DataConfig)
    data_dir: pathlib.Path = pathlib.Path("data/PhySciBench")

    @classmethod
    def from_env(
        cls,
        *,
        dataset: str = "PhySciBench",
        data_dir: str | pathlib.Path = "data/PhySciBench",
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> "ScorerConfig":
        """Build a config from environment variables.

        Env precedence per field: `JUDGE_LLM_*` > `OPENAI_*` > legacy `UTU_LLM_*`.
        """
        provider = ModelProvider(
            type=_env("JUDGE_LLM_TYPE", "UTU_LLM_TYPE", default="chat.completions"),
            model=_env("JUDGE_LLM_MODEL", "OPENAI_MODEL", "UTU_LLM_MODEL"),
            base_url=_env("JUDGE_LLM_BASE_URL", "OPENAI_BASE_URL", "UTU_LLM_BASE_URL"),
            api_key=_env("JUDGE_LLM_API_KEY", "OPENAI_API_KEY", "UTU_LLM_API_KEY"),
        )
        return cls(
            judge_model=JudgeModel(
                model_provider=provider,
                model_params=ModelParams(temperature=temperature, top_p=top_p),
            ),
            data=DataConfig(dataset=dataset),
            data_dir=pathlib.Path(data_dir),
        )
