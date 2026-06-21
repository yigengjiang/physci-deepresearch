"""Prompt loading for the standalone scorer.

Replaces `core/utils/path.py:FileUtils.load_prompts` (which transitively imports
selenium / pyvirtualdisplay) with a tiny `yaml.safe_load` over the vendored
`eval/judge_templates.yaml`. No `core/*` import.
"""

import pathlib

import yaml

_PROMPTS_PATH = pathlib.Path(__file__).parent / "judge_templates.yaml"


def load_prompts(path: str | pathlib.Path | None = None) -> dict:
    """Load judge prompt templates from a YAML file.

    Defaults to the vendored `eval/judge_templates.yaml`, which retains the
    top-level `default:` key required by the ported judging code.
    """
    p = pathlib.Path(path) if path is not None else _PROMPTS_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)
