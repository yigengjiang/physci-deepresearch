"""Regex extractors for fenced code blocks.

Copied VERBATIM from `core/utils/common.py` (origin/submit): `extract_json_re`,
`extract_csv_re`, `extract_code_re`. Imports only `re` — leak-free.
"""

import re


def extract_json_re(text)->str:
    # Find JSON portion between triple backticks
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    return None


def extract_csv_re(text)->str:
    # Find CSV portion between triple backticks
    match = re.search(r'```csv\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        csv_str = match.group(1)
        return csv_str
    return None


def extract_code_re(text: str) -> str:
    """Extract code from markdown code block"""
    # First try to find python code block
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)

    # If not found, try to find any code block
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)

    return None
