"""e2b code-execution wrapper with graceful degradation.

If `e2b_code_interpreter` is not importable OR no `E2B_API_KEY` is set, the
sandbox is considered unavailable: `e2b_available()` returns False and the
judging code skips + annotates `code-generation` items (counted under
`skipped`) while every other type still scores.

When available, `execute_code` runs the source `_execute_code` logic VERBATIM
(create sandbox -> `pip install qutip` -> `run_code` -> collect stdout -> kill).
Any sandbox exception is caught and returned as a per-sample error string so a
single bad sample never crashes the whole run.
"""

import logging
import os

from ._extract import extract_code_re

logger = logging.getLogger(__name__)

# Sentinel returned for code-generation samples when e2b is unavailable.
SANDBOX_UNAVAILABLE = "__E2B_SANDBOX_UNAVAILABLE__"

try:  # optional dependency
    from e2b_code_interpreter import AsyncSandbox

    _E2B_IMPORTABLE = True
except Exception:  # pragma: no cover - exercised only when extra is absent
    AsyncSandbox = None  # type: ignore[assignment]
    _E2B_IMPORTABLE = False


def e2b_available() -> bool:
    """True only when the e2b package is importable AND an API key is configured."""
    return _E2B_IMPORTABLE and bool(os.getenv("E2B_API_KEY"))


async def execute_code(code: str) -> str:
    """Execute Python `code` in an e2b sandbox and return its stdout.

    Returns `SANDBOX_UNAVAILABLE` when e2b is not usable. Mirrors the source
    `PhySciBenchProcesser._execute_code` body for the extraction + run logic.
    """
    if not e2b_available():
        return SANDBOX_UNAVAILABLE

    code_clean = code.strip()

    extracted = extract_code_re(code_clean)
    if extracted:
        code_clean = extracted.strip()
    elif code_clean.startswith("```"):
        lines = code_clean.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        code_clean = "\n".join(lines)

    sbx = None
    try:
        sbx = await AsyncSandbox.create()
        await sbx.commands.run("pip install qutip")
        execution = await sbx.run_code(code_clean)
        result = "".join(execution.logs.stdout)
        return result.strip()
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return f"Code execution error: {str(e)}"
    finally:
        if sbx is not None:
            try:
                await sbx.kill()
            except Exception as e:  # pragma: no cover - cleanup best effort
                logger.error(f"Error killing sandbox: {e}")
