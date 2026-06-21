"""PhySciBench standalone offline scorer (Release 1).

Clean-room package: imports nothing from `core/agents`, `core/context`,
`core/db`, `tools/*`, the OpenAI Agents SDK, `sqlmodel`, or selenium/pyvirtualdisplay.
"""

__all__ = ["score", "judge", "metrics", "data", "config", "sandbox", "prompts"]
