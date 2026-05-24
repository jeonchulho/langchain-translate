from typing import List, Tuple

# (pattern, flags)
DEFAULT_SPECIAL_ENTITY_PATTERNS: List[Tuple[str, int]] = [
    # URL
    (r"\b(?:https?://|www\.)[^\s<>()\]\[\"']+", 0),
    # Email
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0),
    # Unix/Windows-like path
    (r"\b(?:[A-Za-z]:\\\\|/)?(?:[\w.-]+[\\/])+[\w.-]+\b", 0),
    # snake_case / UPPER_SNAKE / dotted identifiers
    (r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][A-Z0-9_]{2,}|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)\b", 0),
    # camelCase / PascalCase identifier
    (r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b", 0),
]
