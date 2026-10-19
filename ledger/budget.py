"""Split a prompt file into the four budget sections."""

SECTION_ORDER = ("system", "history", "retrieved", "question")
_SECTION_NAMES = set(SECTION_ORDER)


def header_name(line: str) -> str | None:
    """Return the section name when `line` is a budget header, else None.

    Header lines look like `--- system`. The header text itself is not part
    of any section body.
    """
    stripped = line.strip()
    if not stripped.startswith("---"):
        return None
    name = stripped[3:].strip()
    if name in _SECTION_NAMES:
        return name
    return None


def parse_sections(text: str) -> dict[str, str]:
    """Collect section bodies, preserving newlines and dropping header lines.

    Text before the first header is ignored. Each body is counted on its own
    later, so a merge cannot cross from one section into the next. Missing
    sections are empty strings.
    """
    bodies = {name: [] for name in SECTION_ORDER}
    current: str | None = None
    for line in text.splitlines(keepends=True):
        name = header_name(line)
        if name is not None:
            current = name
            continue
        if current is not None:
            bodies[current].append(line)
    return {name: "".join(parts) for name, parts in bodies.items()}
