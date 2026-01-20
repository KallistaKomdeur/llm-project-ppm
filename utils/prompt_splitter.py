from typing import List

# ======================
# CONSTANTS
# ======================
SPLIT_MARKERS = {
    "PART": "===PART:",
    "FINAL": "===FINAL QUERY==="
}

# ======================
# MAIN FUNCTION
# ======================
def split_prompt_by_markers(prompt: str) -> List[str]:
    """
    Splits a prompt into multiple parts based on explicit markers.
    Returns a list of prompt strings, in order. If no markers are found, returns [prompt].
    """

    if SPLIT_MARKERS["PART"] not in prompt:
        return [prompt]

    lines = prompt.splitlines()
    parts = []
    current = []

    for line in lines:
        if line.startswith(SPLIT_MARKERS["PART"]):
            if current:
                parts.append("\n".join(current).strip())
                current = []
        current.append(line)

    if current:
        parts.append("\n".join(current).strip())

    return parts
