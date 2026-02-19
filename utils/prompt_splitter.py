from typing import List

# ======================
# CONSTANTS
# ======================
SPLIT_MARKER = {
    "PART": "===PART:"
}

# ======================
# MAIN FUNCTION
# ======================
def split_prompt_by_markers(prompt: str) -> List[str]:
    """
    Splits a prompt into multiple parts based on explicit markers.
    Returns a list of prompt strings, in order. If no markers are found, returns [prompt].
    """

    if SPLIT_MARKER["PART"] not in prompt:
        return [prompt]

    lines = prompt.splitlines()
    parts = []
    current = []

    for line in lines:
        if line.startswith(SPLIT_MARKER["PART"]):
            if current:
                parts.append("\n".join(current).strip())
                current = []
        current.append(line)

    if current:
        parts.append("\n".join(current).strip())

    return parts
