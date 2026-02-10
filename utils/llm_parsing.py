import re

# ======================
# MAIN FUNCTION
# ======================
def parse_llm_output(text: str) -> tuple[str | None, float | None]:
    """
    Extracts numerical answer from the LLM output.
    """
    answer = None

    answer_match = re.search(
        r"\[\[\s*## answer ##\s*\]\]\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE
    )

    if answer_match:
        answer = float(answer_match.group(1))

    return answer
