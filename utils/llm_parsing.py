import re

# ======================
# MAIN FUNCTION
# ======================
def parse_llm_output(text: str) -> tuple[str | None, float | None]:
    """
    Extracts reasoning and answer from the LLM output.
    Returns (reasoning, answer).
    """
    reasoning = None
    answer = None

    '''
    reasoning_match = re.search(
        r"\[\[\s*## reasoning ##\s*\]\](.*?)\[\[\s*## answer ##\s*\]\]",
        text,
        re.DOTALL | re.IGNORECASE
    )

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    '''

    answer_match = re.search(
        r"\[\[\s*## answer ##\s*\]\]\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE
    )

    if answer_match:
        answer = float(answer_match.group(1))

    return reasoning, answer
