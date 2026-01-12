def build_feature_block(
    configuration: str,
    *,
    single_case: str,
    global_features: str | None,
    inter_case: str | None
) -> str:

    blocks = []

    if configuration in {"single", "global_only", "inter-case_only"}:
        blocks.append(single_case)

    if configuration in {"global_only", "inter-case_only"}:
        blocks.append(global_features)

    if configuration == "inter-case_only":
        blocks.append(inter_case)

    return "\n\n".join(b for b in blocks if b)
