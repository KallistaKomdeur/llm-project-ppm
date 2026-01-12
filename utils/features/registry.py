def build_feature_block(
    configuration: str,
    *,
    single_case: str,
    global_features: str | None,
    inter_case: str | None
) -> str:
    if configuration == "global_only":
        return "\n\n".join(b for b in [single_case, global_features] if b)

    return single_case
