def qualitative_score(value: str | None) -> int | None:
    """
    Converts qualitative environmental assessments into
    standardized scores for the Restoration Brief.
    """

    if value is None:
        return None

    value = value.strip().lower()

    mapping = {
        "very high": 95,
        "high": 85,
        "moderate": 65,
        "medium": 65,
        "low": 35,
        "very low": 15,
    }

    return mapping.get(value)