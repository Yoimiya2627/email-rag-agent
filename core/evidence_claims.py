"""Deterministic claim/gold record validation, explicitly not entailment scoring."""
def validate_claim_records(record, visible_sources):
    """Validate reviewer/model-declared claims against exact visible source refs.

    This does not decide whether a sentence follows from the text, whether a
    numeric value is correct, or which conflicting conclusion remains valid.
    Human-reviewed gold records and real-model quality evaluation remain needed.
    """
    if type(record) is not dict or set(record) != {"claims", "conflicts"}:
        raise ValueError("claim record requires claims and conflicts")
    claims, conflicts = record["claims"], record["conflicts"]
    if type(claims) is not list or len(claims) > 100 or type(conflicts) is not list or len(conflicts) > 100:
        raise ValueError("claim/conflict record exceeds its bounded format")
    fields = ("email_id", "chunk_id", "source_version", "visible_start", "visible_end", "visible_hash")
    available = {tuple(source.get(key) for key in fields) for source in visible_sources}
    ids, unknown_refs = set(), []
    for claim in claims:
        if type(claim) is not dict or set(claim) != {"claim_id", "text", "assessment", "evidence_refs"}:
            raise ValueError("invalid claim fields")
        if (type(claim["claim_id"]) is not str or not 1 <= len(claim["claim_id"]) <= 100
                or claim["claim_id"] in ids or type(claim["text"]) is not str
                or not 1 <= len(claim["text"]) <= 2000):
            raise ValueError("invalid claim identifier or text")
        ids.add(claim["claim_id"])
        if type(claim["assessment"]) is not str or claim["assessment"] not in {"supported", "contradicted", "insufficient", "unverified"}:
            raise ValueError("invalid declared claim assessment")
        refs = claim["evidence_refs"]
        if type(refs) is not list or len(refs) > 30:
            raise ValueError("invalid claim evidence list")
        if claim["assessment"] in {"supported", "contradicted"} and not refs:
            unknown_refs.append(claim["claim_id"])
        for ref in refs:
            if (type(ref) is not dict or set(ref) != set(fields)
                    or any(type(ref[key]) is not str or not 1 <= len(ref[key]) <= 500
                           for key in ("email_id", "chunk_id", "source_version", "visible_hash"))
                    or type(ref["visible_start"]) is not int or type(ref["visible_end"]) is not int
                    or not 0 <= ref["visible_start"] < ref["visible_end"]):
                raise ValueError("invalid claim evidence reference")
            if tuple(ref.get(key) for key in fields) not in available:
                unknown_refs.append(claim["claim_id"])
    for conflict in conflicts:
        if type(conflict) is not dict or set(conflict) != {"claim_ids", "relation"}:
            raise ValueError("invalid conflict fields")
        linked = conflict["claim_ids"]
        if (type(linked) is not list or not 2 <= len(linked) <= 20
                or any(type(key) is not str or key not in ids for key in linked)
                or len(linked) != len(set(linked)) or type(conflict["relation"]) is not str
                or conflict["relation"] not in {"conflict", "supersedes"}):
            raise ValueError("conflict must link existing distinct claims")
    return {"format_valid": True, "all_references_visible": not unknown_refs,
            "claims_with_missing_evidence": sorted(set(unknown_refs)),
            "claim_count": len(claims), "conflict_count": len(conflicts),
            "entailment_checked": False, "factual_correctness": "not_measured",
            "assessment_source": "declared_not_independently_verified"}
