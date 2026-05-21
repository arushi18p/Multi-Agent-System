def decision_agent(state: dict) -> dict:
    if state.get("error"):
        return state

    risk_score = state.get("risk_score", 0)
    is_fraud = state.get("is_fraud", 0)

    if is_fraud == 1 or risk_score > 0.8:
        decision = "BLOCK"
        reason = "High risk score or confirmed fraud flag"
    elif risk_score > 0.5:
        decision = "FLAG FOR REVIEW"
        reason = "Moderate risk — requires human review"
    else:
        decision = "APPROVE"
        reason = "Low risk transaction"

    return {
        **state,
        "decision": decision,
        "decision_reason": reason,
        "status": "decision_complete"
    }