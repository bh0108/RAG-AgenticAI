from agents.triage_agent import TriageAgent


def test_triage_faq():
    agent = TriageAgent()
    resp = agent.handle("What is your return policy?", {})
    assert resp.next_agent == "faq"
    assert resp.metadata["intent"] == "general_faq"


def test_triage_order():
    agent = TriageAgent()
    resp = agent.handle("Where is my order 1001?", {})
    assert resp.next_agent == "account"
    assert resp.metadata["intent"] == "order_status"


def test_triage_billing():
    agent = TriageAgent()
    resp = agent.handle("I want a refund", {})
    assert resp.next_agent == "account"
    assert resp.metadata["intent"] == "billing_issue"


def test_triage_escalation():
    agent = TriageAgent()
    resp = agent.handle("I want to talk to a human", {})
    assert resp.next_agent == "escalation"
    assert resp.metadata["intent"] == "escalation"
