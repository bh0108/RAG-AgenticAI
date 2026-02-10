from core.orchestrator import Orchestrator
from agents.triage_agent import TriageAgent
from agents.faq_agent import FAQAgent
from agents.account_agent import AccountAgent
from agents.escalation_agent import EscalationAgent
from core.embeddings import simple_embed


def build_test_system():
    faq_kb = [
        {"question": "What is your return policy?",
         "answer": "You can return items within 30 days.",
         "embedding": simple_embed("What is your return policy? You can return items within 30 days.")}
    ]

    orders = {
        "1001": {"status": "Shipped", "created_at": "2025-01-10", "total_amount": "79.99"}
    }

    customers = {}

    agents = {
        "triage": TriageAgent(),
        "faq": FAQAgent(faq_kb),
        "account": AccountAgent(orders, customers),
        "escalation": EscalationAgent()
    }

    return Orchestrator(agents)


def test_orchestrator_faq():
    orch = build_test_system()
    resp = orch.route("What is your return policy?")
    assert "return" in resp.lower()


def test_orchestrator_order_found():
    orch = build_test_system()
    resp = orch.route("Track order 1001")
    assert "order 1001" in resp.lower()


def test_orchestrator_order_not_found():
    orch = build_test_system()
    resp = orch.route("Track order 999")
    assert "couldn’t find order 999" in resp.lower()
