from agents.faq_agent import FAQAgent
from core.embeddings import simple_embed


def test_faq_exact_match():
    kb = [
        {"question": "What is your return policy?",
         "answer": "You can return items within 30 days.",
         "embedding": simple_embed("What is your return policy? You can return items within 30 days.")}
    ]

    agent = FAQAgent(kb)
    resp = agent.handle("What is your return policy?", {})
    assert "return policy" in resp.text.lower()


def test_faq_no_match():
    kb = [
        {"question": "Do you ship internationally?",
         "answer": "Yes, we ship worldwide.",
         "embedding": simple_embed("Do you ship internationally? Yes, we ship worldwide.")}
    ]

    agent = FAQAgent(kb)
    resp = agent.handle("How do I cancel my subscription?", {})

    assert "couldn’t find" in resp.text.lower()
    assert resp.next_agent == "escalation"
