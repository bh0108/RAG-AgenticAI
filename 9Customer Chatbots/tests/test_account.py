from agents.account_agent import AccountAgent


def test_account_missing_order_id():
    agent = AccountAgent(orders={}, customers={})
    resp = agent.handle("Where is my order?", {})
    assert "provide your order id" in resp.text.lower()


def test_account_order_not_found():
    orders = {"1001": {"status": "Shipped", "created_at": "2025-01-10", "total_amount": "79.99"}}
    agent = AccountAgent(orders, customers={})

    resp = agent.handle("Track order 999", {})
    assert "couldn’t find order 999" in resp.text.lower()


def test_account_order_found():
    orders = {"1001": {"status": "Shipped", "created_at": "2025-01-10", "total_amount": "79.99"}}
    agent = AccountAgent(orders, customers={})

    resp = agent.handle("Track order 1001", {})
    assert "order 1001" in resp.text.lower()
    assert "shipped" in resp.text.lower()
