from typing import Dict, Any
from core.utils import AgentResponse, Agent
import re


class AccountAgent(Agent):
    name = "account"

    def __init__(self, orders, customers):
        self.orders = orders
        self.customers = customers

    def extract_order_id(self, text: str):
        match = re.search(r"\b(\d{3,6})\b", text)
        return match.group(1) if match else None

    def handle(self, user_message: str, context: Dict[str, Any]) -> AgentResponse:
        text = user_message.lower()

        # Extract order ID
        order_id = self.extract_order_id(text)

        # If no order ID provided
        if not order_id:
            return AgentResponse(
                text="Could you provide your order ID? For example: 1001.",
                next_agent=None,
                metadata={"needs_order_id": True}
            )

        # If order ID provided but not found
        if order_id not in self.orders:
            return AgentResponse(
                text=f"I couldn’t find order {order_id} in our system.",
                next_agent=None,
                metadata={"order_found": False}
            )

        # If order exists
        order = self.orders[order_id]
        status = order["status"]
        created = order["created_at"]
        amount = order["total_amount"]

        return AgentResponse(
            text=f"Order {order_id} is currently: {status}. It was created on {created} for ${amount}.",
            next_agent=None,
            metadata={"order_found": True}
        )
