from dataclasses import dataclass, field
from typing import Dict, Any
from core.utils import AgentResponse, Agent


class TriageAgent(Agent):
    name = "triage"

    def handle(self, user_message: str, context: Dict[str, Any]) -> AgentResponse:
        text = user_message.lower()

        # 1. FAQ detection FIRST (policy, shipping, general questions)
        faq_keywords = [
            "policy", "return policy", "shipping policy", "how do i", "what is",
            "do you", "can i", "faq", "information", "details"
        ]
        if any(k in text for k in faq_keywords):
            intent = "general_faq"
            next_agent = "faq"
            return AgentResponse(
                text="Routing to FAQ agent.",
                next_agent=next_agent,
                metadata={"intent": intent}
                #print(text)
            )

        # 2. Order-related
        if any(k in text for k in ["order", "tracking", "shipment", "delivery"]):
            intent = "order_status"
            next_agent = "account"
            return AgentResponse(
                text="Routing to Account agent.",
                next_agent=next_agent,
                metadata={"intent": intent}
                #print(text)
            )

        # 3. Billing/refund
        if any(k in text for k in ["refund", "exchange"]):
            intent = "billing_issue"
            next_agent = "account"
            return AgentResponse(
                text="Routing to Account agent.",
                next_agent=next_agent,
                metadata={"intent": intent}
                #print(text)

            )

        # 4. Escalation
        if any(k in text for k in ["agent", "human", "representative"]):
            intent = "escalation"
            next_agent = "escalation"
            return AgentResponse(
                text="Routing to human support.",
                next_agent=next_agent,
                metadata={"intent": intent}
                #print(text)
            )

        # 5. Default → FAQ
        intent = "general_faq"
        next_agent = "faq"
        return AgentResponse(
            text="Routing to FAQ agent.",
            next_agent=next_agent,
            metadata={"intent": intent}
            #print(text)
        )
