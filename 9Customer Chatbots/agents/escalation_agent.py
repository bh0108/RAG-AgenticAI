from typing import Dict, Any
from core.utils import AgentResponse, Agent


class EscalationAgent(Agent):
    name = "escalation"

    def handle(self, user_message: str, context: Dict[str, Any]) -> AgentResponse:
        summary = (
            f"Customer message: '{user_message}'. "
            f"Intent: {context.get('intent', 'unknown')}."
        )

        return AgentResponse(
            text=(
                "I’m escalating this to a human representative. "
                "Here’s a summary of your issue:\n\n"
                f"{summary}"
            ),
            next_agent=None,
            metadata={"escalated": True, "summary": summary}
        )