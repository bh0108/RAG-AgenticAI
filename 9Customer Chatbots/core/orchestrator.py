from typing import Dict, Any
from core.utils import AgentResponse, Agent


class Orchestrator:
    """
    Central router that coordinates all agents.
    Maintains conversation context and agent transitions.
    """

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents
        self.context: Dict[str, Any] = {}

    def route(self, user_message: str) -> str:
        """
        Routes the user message to the appropriate agent.
        """

        # Always start with triage for every NEW message
        triage_agent = self.agents["triage"]
        triage_response = triage_agent.handle(user_message, self.context)

        # Update context with new intent
        self.context["intent"] = triage_response.metadata.get("intent")

        # Determine next agent from triage
        next_agent_name = triage_response.next_agent

        # SAFETY: Clear any stale next_agent from previous turns
        self.context["next_agent"] = next_agent_name

        # Now call the correct next agent immediately
        next_agent = self.agents[next_agent_name]
        final_response = next_agent.handle(user_message, self.context)

        # Clear next_agent unless the agent sets a new one
        if final_response.next_agent:
            self.context["next_agent"] = final_response.next_agent
        else:
            self.context.pop("next_agent", None)

        return final_response.text

