from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AgentResponse:
    text: str
    next_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    name: str = "base_agent"

    def handle(self, user_message: str, context: Dict[str, Any]) -> AgentResponse:
        raise NotImplementedError("Each agent must implement the handle() method.")