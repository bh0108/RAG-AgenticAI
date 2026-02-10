from typing import Dict, Any, List
from core.utils import AgentResponse, Agent
from core.embeddings import simple_embed, cosine_sim


class FAQAgent(Agent):
    name = "faq"

    def __init__(self, kb: List[Dict[str, Any]]):
        self.kb = kb

    def search_kb(self, query: str, top_k: int = 3):
        q_emb = simple_embed(query)
        scored = []

        for row in self.kb:
            score = cosine_sim(q_emb, row["embedding"])
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def handle(self, user_message: str, context: Dict[str, Any]) -> AgentResponse:
        results = self.search_kb(user_message)

        best_score, best_row = results[0]

        # 🔥 SIMILARITY THRESHOLD — prevents wrong answers
        SIMILARITY_THRESHOLD = 0.85

        if best_score < SIMILARITY_THRESHOLD:
            return AgentResponse(
                text="I couldn’t find an exact answer to that. Would you like me to escalate this?",
                next_agent="escalation",
                metadata={"faq_found": False, "similarity": best_score}
            )

                
        return AgentResponse(
            text=f"Here’s what I found:\n\nQ: {best_row['question']}\nA: {best_row['answer']}",
            next_agent=None,
            metadata={"faq_found": True, "similarity": best_score}
        )
