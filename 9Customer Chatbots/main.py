from agents.account_agent import AccountAgent
from agents.faq_agent import FAQAgent
from agents.triage_agent import TriageAgent
from agents.escalation_agent import EscalationAgent


import csv
from agents import (
    TriageAgent,
    FAQAgent,
    AccountAgent,
    EscalationAgent
)
from core.orchestrator import Orchestrator
from core.embeddings import simple_embed


# -----------------------------
# Data loading helpers
# -----------------------------

def load_faq_kb(path: str):
    kb = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        #print("CSV columns:", reader.fieldnames)
        quit
        for row in reader:
            row["embedding"] = simple_embed(row["question"] + " " + row["answer"])
            kb.append(row)
    print("FAQ rows loaded:", len(kb)) # Test CSV load
    return kb


def load_orders(path: str):
    orders = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
       
        for row in reader:
            orders[row["order_id"]] = row
    print("Orders loaded:", len(orders))
    return orders


def load_customers(path: str):
    customers = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        #quit
        # print("Customers loaded:", len(customers))

        for row in reader:
            customers[row["customer_id"]] = row
    print("Customers loaded:", len(customers))
    return customers


# -----------------------------
# Build system
# -----------------------------

def build_system():
    faq_kb = load_faq_kb("data/faq_kb.csv")
    orders = load_orders("data/orders.csv")
    customers = load_customers("data/customers.csv")

    agents = {
        "triage": TriageAgent(),
        "faq": FAQAgent(faq_kb),
        "account": AccountAgent(orders, customers),
        "escalation": EscalationAgent(),
    }

    return Orchestrator(agents)


# -----------------------------
# Main loop
# -----------------------------

if __name__ == "__main__":
    orchestrator = build_system()
    print("Multi-agent Customer Service Bot")
    print("Type 'quit' to exit.\n")

    while True:
        user = input("You: ")
        if user.strip().lower() in ("quit", "exit"):
            break

        bot_reply = orchestrator.route(user)
        print(f"Bot: {bot_reply}\n")