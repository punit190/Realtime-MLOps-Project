import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
# CHANGE THIS IMPORT:
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="LLMOps Retention Agent Service")

# CHANGE THIS LINE: Use a lightweight free local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(collection_name="retention_rules", embedding_function=embeddings)

@app.on_event("startup")
def populate_vector_db():
    sample_texts = [
        "High tenure customers with contract complaints should be offered a 25% discount for 6 months.",
        "Technical friction users must be immediately fast-tracked to a senior account specialist.",
        "Competitor price matching inquiries should receive free tier features or complementary tier upgrades."
    ]
    vector_db.add_texts(texts=sample_texts)

class ChurnSignal(BaseModel):
    customer_id: str
    predicted_churn: int 

class AgentState(TypedDict):
    customer_id: str
    context: str
    generated_email: str

def retrieve_retention_playbook(state: AgentState) -> dict:
    query = "retention strategies for high contract pricing and technical issues"
    docs = vector_db.similarity_search(query, k=1)
    context_text = docs[0].page_content if docs else "Offer standard customer loyalty save package."
    return {"context": context_text}

def generate_retention_email(state: AgentState) -> dict:
    """Bypasses the exhausted OpenAI API to return a data-driven, free mock response."""
    customer_id = state.get("customer_id", "CUSTOMER-XYZ-12")
    context = state.get("context", "Standard retention protocols apply.")
    
    mock_llm_response = (
        f"Subject: Priority Retention Offer - Account {customer_id}\n\n"
        f"Dear Customer ({customer_id}),\n\n"
        f"We value your partnership. Based on our corporate playbook guidelines:\n"
        f"👉 \"{context}\"\n\n"
        f"We have applied these strategic optimization benefits directly to your account profile.\n\n"
        f"Best regards,\n"
        f"Customer Retention Specialist Team"
    )
    
    # Return the dictionary back to LangGraph
    return {"agent_output": mock_llm_response}
    # Note: If your OpenAI key has zero credits, this step will also return a 429 error.
    # To practice 100% free without an OpenAI key, you can mock this return text or use a free LLM provider.
    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    #prompt = ChatPromptTemplate.from_template(
        #"You are an elite customer success supervisor. Write a brief email to "
        #"customer {customer_id} based on this company protocol: {context}. "
        #"Keep the language warm, personalized, and action-oriented."
    #)
    #chain = prompt | llm
    #response = chain.invoke({"customer_id": state["customer_id"], "context": state["context"]})
    #return {"generated_email": response.content}

# --- State Machine Compilation ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_playbook", retrieve_retention_playbook)
workflow.add_node("generate_email", generate_retention_email)

workflow.set_entry_point("retrieve_playbook")
workflow.add_edge("retrieve_playbook", "generate_email")
workflow.add_edge("generate_email", END)
compiled_agent = workflow.compile()

@app.post("/generate-retention")
def trigger_agent_pipeline(signal: ChurnSignal):
    if signal.predicted_churn == 1:
        initial_state = {"customer_id": signal.customer_id, "context": "", "generated_email": ""}
        output = compiled_agent.invoke(initial_state)
        return {
            "action": "Trigger Retention Campaign",
            "agent_output": output["generated_email"]
        }
    return {"action": "Log Event", "agent_output": "Customer risk stable. No action taken."}