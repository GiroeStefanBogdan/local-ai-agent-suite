import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.responses import PlainTextResponse


# ─── LLM class import ────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Load environment variables ──────────────────────────────────────────
load_dotenv()  # expects OPENAI_API_KEY, GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID

# ─── Instantiate the LLM ─────────────────────────────────────────────────
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True
)

# ─── Define Python tools ─────────────────────────────────────────────────
def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return their product."""
    return a * b

def search_and_fetch(query: str) -> str:
    """
    Uses Google Custom Search JSON API to fetch top-3 snippets for `query`.
    Returns them joined by two newlines, or an error message.
    """
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": os.getenv("GOOGLE_SEARCH_API_KEY"),
                "cx":  os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                "q":   query,
                "num": 3
            },
            timeout=5
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return "No results found."
        return "\n\n".join(item["snippet"] for item in items)
    except Exception as e:
        return f"Search error: {e}"

# ─── Build the REACT agents ───────────────────────────────────────────────
math_expert = create_react_agent(
    model=llm,
    name="math_expert",
    tools=[add, multiply],
    prompt=(
        "You are **math_expert**. You may *only* call `add(a,b)` and `multiply(a,b)`.\n"
        "If it’s not addition or multiplication, respond: “I cannot do that.”\n"
        "Prefix your answer with “Math_expert: ” when you do call a tool."
    )
)

search_expert = create_react_agent(
    model=llm,
    name="search_expert",
    tools=[search_and_fetch],
    prompt=(
        "You are **search_expert**, an expert web researcher.\n"
        "• ALWAYS call `search_and_fetch(query)` to look something up on Google.\n"
        "• Summarize only from that output.\n"
        "• Prefix your answer with “search_expert: ” when you respond."
    )
)


# ─── Supervisor that routes to the right expert ─────────────────────────
workflow = create_supervisor(
    agents=[math_expert, search_expert],
    model=llm,
    prompt=(
        "You are a team supervisor:\n"
        "• math_expert handles all arithmetic.\n"
        "• search_expert handles everything else via search_and_fetch().\n"
        "Route each user request appropriately."
    )
)

app_workflow = workflow.compile()

# ─── FastAPI setup ───────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat", response_class=PlainTextResponse)
def chat(req: ChatRequest) -> str:
    inputs = {"messages": [{"role": "user", "content": req.message}]}
    response = app_workflow.invoke(inputs)
    message = response.get("messages", [])
    lines = [
        f"{getattr(m, 'name', 'assistant')}: {m.content}"
        for m in message
    ]
    # Join with actual newline characters
    return "\n".join(lines)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
