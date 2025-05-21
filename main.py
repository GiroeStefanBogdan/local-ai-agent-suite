# main.py

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

# ─── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Load env & define tools ─────────────────────────────────────────────
load_dotenv()

def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b

math_expert = create_react_agent(
    model="openai:gpt-4o",
    tools=[add, multiply],
    prompt="You are **math_expert**. You ONLY may call `add(a,b)` and `multiply(a,b)`. If something is not related to adding or mulitplying then you should say 'I cannot do that'. If you can use add or multiply answer with this before your answer: 'Math_expert"
)

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
async def chat(req: ChatRequest):
    inputs = {"messages": [{"role": "user", "content": req.message}]}

    result = ""
    for update in math_expert.stream(inputs, stream_mode="updates"):
        logger.info(f"Raw agent update: {update!r}")

        # 1) tool outputs
        tools_block = update.get("tools", {})
        for tm in tools_block.get("messages", []):
            # tm may be a ToolMessage object with .content
            piece = getattr(tm, "content", None) or tm.get("content", "")
            logger.info(f" → tool content: {piece!r}")
            result += piece

        # 2) agent messages
        agent_block = update.get("agent", {})
        for am in agent_block.get("messages", []):
            # am is likely an AIMessage with .content
            piece = getattr(am, "content", "")
            logger.info(f" → agent content: {piece!r}")
            result += piece

    if not result:
        logger.warning("No content was extracted from any update!")

    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
