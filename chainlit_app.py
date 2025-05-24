

# chainlit_app.py
import os
import httpx
import chainlit as cl

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")



# Take the message wrote in UI and send it to the backend. Then display the result from the backend
@cl.on_message
async def handle_message(message: cl.Message):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BACKEND_URL}/chat",
            json={"message": message.content}
        )
        if resp.status_code != 200:
            await cl.Message(content="⚠️ Error contacting backend.").send()
            return

        text = resp.text()
        await cl.Message(content=text).send()



if __name__ == "__main__":
    import sys
    # Replace the current process with: chainlit run chainlit_app.py
    os.execvp("chainlit", ["chainlit", "run", sys.argv[0]])
