import os
import json
import uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL")

# Initialize FastAPI and OpenAI Client
app = FastAPI()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Setup Logging Directory
LOG_DIR = "public_logs"
os.makedirs(LOG_DIR, exist_ok=True)
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")

# In-memory storage for multi-turn conversations
# Format: { chat_id: [{"role": "user", "content": "..."}, ...] }
conversation_history = {}

# --- HELPER FUNCTIONS ---

def write_log(run_id: str, action: str, details: dict):
    """Writes execution steps to a publicly accessible JSONL file."""
    log_file = os.path.join(LOG_DIR, f"{run_id}.jsonl")
    log_entry = {"run_id": run_id, "action": action, "details": details}
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

async def analyze_data_with_llm(chat_id: int, user_message: str, run_id: str) -> dict:
    """Handles the multi-turn logic and calls the LLM to get the answer."""
    
    # 1. Manage Multi-turn History
    if chat_id not in conversation_history:
        # System prompt enforces strict JSON output
        conversation_history[chat_id] = [{
            "role": "system", 
            "content": (
                "You are a Data Analyst AI. Answer the user's data questions. "
                "You must reply ONLY with a valid JSON object representing the answer in the shape the user requested. "
                "Do NOT use markdown code blocks (like ```json). Just return the raw JSON object."
            )
        }]
    
    conversation_history[chat_id].append({"role": "user", "content": user_message})
    
    # Keep history manageable (last 10 messages)
    if len(conversation_history[chat_id]) > 11: 
        conversation_history[chat_id] = [conversation_history[chat_id][0]] + conversation_history[chat_id][-10:]

    write_log(run_id, "llm_request", {"history_length": len(conversation_history[chat_id])})

    # 2. Call the LLM
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", # Or whichever model you prefer
            messages=conversation_history[chat_id],
            temperature=0.0 # Keep it deterministic
        )
        
        llm_response_text = response.choices[0].message.content.strip()
        
        # Save assistant reply to history
        conversation_history[chat_id].append({"role": "assistant", "content": llm_response_text})
        
        # 3. Parse JSON safely (Handle cases where LLM includes markdown)
        if llm_response_text.startswith("```json"):
            llm_response_text = llm_response_text[7:-3].strip()
        elif llm_response_text.startswith("```"):
            llm_response_text = llm_response_text[3:-3].strip()
            
        answer_json = json.loads(llm_response_text)
        write_log(run_id, "llm_success", {"extracted_json": answer_json})
        return answer_json

    except json.JSONDecodeError:
        write_log(run_id, "error", {"message": "LLM did not return valid JSON", "raw_output": llm_response_text})
        return {"error": "Failed to parse data as JSON."}
    except Exception as e:
        write_log(run_id, "error", {"message": str(e)})
        return {"error": "Internal processing error."}

# --- TELEGRAM WEBHOOK ENDPOINT ---

@app.post(f"/webhook/{TELEGRAM_TOKEN}")
async def telegram_webhook(request: Request):
    update = await request.json()
    
    if "message" in update and "text" in update["message"]:
        chat_id = update["message"]["chat"]["id"]
        user_text = update["message"]["text"]
        
        # 1. Create a unique Run ID for logging
        run_id = str(uuid.uuid4())
        write_log(run_id, "received_message", {"text": user_text})
        
        # 2. Get the answer from the LLM agent
        answer_data = await analyze_data_with_llm(chat_id, user_text, run_id)
        
        # 3. Format the final required JSON output
        log_url = f"{PUBLIC_URL}/logs/{run_id}.jsonl"
        final_output = {
            "answer": answer_data,
            "log_url": log_url
        }
        
        write_log(run_id, "final_response_sent", final_output)
        
        # 4. Send back to Telegram
send_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"        payload = {
            "chat_id": chat_id,
            "text": json.dumps(final_output) # Send the required JSON as text
        }
        
        async with httpx.AsyncClient() as http_client:
            await http_client.post(send_url, json=payload)
            
    return JSONResponse(content={"status": "ok"})

# --- STARTUP SCRIPT ---
# Used to automatically register the webhook with Telegram when the server starts
@app.on_event("startup")
async def startup_event():
    webhook_url = f"{PUBLIC_URL}/webhook/{TELEGRAM_TOKEN}"
    # Make sure the next line starts exactly with https://
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={webhook_url}"
    
    async with httpx.AsyncClient() as http_client:
        await http_client.get(url)
        print(f"Webhook set to: {webhook_url}")



if __name__ == "__main__":
    import uvicorn
    # Use the PORT environment variable if available (for Render), otherwise default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)