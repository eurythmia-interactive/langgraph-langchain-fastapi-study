### 1. Do you need to install an SQLite module?

**Yes, but only the LangGraph integration package.** 
The `sqlite3` database driver is built directly into Python's standard library, so you don't need to install a database driver. However, LangGraph's SQLite checkpointer is distributed as a separate package. 

You need to add **`langgraph-checkpoint-sqlite`** to your `requirements.txt`:

```text
# Add this to your requirements.txt
langgraph-checkpoint-sqlite
```
*(Run `uv pip install langgraph-checkpoint-sqlite` or `uv sync` to update your environment).*

### 2. Architectural Adjustment: In-Memory vs. File-Based SQLite
The notebook uses `sqlite3.connect(":memory:")` for quick, ephemeral testing inside Jupyter. However, if we use `:memory:` in a FastAPI app, **the database will be wiped every time the server restarts**, defeating the purpose of "external persistence."

To make this a true production-ready API, I have changed it to use a **file-based SQLite database** (`chatbot_memory.db`). This ensures your conversation summaries persist across server restarts, exactly as the lesson intends.

### 3. The FastAPI Implementation

Here is the complete, production-ready API adapting the **Chatbot with External Memory** lesson. I've also added a `GET /state/{thread_id}` endpoint so you can easily inspect the SQLite database to prove the memory is persisting.



### How to test the Summarization & Persistence:

1. Run the app: `python app_chatbot_external_memory.py`
2. Send a few messages to build up the history (the trigger is > 6 messages):

```json
POST /process
{
    "messages": [{"role": "human", "content": "Hi, my name is Lance and I like the 49ers."}],
    "thread_id": "test-external-1"
}
```
*(Repeat this a few times with different messages, or send an array of 6+ messages at once to trigger the `summarize_conversation` node).*

3. **Verify the External Memory:**
Once the summarization node triggers, check the raw state saved in your SQLite database using the new utility endpoint:

```http
GET /state/test-external-1
```

You will see that the `messages` array has been trimmed down to just the last 2 messages, but the `summary` field contains the compressed history. Because we used a file-based SQLite DB, **if you restart your FastAPI server right now and hit that same `GET` endpoint, the summary will still be there.**