# ai_intake_bot

Minimal SDK for persona-based and RAG-based conversational intake.

What this repo is: an SDK, not an app. No data persistence, no auth system, no background jobs.

Quickstart (persona mode with FakeLLM)

- Create a venv and install deps:

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  pip install -e ".[dev]"

- Example (persona, expert_eval) using the FakeLLM for deterministic results:

```python
from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import FakeLLM

bot = IntakeBot(
    mode="persona",
    template="expert_eval",
    persona="expert_reviewer",
    problem={"description": "Cannot login", "emotional_state": "frustrated", "goals": ["restore access"]},
    api_key="sk-test",
    qdrant_url=None,
    qdrant_api_key=None,
    files=None,
    selection_probability=0.5,
    enable_alerts=True,
    extra_system_prompt=None,
)
bot.set_llm(FakeLLM())
out = bot.handle("Please evaluate this scenario")
print(out)
```

Using a real LLM (OpenAI)

- This SDK supports optional, explicit use of a real LLM. It will only call
  an LLM when you inject one via `IntakeBot.set_llm()` (so the default is safe
  for local development and tests).

- To use OpenAI's API, set `OPENAI_API_KEY` in your environment and then:

```python
from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm import OpenAIChatLLM

llm = OpenAIChatLLM(model="gpt-4o")
bot = IntakeBot(...)
bot.set_llm(llm)
response = bot.handle("Ask something")
```

Security model: the SDK never persists secrets to disk. `OpenAIChatLLM` expects
`OPENAI_API_KEY` to be set and will not log its value.

RAG with Qdrant (dev-friendly)

- The RAG engine uses a `LocalVectorStore` by default; if you provide `qdrant_url`
  when constructing `IntakeBot`, the RAG engine will upload chunks to the
  configured Qdrant instance (ephemeral collection by default) and ground
  responses on the retrieved documents.

- For a full integration with LangChain and Qdrant follow the example in
  `scripts/run_qdrant_integration.sh` (this script starts a Qdrant Docker
  container and runs the Qdrant integration test). The integration requires
  `langchain_qdrant`, `langchain_openai`, and an OpenAI key for embeddings.

Demos

- `scripts/demo_persona_openai.py` — Runs an `expert_eval` persona using OpenAI (requires `OPENAI_API_KEY`).

- `scripts/demo_rag_qdrant.py` — Ingests a PDF using LangChain's `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter`, embeds with `OpenAIEmbeddings`, writes to Qdrant (requires `QDRANT_URL` and `OPENAI_API_KEY`), and performs a retrieval + chat.

- `scripts/demo_tts.py` — Demonstrates the TTS adapters. Prefers macOS `say`, falls back to `pyttsx3`, and ultimately to a no-op adapter.

Run demos (examples):

- Persona demo (OpenAI):

  export OPENAI_API_KEY=<your_key>
  python scripts/demo_persona_openai.py

- RAG demo (Qdrant + LangChain):

  export OPENAI_API_KEY=<your_key>
  export QDRANT_URL=http://localhost:6333
  python scripts/demo_rag_qdrant.py

- TTS demo (local):

  python scripts/demo_tts.py

Security reminder: never commit real API keys. Use `.env` files for local testing and set CI secrets for integration runs.
# AI_INTake
