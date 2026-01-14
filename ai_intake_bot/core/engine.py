"""Persona and RAG engines + IntakeBot entry point.

Phase 7: PersonaEngine wiring with prompt composition. Uses a FakeLLM by default
for deterministic tests. Real LLM integration (LangChain) will be added later.
"""
from typing import Optional, List, Dict, Any
import json
from .session import ExpertEvalSession
from .llm_registry import create_llm

from pydantic import BaseModel, ValidationError, Field

from .prompts import (
    compose_base_system_prompt,
    compose_persona_role_prompt,
    compose_problem_scenario_prompt,
)
from .templates import compose_template_prompt
from .personas import PERSONAS
from .llm import FakeLLM, BaseLLM
from .router import Router


class IntakeConfig(BaseModel):
    mode: str
    template: str
    persona: str
    problem: Optional[dict] = None
    api_key: str
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    files: Optional[List[str]] = None
    selection_probability: Optional[float] = None
    enable_alerts: bool = False
    extra_system_prompt: Optional[str] = None

    class Config:
        extra = "forbid"
    




class OutputContract(BaseModel):
    reply: str = ""
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, Any] = Field(default_factory=dict)
    candidate_score: Optional[float] = None
    recommended_selection: Optional[bool] = None
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list)


class PersonaEngine:
    """
    Persona-mode engine.

    - Non-expert templates: single-turn via run()
    - expert_eval: multi-turn via session APIs
    """

    def __init__(self, config: IntakeConfig, llm: Optional[BaseLLM] = None):
        self.config = config
        self.llm = llm or FakeLLM()

        if self.config.persona not in PERSONAS:
            raise ValueError(f"Unknown persona: {self.config.persona}")

    # ------------------------------------------------------------------
    # expert_eval (MULTI-TURN)
    # ------------------------------------------------------------------

    def start_expert_eval(self) -> Dict[str, Any]:
        if self.config.template != "expert_eval":
            raise ValueError("start_expert_eval only valid for expert_eval")

        base = compose_base_system_prompt(self.config.extra_system_prompt)
        persona_prompt = compose_persona_role_prompt(
            self.config.persona,
            PERSONAS[self.config.persona],
        )
        problem_prompt = compose_problem_scenario_prompt(self.config.problem)

        prompt = f"""
{base}

{persona_prompt}

{problem_prompt}

You are the USER with this problem.
Start the conversation naturally.
Do NOT evaluate the other person.

Respond with JSON:
{{ "reply": "<text>" }}
"""

        raw = self.llm.generate(prompt)
        return self._safe_json_reply(raw)

    def chat_expert_eval(self, session, user_input: str) -> Dict[str, Any]:
        history = "\n".join(
            f"{m['role'].capitalize()}: {m['text']}"
            for m in session.history
        )

        persona_prompt = compose_persona_role_prompt(
            self.config.persona,
            PERSONAS[self.config.persona],
        )
        problem_prompt = compose_problem_scenario_prompt(self.config.problem)

        prompt = f"""
{persona_prompt}

{problem_prompt}

Conversation so far:
{history}

Expert: {user_input}

Respond as the USER.
React emotionally and naturally.
Do NOT evaluate.

Respond with JSON:
{{ "reply": "<text>" }}
"""

        raw = self.llm.generate(prompt)
        return self._safe_json_reply(raw)

    def evaluate_expert_eval(self, session) -> Dict[str, Any]:
        from .prompts import compose_evaluation_prompt
        from .scoring import EXPECTED_SIGNALS, compute_score, apply_selection
        from .actions import detect_actions

        conversation = "\n".join(
            f"{m['role'].capitalize()}: {m['text']}"
            for m in session.history
        )

        eval_prompt = compose_evaluation_prompt(
            evaluation_task="Evaluate the USER in the conversation below.",
            instructions=f"""
    You are an evaluator.

    ROLE DEFINITIONS (IMPORTANT):
    - USER = the human participant being evaluated.
    - BOT = the simulated persona/problem-holder.
    - ONLY evaluate messages spoken by USER.
    - Do NOT evaluate BOT responses.

    Conversation transcript:
    ------------------------
    {conversation}
    ------------------------

    Evaluate the USER on the following signals.
    Each signal must be a number between 0 and 1.

    Signals:
    - empathy (understanding and emotional awareness)
    - clarity (clear communication and intent)
    - aggression (hostility or impatience; higher is worse)
    - urgency (sense of priority or seriousness)

    Return STRICT JSON in the following format ONLY:

    {{
    "signals": {{
        "empathy": 0.0,
        "clarity": 0.0,
        "aggression": 0.0,
        "urgency": 0.0
    }},
    "structured_data": {{
        "summary": "One sentence summary of USER behavior"
    }}
    }}

    RULES:
    - Do NOT include prose outside JSON.
    - Do NOT explain your reasoning.
    - Do NOT evaluate the BOT.
    """
        )

        raw = self.llm.generate(eval_prompt)
        parsed = self._safe_json(raw)

        signals = parsed.get("signals", {}) or {}
        numeric_signals = {
            k: float(v)
            for k, v in signals.items()
            if isinstance(v, (int, float, str))
        }

        candidate_score = compute_score(numeric_signals)

        recommended_selection = None
        if self.config.selection_probability is not None:
            recommended_selection = apply_selection(
                candidate_score,
                self.config.selection_probability,
            )

        parsed["candidate_score"] = candidate_score
        parsed["recommended_selection"] = recommended_selection
        parsed["recommended_actions"] = detect_actions(parsed)

        contract = OutputContract(**{**OutputContract().dict(), **parsed})
        return contract.dict()

    # ------------------------------------------------------------------
    # NON-expert templates (SINGLE-TURN)
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> Dict[str, Any]:
        if self.config.template == "expert_eval":
            raise RuntimeError(
                "expert_eval must be used via start_expert_eval / chat / evaluate"
            )

        base = compose_base_system_prompt(self.config.extra_system_prompt)
        persona_prompt = compose_persona_role_prompt(
            self.config.persona,
            PERSONAS[self.config.persona],
        )
        template_prompt = compose_template_prompt(self.config.template)

        final_prompt = f"""
{base}

{persona_prompt}

{template_prompt}

User: {user_input}

Respond with JSON matching the output contract.
"""

        raw = self.llm.generate(final_prompt)
        parsed = self._safe_json(raw)

        from .scoring import compute_score, apply_selection
        from .actions import detect_actions

        signals = parsed.get("signals", {}) or {}
        numeric_signals = {
            k: float(v)
            for k, v in signals.items()
            if isinstance(v, (int, float, str))
        }

        candidate_score = parsed.get("candidate_score")
        if candidate_score is None:
            candidate_score = compute_score(numeric_signals)

        recommended_selection = None
        if self.config.selection_probability is not None:
            recommended_selection = apply_selection(
                candidate_score,
                self.config.selection_probability,
            )

        parsed["candidate_score"] = candidate_score
        parsed["recommended_selection"] = recommended_selection
        parsed["recommended_actions"] = detect_actions(parsed)

        contract = OutputContract(**{**OutputContract().dict(), **parsed})
        return contract.dict()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _safe_json_reply(self, raw: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw)
            return {"reply": data.get("reply", "")}
        except Exception:
            return {"reply": raw}

    def _safe_json(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            return {}


class RAGEngine:
    """RAG-mode engine: load documents, retrieve relevant chunks, and ground responses."""

    def __init__(self, config: IntakeConfig, llm: Optional[BaseLLM] = None):
        self.config = config
        self.llm = llm or FakeLLM()

        # Enforce rag-specific requirements
        if not self.config.files or not isinstance(self.config.files, list):
            raise ValueError("mode 'rag' requires a non-empty 'files' list")
        if not self.config.qdrant_url:
            # We allow running in local mode without Qdrant; a real Qdrant integration
            # would require qdrant_url and qdrant_api_key.
            self.local_only = True
        else:
            self.local_only = False

    def run(self, user_input: str) -> Dict[str, Any]:
        # Load documents
        from ..rag.loader import load_documents
        from ..core.prompts import compose_rag_grounding_prompt
        docs = load_documents(self.config.files)
        if not docs:
            return OutputContract().dict()

        # If qdrant_url is configured, use QdrantVectorStore for retrieval; otherwise use local retriever
        retrieved_chunks = []
        if self.local_only:
            from ..rag.retriever import build_retriever_from_texts

            retriever = build_retriever_from_texts(docs)
            retrieved_chunks = retriever.retrieve(user_input, top_k=5)
        else:
            # Use QdrantVectorStore wrapper; create ephemeral collection unless configured otherwise
            from ..rag.vectorstore import QdrantVectorStore
            qstore = QdrantVectorStore(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key)
            try:
                # Split documents into texts using splitter
                from ..rag.splitter import split_documents_texts

                texts = split_documents_texts(docs)
                qstore.from_documents(texts)
                results = qstore.similarity_search(user_input, k=5)
                # Normalize results into strings
                chunks = []
                for r in results:
                    # Try common attributes used by LangChain Documents
                    if hasattr(r, "page_content"):
                        chunks.append(r.page_content)
                    elif isinstance(r, (tuple, list)) and len(r) >= 1:
                        chunks.append(r[0])
                    else:
                        chunks.append(str(r))
                retrieved_chunks = chunks
            finally:
                try:
                    qstore.cleanup()
                except Exception:
                    pass

        grounding = compose_rag_grounding_prompt(retrieved_chunks)

        base = compose_base_system_prompt(self.config.extra_system_prompt)
        template_prompt = compose_template_prompt(self.config.template)

        final_prompt = (
            f"{base}\n\n{grounding}\n\n{template_prompt}\n\nUser: {user_input}\n\n"
            "Respond with valid JSON matching the output contract."
        )

        raw = self.llm.generate(final_prompt)
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"reply": raw}

        # Compute signals, score and actions similar to persona engine
        from .scoring import compute_score, apply_selection
        from .actions import detect_actions

        signals = parsed.get("signals", {}) or {}
        parsed_signals = {}
        for k, v in signals.items():
            try:
                parsed_signals[k] = float(v)
            except Exception:
                continue

        candidate_score = parsed.get("candidate_score")
        if candidate_score is None:
            candidate_score = compute_score(parsed_signals)

        recommended_selection = None
        if self.config.selection_probability is not None:
            recommended_selection = apply_selection(candidate_score, self.config.selection_probability)

        parsed.setdefault("candidate_score", candidate_score)
        parsed.setdefault("recommended_selection", recommended_selection)

        recommended_actions = detect_actions(parsed)
        parsed.setdefault("recommended_actions", recommended_actions)

        contract = OutputContract(**{**OutputContract().dict(), **parsed})
        return contract.dict()


class IntakeBot:
    """Main SDK entry point. Routes to engines and returns validated output.

    __init__ is strict and fail-fast — no defaults that hide misconfiguration.
    """

    def __init__(
        self,
        mode: str,
        template: str,
        persona: str,
        problem: Optional[dict],
        api_key: str,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        files: Optional[List[str]],
        selection_probability: Optional[float],
        enable_alerts: bool,
        extra_system_prompt: Optional[str],
    ):
        try:
            self.config = IntakeConfig(
                mode=mode,
                template=template,
                persona=persona,
                problem=problem,
                api_key=api_key,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                files=files,
                selection_probability=selection_probability,
                enable_alerts=enable_alerts,
                extra_system_prompt=extra_system_prompt,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid IntakeBot configuration: {e}") from e

        self._router: Optional[Router] = None
        self._llm: Optional[BaseLLM] = None
        self._voice = None
    def start_expert_eval(self):
        if self.config.template != "expert_eval":
            raise RuntimeError("start_expert_eval only valid for expert_eval")

        if self._router is None:
            self._router = Router()

        engine = self._router.route(self.config)
        if self._llm:
            engine.llm = self._llm

        session = ExpertEvalSession(self.config.persona, self.config.problem)
        first = engine.start_expert_eval()
        session.add_bot(first["reply"])
        return session, first["reply"]
    
    def set_llm_by_name(self, provider: str, **kwargs):
        """
        Example:
        bot.set_llm_by_name("openai", model="gpt-4o")
        bot.set_llm_by_name("gemini", model="gemini-1.5-flash")
        bot.set_llm_by_name("fake")
        """
        self._llm = create_llm(provider, **kwargs)
    
    def chat(self, session, user_input: str):
        if self._router is None:
            self._router = Router()

        engine = self._router.route(self.config)
        if self._llm:
            engine.llm = self._llm

        session.add_user(user_input)
        reply = engine.chat_expert_eval(session, user_input)
        session.add_bot(reply["reply"])
        return reply["reply"]
    def evaluate(self, session):
        if self._router is None:
            self._router = Router()

        engine = self._router.route(self.config)
        if self._llm:
            engine.llm = self._llm

        return engine.evaluate_expert_eval(session)

    def set_voice(self, voice):
        """Opt-in: set a voice adapter (duck-typed) for spoken replies.

        Accepts any object with a callable `speak(text: str, filename: Optional[str]=None)` method.
        """
        if not hasattr(voice, "speak") or not callable(getattr(voice, "speak")):
            raise ValueError("voice must implement a callable `speak(text, filename=None)`")
        self._voice = voice

    def set_llm(self, llm: BaseLLM):
        """Inject an LLM instance (useful for testing or real LLMs)."""
        self._llm = llm

    def handle(self, user_input: str) -> Dict[str, Any]:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        if self._router is None:
            self._router = Router()

        engine = self._router.route(self.config)
        # If a custom llm was injected into the bot, pass it to engine
        if self._llm is not None:
            engine.llm = self._llm

        result = engine.run(user_input)

        # If a voice adapter is set, attempt to speak the reply (best-effort, swallow errors)
        if self._voice is not None:
            try:
                reply_text = result.get("reply", "")
                # Do not block or fail if speak raises
                self._voice.speak(reply_text)
            except Exception:
                pass

        return result