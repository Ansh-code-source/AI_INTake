"""Persona and RAG engines + IntakeBot entry point.

Phase 7: PersonaEngine wiring with prompt composition. Uses a FakeLLM by default
for deterministic tests. Real LLM integration (LangChain) will be added later.
"""
from typing import Optional, List, Dict, Any
import json

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
    """Persona-mode engine: assemble prompts, call LLM, parse output."""

    def __init__(self, config: IntakeConfig, llm: Optional[BaseLLM] = None):
        self.config = config
        self.llm = llm or FakeLLM()
        # Validate persona exists
        if self.config.persona not in PERSONAS:
            raise ValueError(f"Unknown persona: {self.config.persona}")

    def run(self, user_input: str) -> Dict[str, Any]:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        base = compose_base_system_prompt(self.config.extra_system_prompt)
        persona_prompt = compose_persona_role_prompt(
            self.config.persona, PERSONAS[self.config.persona]
        )
        problem_prompt = ""
        if self.config.template == "expert_eval" and self.config.problem:
            problem_prompt = compose_problem_scenario_prompt(self.config.problem)
        template_prompt = compose_template_prompt(self.config.template)

        # If template requests scoring, ask the model to provide signal scores (0-1)
        scoring_prompt = ""
        if self.config.template:
            from .templates import get_template

            tpl = get_template(self.config.template)
            if tpl.get("scoring"):
                from .prompts import compose_scoring_prompt
                from .scoring import EXPECTED_SIGNALS

                scoring_prompt = compose_scoring_prompt(list(EXPECTED_SIGNALS))

        # For expert_eval, perform a role-play call and then a separate evaluation call.
        if self.config.template == "expert_eval":
            # Role-play call (persona role only, no evaluation instructions)
            roleplay_prompt = (
                f"{base}\n\n{persona_prompt}\n\n{problem_prompt}\n\n{template_prompt}\n\n"
                f"User: {user_input}\n\nRespond with valid JSON matching the output contract:"
            )
            raw_role = self.llm.generate(roleplay_prompt)
            try:
                parsed_role = json.loads(raw_role)
            except Exception:
                parsed_role = {"reply": raw_role}

            # Now evaluation call: pass candidate's reply and ask the evaluator to score and recommend actions.
            from .prompts import compose_evaluation_prompt

            candidate_text = parsed_role.get("reply", "")
            eval_prompt = compose_evaluation_prompt(
                "Evaluate the candidate response",
                instructions=(f"Candidate reply: {candidate_text}\n\n" + (scoring_prompt or "")),
            )
            raw_eval = self.llm.generate(eval_prompt)
            try:
                parsed_eval = json.loads(raw_eval)
            except Exception:
                parsed_eval = {}

            # Merge roleplay (content) with evaluation (signals/actions/score)
            merged = {**parsed_role, **parsed_eval}
            parsed = merged
        else:
            # Non-expert templates: single call. Include scoring prompt inline if present.
            final_prompt = (
                f"{base}\n\n{persona_prompt}\n\n{problem_prompt}\n\n{template_prompt}\n\n"
                f"{scoring_prompt}\n\nUser: {user_input}\n\nRespond with valid JSON matching the output contract:"
            )
            raw = self.llm.generate(final_prompt)
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {"reply": raw}

        # Compute signals, score, selection, and recommended actions
        from .scoring import compute_score, apply_selection
        from .actions import detect_actions

        signals = parsed.get("signals", {}) or {}
        parsed_signals = {}
        for k, v in signals.items():
            try:
                parsed_signals[k] = float(v)
            except Exception:
                # ignore non-numeric signals
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

        # Ensure contract fields exist (fill missing ones with defaults)
        contract = OutputContract(**{**OutputContract().dict(), **parsed})
        return contract.dict()


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