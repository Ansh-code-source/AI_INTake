"""Centralized prompts and prompt composition helpers for ai_intake_bot.

All prompt text is centralized here. Prompts are composable via small helper
functions. Prompts *must* honor the separation rules:
- Persona role-play prompts must NOT include evaluation instructions.
- Evaluation prompts must NOT include role-play instructions.
- Developer / extra system prompt is append-only and merged last.

These helpers return strings; engines will assemble the final prompt(s).
"""
from typing import List, Optional


BASE_SYSTEM_PROMPT = (
    "You are an assistant that converts conversational input into a structured JSON output "
    "according to the requested template. Follow the output contract strictly."
)


def compose_base_system_prompt(extra_system_prompt: Optional[str] = None) -> str:
    """Compose the base system prompt, appending any extra system prompt.

    The extra_system_prompt is append-only and should never override base guidance.
    """
    prompt = BASE_SYSTEM_PROMPT
    if extra_system_prompt:
        prompt = f"{prompt}\n\n" + extra_system_prompt
    return prompt


def compose_persona_role_prompt(persona_name: str, persona_spec: dict) -> str:
    """Return a role-play prompt for a persona.

    IMPORTANT: This prompt must instruct role-play and must NOT include any evaluation
    or scoring instructions. Those belong in a separate evaluation prompt.
    """
    tone = persona_spec.get("tone", "neutral")
    baseline = persona_spec.get("emotional_baseline", "stable")
    behavior = persona_spec.get("behavior", "respond helpfully")

    return (
        f"You are now role-playing as '{persona_name}'. Adopt a {tone} tone, with an "
        f"emotional baseline of {baseline}. Behavior guidance: {behavior}. "
        "Respond in a concise, user-facing manner and ask clarifying questions as needed."
    )


def compose_problem_scenario_prompt(problem: dict) -> str:
    """Return a problem scenario prompt (used only inside persona role-play when
    `expert_eval` template is selected).

    This is only injected for persona role-play expert evaluations.
    """
    if not problem:
        return ""
    desc = problem.get("description", "")
    state = problem.get("emotional_state", "neutral")
    goals = problem.get("goals", [])
    constraints = problem.get("constraints", [])

    parts = [f"Scenario: {desc}", f"Emotional state: {state}"]
    if goals:
        parts.append(f"Goals: {', '.join(goals)}")
    if constraints:
        parts.append(f"Constraints: {', '.join(constraints)}")

    return "\n".join(parts)


def compose_rag_grounding_prompt(retrieved_chunks: List[str]) -> str:
    """Produce a grounding prompt that instructs the model how to use retrieved docs."""
    if not retrieved_chunks:
        return ""
    return (
        "Use the following retrieved documents as grounding. Refer to them only when they "
        "are helpful and cite the source. Do not hallucinate facts not in the documents.\n\n"
        + "\n\n".join(retrieved_chunks)
    )


def compose_evaluation_prompt(evaluation_task: str, instructions: Optional[str] = None) -> str:
    """Return an evaluation prompt. MUST NOT include role-play instructions."""
    prompt = f"You are an expert evaluator. Task: {evaluation_task}. Evaluate the candidate's " "response against the criteria and return structured JSON as specified."
    if instructions:
        prompt = f"{prompt}\n\n{instructions}"
    return prompt


def compose_scoring_prompt(metrics: List[str]) -> str:
    """Return a short prompt instructing the model to score signals (0-1)."""
    metrics_list = ", ".join(metrics)
    return (
        f"For each of the following signals: {metrics_list}, provide a numeric score between" " 0 and 1 (inclusive) and a 1-2 sentence rationale for each score."
    )


def compose_agentic_recommendation_prompt() -> str:
    """Return a short prompt that asks for recommended actions (ALERT, ESCALATE, API_CALL, FOLLOW_UP)."""
    return (
        "Considering the structured output and signals, recommend a short list of actions. "
        "Each action must include type (ALERT/ESCALATE/API_CALL/FOLLOW_UP), priority (LOW/MEDIUM/HIGH), "
        "and a concise reason. Do NOT execute any actions. Output must be JSON." 
    )
