import os
from ai_intake_bot.core.engine import IntakeBot
from ai_intake_bot.core.llm_registry import create_llm

bot = IntakeBot(
    mode="persona",
    template="expert_eval",
    persona="expert_reviewer",
    problem={
        "description": "Not able to find the purpose of my life.",
        "emotional_state": "frustrated",
        "goals": ["restore access"],
    },
    api_key="unused",
    qdrant_url=None,
    qdrant_api_key=None,
    files=None,
    selection_probability=0.6,
    enable_alerts=True,
    extra_system_prompt=None,
)

# ✅ THIS is the correct Gemini OpenAI-compat model
bot.set_llm(
    create_llm(
        "gemini",
        model="gemini-3-flash-preview",
        api_key="AIzaSyCBTm2ChQn_iw4Uh9OSt7WgzeNim6dI4fg",
    )
)

session, intro = bot.start_expert_eval()
print("Bot:", intro)
print("Type END++ to finish and evaluate.\n")

while True:
    msg = input("You: ").strip()
    if msg == "END++":
        print("\n[Conversation ended. Evaluating…]\n")
        break
    print("Bot:", bot.chat(session, msg))

print("Evaluation:")
print(bot.evaluate(session))
