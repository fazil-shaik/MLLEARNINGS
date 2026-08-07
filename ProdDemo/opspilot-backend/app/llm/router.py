"""
Every agent gets its LLM from here instead of hardcoding a model.
This is what makes the cost-tiering work: swap MODEL_FAST / MODEL_BALANCED /
MODEL_POWERFUL in .env and every agent picks it up, no code changes.

Tiers:
  fast      -> classification, triage, cheap structured extraction
  balanced  -> default drafting/summarizing work
  powerful  -> high-stakes reasoning (negotiation copy, escalation judgment)
"""
from functools import lru_cache
from langchain_openai import ChatOpenAI

from app.config import settings

_TIER_TO_MODEL = {
    "fast": settings.model_fast,
    "balanced": settings.model_balanced,
    "powerful": settings.model_powerful,
}


@lru_cache(maxsize=8)
def get_llm(tier: str = "balanced", temperature: float = 0.3) -> ChatOpenAI:
    model = _TIER_TO_MODEL.get(tier, settings.model_balanced)
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        default_headers={
            # OpenRouter uses these for its own analytics/rankings — optional but good practice
            "HTTP-Referer": "https://opspilot.local",
            "X-Title": "OpsPilot",
        },
    )
