import os
from abc import abstractmethod, ABC
from typing import Literal

from google import genai
from google.genai import types
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel, Field


class Query(BaseModel):
    # Use default_factory to avoid sharing the same list across instances.
    turns: list[dict[Literal["user", "assistant"], str]] = Field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def query(self, model_id: str, query: Query) -> GenerateContentResponse:
        pass


class GoogleProvider:
    MODEL_MAPPING = {
        "A": "gemma-4-31b-it",
    }

    # System prompts for each model variant
    SYSTEM_PROMPTS = {
        "A": "You are an expert evaluator in English-Chinese translation. "
             "You will be given a pair of parallel sentences as a JSON dict in the following format: "
             "{ 'zh': 'Chinese sentence', 'en': 'English sentence' }."
             "Your task is to generate three 'hard negatives' - slightly incorrect variations of the given correct English translation. "
             "They must sound grammatically correct and look highly plausible, but introduce a critical semantic error. "
             "(e.g., literal translation mistake, word-sense ambiguity, or flipping the polarity). "
             "Do not make them obvious gibberish. "
             "Your output should be a valid JSON list containing the three sentences, like so: "
             "[Variation 1, Variation 2, Variation 3].",
    }

    PRICING = {
        "gemma-4-31b-it": {"input": 0.0, "output": 0.0},  # free on AI Studio
    }

    def __init__(self):
        super().__init__()

        api_key = os.getenv("AI_STUDIO_API_KEY")
        if api_key is None:
            raise ValueError(
                "GEMINI_API_KEY is not set in the environment variables. Make sure to copy the .env.template file to .env and fill in your Gemini API key from Google AI Studio.")

        self.client = genai.Client(api_key=api_key)

    async def query(self, model_id: str, query: Query) -> GenerateContentResponse:
        if model_id not in self.__class__.get_supported_models():
            raise ValueError(f"Unsupported model: {model_id}")

        # Get actual gemini model and system prompt
        actual_model = self.MODEL_MAPPING[model_id]
        system_prompt = self.SYSTEM_PROMPTS[model_id]

        # Convert query format to Gemini format with system prompt
        generation_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(
                # "high" to enable reasoning, "minimal" to disable it
                thinking_level="high"
            ),
        )

        contents = []
        for turn in query.turns:
            for role, content in turn.items():
                if role not in ["user", "model"]:
                    raise ValueError(f"Invalid role in query: {role}. Role must be one of 'user' or 'model'.")
                contents.append(types.Content(role=role, parts=[types.Part(text=content)]))
        try:
            async with self.client.aio as aclient:
                response = await aclient.models.generate_content(
                    model=actual_model,
                    config=generation_config,
                    contents=contents,
                )
        except Exception as e:
            raise Exception(f"API error: {str(e)}")

        return response

    @classmethod
    def get_supported_models(cls) -> list[str]:
        return list(cls.MODEL_MAPPING.keys())

    # Provider registry - easy to extend with new providers
PROVIDERS = {
    "gemma": GoogleProvider,
}

def get_provider(model_id: str) -> tuple[str, LLMProvider]:
    """Get the appropriate provider for a model"""
    for provider_name, provider_cls in PROVIDERS.items():
        if model_id in provider_cls.get_supported_models():
            provider = provider_cls()
            return provider_name, provider

    raise ValueError(f"No provider found for model: {model_id}")