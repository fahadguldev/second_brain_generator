from collections.abc import Iterator
from typing import Optional
from src.config import settings
from src.gemini_keys import gemini_key_manager

_generation_model = settings.GENERATION_MODEL


class GenerationError(RuntimeError):
    """Raised when the generation provider cannot complete a response."""


class GeminiLLM:
    def __init__(self, model: str = _generation_model):
        self.model = model

    def generate(self, prompt: str, system_instructions: Optional[str] = None) -> str:
        """Generate text using the Gemini model."""
        try:
            full_prompt = prompt
            if system_instructions:
                full_prompt = f"{system_instructions}\n\nUser: {prompt}"

            response = gemini_key_manager.run(
                lambda client: client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                )
            )
            return response.text
        except Exception as e:
            raise GenerationError("Could not generate response") from e

    def generate_stream(
        self, prompt: str, system_instructions: Optional[str] = None,
    ) -> Iterator[str]:
        full_prompt = prompt
        if system_instructions:
            full_prompt = f"{system_instructions}\n\nUser: {prompt}"
        try:
            for chunk in gemini_key_manager.stream(
                lambda client: client.models.generate_content_stream(
                    model=self.model, contents=full_prompt,
                )
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise GenerationError("Could not generate response") from e


# Global instance
llm = GeminiLLM()
