"""Base class for LLM providers."""


class BaseLLMProvider:
    PROVIDER_ID: str = ""
    DEFAULT_MODEL: str = ""
    REQUIRES_API_KEY: bool = True

    @classmethod
    def from_config(cls, model: str, settings: dict, **kwargs) -> "BaseLLMProvider":
        """Factory method. Subclasses override to pick their own settings keys."""
        raise NotImplementedError(f"{cls.__name__}.from_config() is not implemented")

    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: The full system prompt (already built by build_system_prompt).
            messages: List of {"role": str, "content": str} dicts (user/assistant only).

        Returns:
            The assistant's response text.
        """
        raise NotImplementedError
