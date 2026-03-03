"""Base class for LLM providers."""


class BaseLLMProvider:
    async def generate(self, system_prompt: str, messages: list[dict]) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: The full system prompt (already built by build_system_prompt).
            messages: List of {"role": str, "content": str} dicts (user/assistant only).

        Returns:
            The assistant's response text.
        """
        raise NotImplementedError
