from langchain_core.language_models import LLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
import requests

class OpenRouterChat(LLM):
    """Custom LangChain wrapper for OpenRouter API."""

    # Pydantic fields
    api_key: str
    model_name: str

    @property
    def _llm_type(self):
        return "openrouter_chat"

    def _call(self, prompt: str, **kwargs) -> str:
        """Simplified interface for a single prompt."""
        return self._generate([HumanMessage(content=prompt)]).generations[0][0].text

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> LLMResult:
        """Generate response using OpenRouter Chat API with robust message handling."""

        mapped_messages = []

        for m in messages:
            # ✅ Handle different types of messages
            if isinstance(m, str):
                role = "user"
                content = m

            elif hasattr(m, "type") and hasattr(m, "content"):
                # BaseMessage object
                if m.type == "human":
                    role = "user"
                elif m.type == "ai":
                    role = "assistant"
                elif m.type == "system":
                    role = "system"
                else:
                    role = "user"
                content = m.content

            elif isinstance(m, dict):
                # Dict message
                role = m.get("role", "user")
                content = m.get("content", "")

            else:
                # Fallback
                role = "user"
                content = str(m)

            mapped_messages.append({"role": role, "content": content})

        # Prepare API payload
        payload = {"model": self.model_name, "messages": mapped_messages}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API error: {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        return LLMResult(
            generations=[[{"text": content, "message": AIMessage(content=content)}]]
        )
