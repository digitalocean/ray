import abc
from typing import Optional

from ray.llm._internal.common.base_pydantic import BaseModelExtended
from ray.llm._internal.serve.configs.openai_api_models import ChatCompletionRequest, DeltaMessage, ToolCall

class LLMToolCallInformation(BaseModelExtended):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: Optional[str] = None

class LLMToolParser(abc.ABC):
    """Base class for all LLM tool parsers"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> LLMToolCallInformation:
        """Extract tool calls from the response."""
        pass
    
    @abc.abstractmethod
    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: list[int],
        current_token_ids: list[int],
        delta_token_ids: list[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from a stream chunk"""
        pass
    
class LLMToolParserManager(abc.ABC):
    """Manager class for LLMToolParser instances."""

    def __init__(self):
        pass
    
    @abc.abstractmethod
    def get_tool_parser(self, name: str) -> LLMToolParser:
        """Get a tool parser by name."""
        pass



