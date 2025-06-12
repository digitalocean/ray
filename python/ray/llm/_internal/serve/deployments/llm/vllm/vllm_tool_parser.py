
from ray.llm._internal.serve.deployments.llm.llm_tool_parser import LLMToolCallInformation, LLMToolParser, LLMToolParserManager
from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    ChatCompletionRequest,
    DeltaToolCall,
    ToolCall,
    FunctionCall,
)

from vllm.entrypoints.openai.tool_parsers import (
  ToolParserManager as _VLLMToolParserManager,
  ToolParser as _VLLMToolParser,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer

class VLLMToolParser(LLMToolParser):
    def __init__(
        self,
        tool_parser: _VLLMToolParser,
    ):
        super().__init__()
        self._tool_parser = tool_parser
   
    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> LLMToolCallInformation:
        """Extract tool calls from the response."""
        #TODO: Convert our ChatCompletionRequest to VLLMChatCompletionRequest if needed
        tool_call_info = self._tool_parser.extract_tool_calls(model_output, request)
        return LLMToolCallInformation(
            tools_called=tool_call_info.tools_called,
            content=tool_call_info.content,
            tool_calls=[
                ToolCall(
                    id=tool.id,
                    type=tool.type,
                    function=FunctionCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                )
                for tool in tool_call_info.tool_calls
            ]
        )

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
        """Extract tool calls from a stream chunk."""
        #TODO: Convert our ChatCompletionRequest to VLLMChatCompletionRequest if needed
        delta_message = self._tool_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request
        )
        if delta_message is None:
            return None
        return DeltaMessage(
            role=delta_message.role,
            content=delta_message.content,
            reasoning_content=delta_message.reasoning_content,
            tool_calls=[
                DeltaToolCall(
                    id=tool.id,
                    type=tool.type,
                    index=tool.index,
                    function=DeltaFunctionCall(
                        name=tool.function.name,
                        arguments=tool.function.arguments,
                    )
                )
                for tool in delta_message.tool_calls
            ]
        )
    
class VLLMToolParserManager(LLMToolParserManager):
    def __init__(self, tokenizer: AnyTokenizer | None):
        super().__init__()
        self.tokenizer = tokenizer

    def get_tool_parser(self, name: str) -> LLMToolParser:
        try:
          raw_parser_cls = _VLLMToolParserManager.get_tool_parser(name)
        except KeyError as e:
          raise ValueError(f"Failed to get tool parser for {name}: {e}")
        tool_parser = raw_parser_cls(tokenizer=self.tokenizer)
        return VLLMToolParser(tool_parser)    
    