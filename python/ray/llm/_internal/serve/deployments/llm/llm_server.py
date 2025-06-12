import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, Dict, Any, Optional, Type, Union, List

# Third-party imports
from ray import serve
from ray._common.utils import import_attr
# vllm imports Needs abstractions/wrapper types
from ray.llm._internal.serve.deployments.llm.llm_tool_parser import LLMToolParser, LLMToolParserManager
from ray.llm._internal.serve.deployments.llm.vllm.vllm_tool_parser import VLLMToolParserManager
from vllm.entrypoints.openai.tool_parsers.mistral_tool_parser import MistralToolCall
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

# Local imports
from ray.llm._internal.serve.configs.constants import (
    DEFAULT_HEALTH_CHECK_PERIOD_S,
    DEFAULT_HEALTH_CHECK_TIMEOUT_S,
    ENGINE_START_TIMEOUT_S,
    RAYLLM_VLLM_ENGINE_CLS_ENV,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionNamedToolChoiceParam,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    LLMChatResponse,
    LLMCompletionsResponse,
    UsageInfo,
    ToolCall,
    FunctionCall,
)
from ray.llm._internal.serve.configs.openai_api_models_patch import (
    ErrorResponse,
)
from ray.llm._internal.serve.configs.prompt_formats import Message, Prompt
from ray.llm._internal.serve.configs.server_models import (
    DiskMultiplexConfig,
    LLMConfig,
    LLMRawResponse,
)
from ray.llm._internal.serve.deployments.llm.llm_engine import LLMEngine
from ray.llm._internal.serve.deployments.llm.image_retriever import ImageRetriever
from ray.llm._internal.serve.deployments.llm.multiplex.lora_model_loader import (
    LoraModelLoader,
)
from ray.llm._internal.serve.deployments.llm.vllm.vllm_engine import VLLMEngine
from ray.llm._internal.serve.deployments.utils.error_handling_utils import (
    StreamingErrorHandler,
)
from ray.llm._internal.serve.deployments.utils.server_utils import (
    get_model_request_id,
    get_response_for_error,
    get_serve_request_id,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.observability.usage_telemetry.usage import (
    push_telemetry_report_for_all_models,
)

logger = get_logger(__name__)


class _LLMServerBase(ABC):
    """
    This is the common interface between all the llm deployment. All llm deployments
    need to implement an async constructor, an async predict, and check_health method.
    """

    async def __init__(self, llm_config: LLMConfig):
        """
        Constructor takes in an LLMConfig object and start the underlying engine.
        """
        self._llm_config = llm_config

    @abstractmethod
    async def chat(self, request: ChatCompletionRequest) -> LLMChatResponse:
        """
        Inferencing to the engine for chat, and return the response as LLMChatResponse.
        """
        ...

    @abstractmethod
    async def completions(self, request: CompletionRequest) -> LLMCompletionsResponse:
        """
        Inferencing to the engine for completion api, and return the response as LLMCompletionsResponse.
        """
        ...

    @abstractmethod
    async def check_health(self) -> None:
        """
        Check the health of the replica. Does not return anything. Raise error when
        the engine is dead and needs to be restarted.
        """
        ...

    async def llm_config(self) -> LLMConfig:
        return self._llm_config


class ResponsePostprocessor:
    def __init__(
        self,
        enable_auto_tools: bool = False,
    ) -> None:
        self.metrics_wrapper = StreamingErrorHandler()
        self.enable_auto_tools: bool = enable_auto_tools

    async def handle_failure(
        self, model: str, gen: AsyncGenerator[LLMRawResponse, None]
    ) -> AsyncGenerator[LLMRawResponse, None]:
        async for llm_response in self.metrics_wrapper.handle_failure(model, gen):
            yield llm_response

    @staticmethod
    async def merge_stream(
        response_stream: AsyncGenerator[LLMRawResponse, None]
    ) -> LLMRawResponse:
        responses = [resp async for resp in response_stream]
        return LLMRawResponse.merge_stream(*responses)

    async def _chat_completions_wrapper(
        self,
        model: str,
        generator: AsyncGenerator[LLMRawResponse, None],
        request: ChatCompletionRequest,
        tokenizer: AnyTokenizer,
        tool_parser: Optional[LLMToolParser] = None,
    ) -> AsyncGenerator[Union[ChatCompletionStreamResponse, ErrorResponse], None]:
        had_error = False
        request_id = get_serve_request_id()
        completion_id = get_model_request_id(model)
        finish_reason = None

        yielded_role = False
        all_results = []

        # TODO: use request.N
        num_choices = 1
        previous_num_tokens = [0] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and (request.tools and tool_parser and self.enable_auto_tools
                and request.tool_choice in ['auto', None]))
        
        all_previous_token_ids: Optional[list[list[int]]]
        function_name_returned: Optional[list[bool]] = None

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if tool_choice_auto:
            # These are only required in "auto" tool choice case
            previous_texts = [""] * num_choices
            all_previous_token_ids = [[]] * num_choices
        elif request.tool_choice == "required":
            previous_texts = [""] * num_choices
            function_name_returned = [False] * num_choices
            all_previous_token_ids = None
        else:
            previous_texts, all_previous_token_ids = None, None

        # Prepare the tool parser if it's needed
        try:
            if tool_choice_auto and tool_parser:
                tool_parsers: list[Optional[LLMToolParser]] = [tool_parser] * num_choices
            else:
                tool_parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in tool parser creation.")
            raise e
        
        try:
            async for batched_results in generator:
                for result in batched_results.unpack():
                    all_results.append(result)

                    if result.error:
                        logger.error(f"{result.error}")
                        # Drop finish reason as OpenAI doesn't expect it
                        # for errors in streaming
                        result.finish_reason = None
                        all_results.pop()
                        had_error = True

                        yield result.error
                        # Return early in case of an error
                        break

                    else:
                        i = result.index or 0
                        index_tool_parser = tool_parsers[i]
                        delta_text = result.generated_text or ""
                        finish_reason = result.finish_reason

                        delta_message = DeltaMessage(
                            content=delta_text,
                        )

                        # just update previous_texts and previous_token_ids
                        if tool_choice_auto:
                            assert previous_texts is not None
                            assert all_previous_token_ids is not None
                            previous_text = previous_texts[i]
                            previous_token_ids = all_previous_token_ids[i]
                            current_text = previous_text + delta_text
                            current_token_ids = previous_token_ids + list(
                                result.token_ids)

                        if not yielded_role:
                            choices = [
                                ChatCompletionResponseStreamChoice(
                                    delta=DeltaMessage(role="assistant"),
                                    index=0,
                                    finish_reason=None,
                                    logprobs=ChatCompletionLogProbs(content=[]),
                                )
                            ]
                            yield ChatCompletionStreamResponse(
                                id=completion_id,
                                model=model,
                                choices=choices,
                                usage=None,
                            )
                            yielded_role = True

                        logprobs = None
                        if result.logprobs:
                            logprobs = ChatCompletionLogProbs(
                                content=[
                                    ChatCompletionLogProbsContent(
                                        token=logprobs.token,
                                        logprob=logprobs.logprob,
                                        bytes=logprobs.bytes,
                                        top_logprobs=[
                                            ChatCompletionLogProb(
                                                token=logprob.token,
                                                logprob=logprob.logprob,
                                                bytes=logprob.bytes,
                                            )
                                            for logprob in logprobs.top_logprobs
                                        ],
                                    )
                                    for logprobs in result.logprobs
                                ]
                            )

                        if tool_choice_function_name:
                            delta_message.tool_calls=[
                                DeltaToolCall(function=DeltaFunctionCall(
                                    name=tool_choice_function_name,
                                    arguments=delta_text),
                                              index=i)
                            ]
                        elif request.tool_choice == "required":
                            assert previous_texts is not None
                            assert function_name_returned is not None
                            previous_text = previous_texts[i]
                            current_text = previous_text + delta_text
                            #fn_name_returned = function_name_returned[i]

                            # TODO
                            # delta_message, function_name_returned[i] = (
                            #     self.extract_tool_call_required_streaming(
                            #         previous_text=previous_text,
                            #         current_text=current_text,
                            #         delta_text=delta_text,
                            #         function_name_returned=fn_name_returned))

                            # update the previous values for the next iteration
                            previous_texts[i] = current_text
                        
                        elif tool_choice_auto:
                            assert index_tool_parser is not None
                            tool_parsed_delta_message = index_tool_parser.extract_tool_calls_streaming(
                                previous_text=previous_text,
                                current_text=current_text,
                                delta_text=delta_text,
                                previous_token_ids=previous_token_ids,
                                current_token_ids=current_token_ids,
                                delta_token_ids=result.token_ids,
                                request=request
                            )
                            if tool_parsed_delta_message:
                                delta_message = tool_parsed_delta_message
                            
                        # update the previous values for the next iteration
                        if tool_choice_auto:
                            assert previous_texts is not None
                            assert all_previous_token_ids is not None
                            previous_texts[i] = current_text
                            all_previous_token_ids[i] = current_token_ids

                        # set the previous values for the next iteration
                        previous_num_tokens[i] += len(result.token_ids)

                        # if the message delta is None (e.g. because it was a
                        # "control token" for tool calls or the parser otherwise
                        # wasn't ready to send a token, then
                        #   get the next token without streaming a chunk
                        if delta_message is None:
                            continue

                        choices = [
                            ChatCompletionResponseStreamChoice(
                                delta=delta_message,
                                index=i,
                                finish_reason=None,
                                logprobs=logprobs,
                            )
                        ]

                        resp = ChatCompletionStreamResponse(
                            id=completion_id,
                            model=model,
                            choices=choices,
                            usage=None,
                        )
                        yield resp

                if had_error:
                    # Return early in case of an error
                    break

        except Exception as e:
            logger.error(
                f"Failed while handling chat-completions for request ({request_id}): {repr(e)}",
                exc_info=e,
            )

            yield get_response_for_error(e, request_id).error
            had_error = True

        if not had_error:
            choices = [
                ChatCompletionResponseStreamChoice(
                    delta=DeltaMessage(),
                    index=0,
                    finish_reason=finish_reason,
                )
            ]
            merged_results = LLMRawResponse.merge_stream(*all_results)
            usage = UsageInfo(
                prompt_tokens=merged_results.num_input_tokens or 0,
                completion_tokens=merged_results.num_generated_tokens or 0,
                total_tokens=(merged_results.num_input_tokens or 0)
                + (merged_results.num_generated_tokens or 0),
            )
            yield ChatCompletionStreamResponse(
                id=completion_id,
                model=model,
                choices=choices,
                usage=usage,
            )

    @staticmethod
    async def _completions_wrapper(
        model: str,
        generator: AsyncGenerator[LLMRawResponse, None],
    ) -> AsyncGenerator[Union[CompletionStreamResponse, ErrorResponse], None]:
        all_results = []
        request_id = get_serve_request_id()
        had_error = False
        try:
            async for batched_results in generator:
                for result in batched_results.unpack():
                    all_results.append(result)
                    if result.error:
                        # Drop finish reason as OpenAI doesn't expect it
                        # for errors in streaming
                        result.finish_reason = None
                        logger.error(
                            f"Reporting back an error: {result.error}",
                            extra={"ray_serve_extra_fields": {"response": str(result)}},
                        )
                        all_results.pop()
                        had_error = True

                        yield result.error
                        # Return early in case of an error
                        break

                    choices = [
                        CompletionResponseStreamChoice(
                            text=result.generated_text or "",
                            index=0,
                            logprobs={},
                            finish_reason=result.finish_reason,
                        )
                    ]

                    usage = None
                    if result.finish_reason:
                        merged_results = LLMRawResponse.merge_stream(*all_results)
                        usage = UsageInfo(
                            prompt_tokens=merged_results.num_input_tokens or 0,
                            completion_tokens=merged_results.num_generated_tokens or 0,
                            total_tokens=(merged_results.num_input_tokens or 0)
                            + (merged_results.num_generated_tokens or 0),
                        )

                    yield CompletionStreamResponse(
                        id=get_model_request_id(model),
                        model=model,
                        choices=choices,
                        usage=usage,
                    )

                if had_error:
                    # Return early in case of an error
                    break

        except Exception as e:
            logger.error(
                f"Failed while handling completions for request ({request_id}): {repr(e)}",
                exc_info=e,
            )

            yield get_response_for_error(e, request_id).error

    async def process_chat(
        self, 
        request: ChatCompletionRequest,
        model: str, 
        gen: AsyncGenerator[LLMRawResponse, None], 
        stream: bool,
        tokenizer: AnyTokenizer,
        tool_parser: Optional[LLMToolParser] = None,
    ) -> LLMChatResponse:
        gen = self.handle_failure(model=model, gen=gen)

        if stream:
            async for response in self._chat_completions_wrapper(
                model=model,
                generator=gen,
                request=request,
                tokenizer=tokenizer,
                tool_parser=tool_parser,
            ):
                yield response
        else:
            results: LLMRawResponse = await self.merge_stream(gen)
            if results.error:
                yield results.error

            logprobs = None
            if results.logprobs:
                logprobs = ChatCompletionLogProbs(
                    content=[
                        ChatCompletionLogProbsContent(
                            token=logprobs.token,
                            logprob=logprobs.logprob,
                            bytes=logprobs.bytes,
                            top_logprobs=[
                                ChatCompletionLogProb(
                                    token=logprob.token,
                                    logprob=logprob.logprob,
                                    bytes=logprob.bytes,
                                )
                                for logprob in logprobs.top_logprobs
                            ],
                        )
                        for logprobs in results.logprobs
                    ]
                )
            
            generated_text = results.generated_text or ""

            message = ChatMessage(
                role="assistant",
                content=generated_text or "",
            )

            # Handle tool calling extraction
            auto_tools_called = False
            # handle when tool_choice is explicitly set to none
            if request.tool_choice == "none":
                logger.info("Tool choice is set to 'none', no tool calls will be made.")
                pass
                
            # handle when there are tools and tool choice is auto
            elif request.tools and (
                request.tool_choice == "auto"
                or request.tool_choice is None) and self.enable_auto_tools \
                    and tool_parser:
                tool_call_info = tool_parser.extract_tool_calls(
                    generated_text, request=request
                )
                logger.info(f"Tool call info: {tool_call_info}")
                auto_tools_called = tool_call_info.tools_called
                if tool_call_info.tools_called:
                    message.tool_calls = tool_call_info.tool_calls
                    message.content = tool_call_info.content

            # handle when request uses tools and specified a tool choice
            elif request.tool_choice and type(
                request.tool_choice) is ChatCompletionNamedToolChoiceParam:

                tool_call_class = MistralToolCall if isinstance(
                    tokenizer, MistralTokenizer) else ToolCall
                message.content=""
                message.tool_calls=[
                    tool_call_class(function=FunctionCall(
                        name=request.tool_choice.function.name,
                        arguments=generated_text))
                ]

            choices = [
                ChatCompletionResponseChoice(
                    message=message,
                    index=0,
                    finish_reason="tool_calls" if auto_tools_called else results.finish_reason,
                    logprobs=logprobs,
                )
            ]
            usage = UsageInfo(
                prompt_tokens=results.num_input_tokens or 0,
                completion_tokens=results.num_generated_tokens or 0,
                total_tokens=(results.num_input_tokens or 0)
                + (results.num_generated_tokens or 0),
            )
            yield ChatCompletionResponse(
                id=get_model_request_id(model),
                model=model,
                choices=choices,
                usage=usage,
            )

    async def process_completions(
        self, model: str, gen: AsyncGenerator[LLMRawResponse, None], stream: bool
    ) -> LLMCompletionsResponse:
        gen = self.handle_failure(model=model, gen=gen)

        if stream:
            async for response in self._completions_wrapper(
                model=model,
                generator=gen,
            ):
                yield response
        else:
            results: LLMRawResponse = await self.merge_stream(gen)
            if results.error:
                yield results.error

            choices = [
                CompletionResponseChoice(
                    text=results.generated_text or "",
                    index=0,
                    logprobs={},
                    finish_reason=results.finish_reason,
                )
            ]
            usage = UsageInfo(
                prompt_tokens=results.num_input_tokens or 0,
                completion_tokens=results.num_generated_tokens or 0,
                total_tokens=(results.num_input_tokens or 0)
                + (results.num_generated_tokens or 0),
            )
            yield CompletionResponse(
                id=get_model_request_id(model),
                model=model,
                choices=choices,
                usage=usage,
            )


class LLMServer(_LLMServerBase):
    _default_engine_cls = VLLMEngine
    _default_image_retriever_cls = ImageRetriever

    async def __init__(
        self,
        llm_config: LLMConfig,
        *,
        engine_cls: Optional[Type[VLLMEngine]] = None,
        image_retriever_cls: Optional[Type[ImageRetriever]] = None,
        model_downloader: Optional[LoraModelLoader] = None,
    ):
        """Constructor of LLMServer.

        Only the llm_config is public api, the other arguments are private
        and used for testing.

        Args:
            llm_config: LLMConfig for the model.

        Keyword Args:
            engine_cls: Dependency injection for the vllm engine class. Defaults to
                `VLLMEngine`.
            image_retriever_cls: Dependency injection for the image retriever class.
                Defaults to `ImageRetriever`.
            model_downloader: Dependency injection for the model downloader object.
                Defaults to be initialized with `LoraModelLoader`.
        """
        await super().__init__(llm_config)

        self._engine_cls = engine_cls or self._default_engine_cls
        self.engine = self._get_engine_class(self._llm_config)
        await asyncio.wait_for(self._start_engine(), timeout=ENGINE_START_TIMEOUT_S)

        self.tokenizer = self.engine.get_tokenizer()

        # set up tool use
        enable_auto_tools: bool = llm_config.enable_auto_tools
        tool_parser_manager: Optional[LLMToolParserManager] = None
        if isinstance(self.engine, VLLMEngine):
            logger.info("Using VLLMToolParserManager for tool parsing.")
            tool_parser_manager = VLLMToolParserManager(self.tokenizer)

        if enable_auto_tools and tool_parser_manager:
            try:
                self.tool_parser = tool_parser_manager.get_tool_parser(
                    llm_config.tool_parser)
            except Exception as e:
                raise TypeError("Error: --enable-auto-tool-choice requires "
                                f"tool_parser:'{llm_config.tool_parser}' which has not "
                                "been registered") from e
            
        self.response_postprocessor = ResponsePostprocessor(
            enable_auto_tools=enable_auto_tools,
        )

        self.image_retriever = (
            image_retriever_cls()
            if image_retriever_cls
            else self._default_image_retriever_cls()
        )

        multiplex_config = self._llm_config.multiplex_config()
        if model_downloader:
            self.model_downloader = model_downloader
        elif multiplex_config:
            self.model_downloader = LoraModelLoader(
                download_timeout_s=multiplex_config.download_timeout_s,
                max_tries=multiplex_config.max_download_tries,
            )
        else:
            self.model_downloader = LoraModelLoader()

        # Hack that lets us set max_num_models_per_replica from the llm_config
        if multiplex_config:
            self.load_model = serve.multiplexed(
                max_num_models_per_replica=multiplex_config.max_num_models_per_replica
            )(lambda lora_model_id: self._load_model(lora_model_id))

    @property
    def _get_engine_class(self) -> Type[LLMEngine]:
        """Helper to load the engine class from the environment variable.

        This is used for testing or escape-hatch for patching purposes.
        If env variable is not set, it will fallback to the default engine class.
        """
        engine_cls_path = os.environ.get(RAYLLM_VLLM_ENGINE_CLS_ENV)
        if engine_cls_path:
            try:
                return import_attr(engine_cls_path)
            except AttributeError:
                logger.warning(
                    f"Failed to import engine class {engine_cls_path}. "
                    f"Using the default engine class {self._engine_cls}."
                )
        return self._engine_cls

    async def _start_engine(self):
        await self.engine.start()

        # Push telemetry reports for the model in the current deployment.
        # Note: the model architecture is only available after node initialized and the
        # engine is started.
        if self._llm_config.model_architecture:
            push_telemetry_report_for_all_models(all_models=[self._llm_config])

    async def _predict(
        self,
        request_id: str,
        prompt: Prompt,
        stream: bool,
    ) -> AsyncGenerator[LLMRawResponse, None]:
        """A thin wrapper around VLLMEngine.generate().

        1. Load the model to disk
        2. Format parameters correctly
        3. Forward request to VLLMEngine.generate()
        """

        logger.info(f"Received streaming request {request_id}")
        multiplexed_model_id = serve.get_multiplexed_model_id()

        if multiplexed_model_id:
            assert (
                self._llm_config.lora_config is not None
            ), "Must setup lora config for multiplexed requests."
            disk_lora_model = await self._disk_lora_model(multiplexed_model_id)
        else:
            disk_lora_model = None

        llm_request = await self.engine.prepare_request(
            request_id=request_id,
            prompt=prompt,
            stream=stream,
            disk_lora_model=disk_lora_model,
        )

        async for llm_response in self.engine.generate(llm_request):
            yield llm_response

    async def chat(self, request: ChatCompletionRequest) -> LLMChatResponse:
        """Runs a chat request to the vllm engine, and return the response.

        Args:
            request: A ChatCompletionRequest object.

        Returns:
            A LLMChatResponse object.
        """
        request_id = get_serve_request_id()
        prompt = Prompt(
            prompt=[Message.model_validate(message) for message in request.messages],
            parameters=request,
        )
        stream = request.stream
        gen = self._predict(request_id=request_id, prompt=prompt, stream=stream)
        return self.response_postprocessor.process_chat(
            request=request, model=self._llm_config.model_id, gen=gen, stream=stream, tool_parser=self.tool_parser, tokenizer=self.tokenizer
        )

    async def completions(self, request: CompletionRequest) -> LLMCompletionsResponse:
        """Runs a completion request to the vllm engine, and return the response.

        Args:
            request: A CompletionRequest object.

        Returns:
            A LLMCompletionsResponse object.
        """
        request_id = get_serve_request_id()
        prompt = Prompt(
            prompt=request.prompt,
            parameters=request,
            use_prompt_format=False,
        )
        stream = request.stream
        gen = self._predict(request_id=request_id, prompt=prompt, stream=stream)
        return self.response_postprocessor.process_completions(
            model=self._llm_config.model_id, gen=gen, stream=stream
        )

    async def check_health(self) -> bool:
        """Check the health of the llm engine."""
        return await self.engine.check_health()

    async def _load_model(self, lora_model_id: str) -> DiskMultiplexConfig:
        return await self.model_downloader.load_model(
            lora_model_id=lora_model_id,
            llm_config=self._llm_config,
        )

    async def _disk_lora_model(self, lora_model_id: str) -> DiskMultiplexConfig:
        disk_lora_model: DiskMultiplexConfig = await self.load_model(lora_model_id)
        return disk_lora_model

    @classmethod
    def as_deployment(
        cls, deployment_options: Dict[str, Any] = None
    ) -> serve.Deployment:
        """Convert the LLMServer to a Ray Serve deployment.

        Args:
            deployment_options: A dictionary of deployment options.

        Returns:
            A Ray Serve deployment.
        """
        deployment_options = deployment_options or {}
        return LLMDeployment.options(**deployment_options)


@serve.deployment(
    # TODO make this configurable
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": int(
            os.environ.get(
                "RAYLLM_ROUTER_TARGET_ONGOING_REQUESTS",
                os.environ.get(
                    "RAYLLM_ROUTER_TARGET_NUM_ONGOING_REQUESTS_PER_REPLICA", 10
                ),
            )
        ),
    },
    max_ongoing_requests=20,  # Maximum backlog for a single replica
    health_check_period_s=DEFAULT_HEALTH_CHECK_PERIOD_S,
    health_check_timeout_s=DEFAULT_HEALTH_CHECK_TIMEOUT_S,
)
class LLMDeployment(LLMServer):
    # Note (genesu): We are separating the LLMServer and LLMDeployment just
    # to give developers an ability to test the implementation outside the Ray Serve.
    # But in practice we should always test the LLMDeployment class as a Serve
    # deployment to ensure all functionalities can be run remotely asynchronously.
    ...
