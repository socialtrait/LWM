import time

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Union, Literal, Any
from http import HTTPStatus

from openai_api_server.utils import (
    random_uuid,
    estimate_tokens_from_prompt,
    estimate_tokens_from_messages,
    create_utc_timestamp,
)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion


class RateLimitInfo(BaseModel):
    max_tpm: int
    max_rpm: Optional[int] = None
    max_rps: Optional[int] = None

    current_tpm: int = 0
    current_rpm: int = 0
    current_rps: int = 0


class ModelEndpoint(BaseModel):
    endpoint_id: str = Field(default_factory=lambda: f"ep-{random_uuid()}")
    services: List[str]
    model_name: str
    provider: str
    deployment_name: str
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    api_base: Optional[str] = None
    max_tpm: Optional[int] = None
    max_rpm: Optional[int] = None
    # tpm: int = 0
    # rpm: int = 0

    @field_validator("services")
    def normalize_service_name(cls, value):
        for i in range(len(value)):
            value[i] = value[i].upper().replace(" ", "_").replace("-", "_")
        return value


class OpenAIEndpoint(ModelEndpoint):
    api_key: str


class AzureOpenAIEndpoint(OpenAIEndpoint):
    api_version: str
    api_base: str


class MessageProperties(BaseModel):
    content_type: Optional[str] = None
    content_encoding: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    delivery_mode: Optional[int] = None
    priority: Optional[int] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expiration: Optional[str] = None
    message_id: Optional[str] = None
    timestamp: Optional[int] = None
    type: Optional[str] = None
    user_id: Optional[str] = None
    app_id: Optional[str] = None
    reserved: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class BaseRequest(BaseModel):
    # Additional parameters 
    pass


class ChatCompletionRequest(BaseRequest):
    model: str
    messages: Union[str, List[Dict[str, str]]]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[bool] = None
    logit_bias: Optional[Dict[str, float]] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None



class CompletionRequest(BaseRequest):
    model: str
    # a string, array of strings, array of tokens, or array of token arrays
    prompt: Union[List[int], List[List[int]], str, List[str]]
    max_tokens: Optional[int] = 1024
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[bool] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None



class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(Completion):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(ChatCompletion):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(
        default=None, description="data about request and response"
    )


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[Union[int, str, HTTPStatus]] = None


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "N/A"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)
