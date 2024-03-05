import uuid
import json
import base64
import tiktoken
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Union, Optional, Literal, Any



# adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# TODO: add non-OpenAI models


def num_tokens_from_messages(
    messages: List[Dict[str, str]], model="gpt-3.5-turbo-0613"
):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        print(f"Warning: using gpt-4-0613 tokenizer for unknown model {model}.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    # else:
    #     raise NotImplementedError(
    #         f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
    #     )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_prompt(prompt: Union[str, List[int]], model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a prompt."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(prompt, str):
        return len(encoding.encode(prompt))
    else:
        print("Computing input token number for already tokenized sequence")
        return len(prompt)


def estimate_tokens(text, method="max"):
    # method can be "average", "words", "chars", "max", "min", defaults to "max"
    # "average" is the average of words and chars
    # "words" is the word count divided by 0.75
    # "chars" is the char count divided by 4
    # "max" is the max of word and char
    # "min" is the min of word and char
    word_count = len(text.split(" "))
    char_count = len(text)
    tokens_count_word_est = word_count / 0.75
    tokens_count_char_est = char_count / 4.0
    output = 0
    if method == "average":
        output = (tokens_count_word_est + tokens_count_char_est) / 2
    elif method == "words":
        output = tokens_count_word_est
    elif method == "chars":
        output = tokens_count_char_est
    elif method == "max":
        output = max([tokens_count_word_est, tokens_count_char_est])
    elif method == "min":
        output = min([tokens_count_word_est, tokens_count_char_est])
    else:
        # return invalid method message
        return "Invalid method. Use 'average', 'words', 'chars', 'max', or 'min'."

    return int(output)


def estimate_tokens_from_messages(
    messages: List[Dict[str, str]], tokens_per_message=0, method="max"
):
    """Return the estimated number of tokens used by a list of messages."""
    total_tokens = 0
    for message in messages:
        total_tokens += tokens_per_message
        for key, value in message.items():
            total_tokens += estimate_tokens(value, method)
    return total_tokens


def estimate_tokens_from_prompt(prompt: Union[str, List[int]], method="max"):
    """Return the estimated number of tokens used by a prompt."""
    if isinstance(prompt, str):
        return estimate_tokens(prompt, method)
    else:
        print("Computing input token number for already tokenized sequence")
        return len(prompt)


# for fastapi app
def add_consumer_endpoint(
    base_url: str,
    api_key: str,
    api_version: str,
    deployment_name: str,
    model_name: str,
    services: List[str],
):
    pass


def base64_encode_dict(d: dict) -> dict:
    """
    Returns the base64 encoding of the given dict.
    """
    dict_str = json.dumps(d)
    encoded_dict = base64.b64encode(dict_str.encode("utf-8"))
    return encoded_dict.decode("utf-8")


def base64_decode_dict(d: dict) -> dict:
    """
    Returns the base64 decoding of the given dict.
    """
    decoded_dict = base64.b64decode(d)
    dict_str = decoded_dict.decode("utf-8")
    return json.loads(dict_str)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def create_utc_timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def string_to_utc_timestamp(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


def timestamp_difference(start, end):
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    start_dt = datetime.strptime(start, fmt)
    end_dt = datetime.strptime(end, fmt)
    return (end_dt - start_dt).total_seconds()
