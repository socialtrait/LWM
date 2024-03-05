# host a mock completion endpoint with fastAPI
import os
import subprocess
import threading
import uuid

import asyncio
import argparse
import fastapi
import uvicorn
from time import sleep, time
from fastapi import Request, HTTPException, status
from fastapi import FastAPI, File, Form, HTTPException, Path, UploadFile, status
from fastapi.responses import JSONResponse

from typing import List, Optional

from starlette.exceptions import HTTPException as StarletteHTTPException

from openai_api_server.data_models import (
    CompletionRequest,
    ChatCompletionRequest,
    CompletionResponse,
    ChatCompletionResponse,
    CompletionResponseChoice,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
    FileObject,
)

from openai.types.chat.chat_completion import (
    ChatCompletion,
    CompletionUsage,
    Choice,
    ChatCompletionMessage,
)

from dotenv import load_dotenv

load_dotenv()

from logging import getLogger
logger = getLogger("api-server")


tpm_limit = None
tpm_log = []

update_token_log_lock = asyncio.Lock()


class RateLimitError(Exception):
    status_code = 429


async def update_tpm_log(num_tokens: int):
    global tpm_log, update_token_log_lock

    async with update_token_log_lock:
        tpm_log = tpm_log or []
        tpm_log.append((time(), num_tokens))
        logger.debug(f"Updated tpm log with entry {tpm_log[-1]}")


async def get_tokens_last_minute():
    global tpm_log, update_token_log_lock

    async with update_token_log_lock:
        tpm_log = tpm_log or []
        now = time()
        token_entries_last_min = [n for t, n in tpm_log if t > now - 60]
        sum_tokens = sum(token_entries_last_min)
        logger.debug(f"Got {sum_tokens} tokens in the last minute")
        return sum_tokens


async def check_tpm_limit(new_tokens: int):
    """
    Raises rate limit error if the token per minute limit is reached.
    """
    global tpm_limit
    
    if tpm_limit is not None:
        tokens_last_minute = await get_tokens_last_minute()
        if tokens_last_minute + new_tokens > tpm_limit:
            logger.error(
                f"Token per minute limit of {tpm_limit} reached. Tokens last minute: {tokens_last_minute}, new tokens: {new_tokens}"
            )
            raise RateLimitError(
                f"Token per minute limit of {tpm_limit} reached. Please wait for a while and try again."
            )


async def create_completion(
    request: CompletionRequest,
) -> CompletionResponse:

    raise NotImplementedError("Text completion not implemented yet")

    await update_tpm_log(new_tokens)
    return response


async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:

    new_tokens = 0
    await check_tpm_limit(new_tokens)

    # new data format
    response = ChatCompletionResponse(
        id="chatcmpl-123",
        created=int(time()),
        object="chat.completion",
        model="mock-gpt",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="This is a mock chat completion, using 100 mock completion tokens.",
                ),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=request.input_tokens,
            total_tokens=request.input_tokens + 100,
            completion_tokens=100,
        ),
    )

    await update_tpm_log(new_tokens)
    return response


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle exceptions and return proper error message to client.
    """
    if isinstance(exc, StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": f"{exc.detail}"},
        )
    elif isinstance(exc, RateLimitError):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"{exc.__class__.__name__}: {exc}"},
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"{exc.__class__.__name__}: {exc}"},
        )


app = fastapi.FastAPI(
    title="OpenAI-Compatible Completions API server",
    description="An API server in OpenAI format.",
    version="0.0.1",
)

app.add_exception_handler(Exception, general_exception_handler)


@app.get("/health")
async def health():
    return {"status": "ok"}


########################
# COMPLETION ENDPOINTS #
########################

@app.post("/completions")
async def completion(request: CompletionRequest):
    logger.debug(request)
    try:
        response = await create_completion(request)
        logger.debug(response)
        return response
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    logger.debug(request)
    try:
        response = await create_chat_completion(request)
        logger.debug(response)
        return response
    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


##################
# FILE ENDPOINTS #
##################


# Your mock database for the example
mock_database = []

async def get_file_from_db(file_id: str):
    for file in mock_database:
        if file.id == file_id:
            return file
    return None

async def list_files_from_db(purpose: Optional[str] = None):
    if purpose:
        return [file for file in mock_database if file.purpose == purpose]
    return mock_database

async def delete_file_from_db(file_id: str):
    global mock_database
    mock_database = [file for file in mock_database if file.id != file_id]

async def create_file(file: UploadFile, purpose: str):
    # Create a unique ID for the file
    file_id = str(uuid.uuid4())

    os.makedirs("temp_files", exist_ok=True)

    # Define file path (for example purpose, you might want to save it somewhere specific)
    file_location = f"temp_files/{file_id}_{file.filename}"

    # Write the uploaded file to a new location
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())  # Writing the file content to the local file system

    # Assuming you have a function to get file size, or use os.path.getsize
    file_size = os.path.getsize(file_location)

    # Creating the file object that will be saved to your mock database
    new_file = FileObject(
        id=file_id,
        filename=file.filename,
        bytes=file_size,
        created_at=int(time()), 
        purpose=purpose
    )

    # Append the new file to your database (here, it's a mock database)
    mock_database.append(new_file)

    logger.debug(f"Uploaded file {file.filename} with id {file_id}")

    # Return the file metadata
    return new_file


async def retrieve_file_content_from_db(file_id: str):
    file = await get_file_from_db(file_id)
    if file:
        return {"content": file.bytes}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="File not found",
    )



@app.post("/files", response_model=FileObject)
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    try:
        new_file = await create_file(file, purpose)
        # Return the file metadata
        return new_file

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.get("/files", response_model=List[FileObject])
async def list_files(purpose: Optional[str] = None):
    try:
        files = await list_files_from_db(purpose)
        return files
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.get("/files/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str = Path(...)):
    try:
        file = await get_file_from_db(file_id)
        if file:
            return file
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.delete("/files/{file_id}")
async def delete_file(file_id: str = Path(...)):
    try:
        await delete_file_from_db(file_id)
        return {"detail": "File deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@app.get("/files/{file_id}/content")
async def retrieve_file_content(file_id: str = Path(...)):
    try:
        file_content = await retrieve_file_content_from_db(file_id)
        return {"content": file_content}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )



def parse_args():
    parser = argparse.ArgumentParser(description="Mock OpenAI API server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--tpm",
        type=int,
        default=None,
        help="Token per minute limit for the mock server",
    )

    return parser.parse_args()


def main(host: str, port: int, tpm: int = None):
    global tpm_limit
    tpm_limit = tpm
    logger.info(
        f"Starting OpenAI-compatible API server at {host}:{port} with tpm limit {tpm_limit}"
    )

    # run the Uvicorn server
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    args = parse_args()
    main(args.host, args.port, args.tpm)
