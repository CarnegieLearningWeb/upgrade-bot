import os
import json
import tiktoken
from openai import OpenAI
from anthropic import Anthropic

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TARGET_LLM = os.getenv("TARGET_LLM", "gpt").lower()

# Check if any required environment variable is missing or invalid
missing_vars = [var_name for var_name, var_value in [
    ("SLACK_BOT_TOKEN", SLACK_BOT_TOKEN),
    ("SLACK_APP_TOKEN", SLACK_APP_TOKEN),
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    ("TARGET_LLM", TARGET_LLM)
] if var_value is None]

if missing_vars:
    raise ValueError(f"""Missing required environment variables: {", ".join(missing_vars)}""")

if TARGET_LLM not in ["gpt", "claude"]:
    raise ValueError(f"Invalid TARGET_LLM: {TARGET_LLM}")

# OpenAI client
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)
GPT_MODEL = "gpt-4-0125-preview"
GPT_TEMPERATURE = 0.1

# Anthropic client
anthropic_client = Anthropic(
    api_key=ANTHROPIC_API_KEY,
)
CLAUDE_MODEL = "claude-3-opus-20240229"
CLAUDE_TEMPERATURE = 0.1
CLAUDE_MAX_TOKENS = 1024

# Target model
TARGET_MODEL = GPT_MODEL if TARGET_LLM == "gpt" else CLAUDE_MODEL

# Context processing variables
BASE_DIRECTORY_PATH = "context"
PROCESSED_DIRECTORIES = ["design", "reference"]
PROCESSED_CODE_EXTENSIONS = {"json", "html", "css", "scss", "ts", "txt"}
PROCESSED_IMAGE_EXTENSIONS = {"png", "webp", "gif"}


def minify_json_content(filepath, content):
    try:
        json_object = json.loads(content)
        return json.dumps(json_object, separators=(",", ":"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding {filepath}: {e}") from e


def process_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if filepath.endswith(".json"):
                content = minify_json_content(filepath, content)
        return content
    except UnicodeDecodeError as e:
        raise ValueError(f"Error reading {filepath}: {e}") from e


def process_directory(folder_path):
    directory_content = {}
    sorted_entries = sorted(os.listdir(folder_path))
    for entry in sorted_entries:
        full_path = os.path.join(folder_path, entry)
        if os.path.isdir(full_path):
            directory_content[entry] = process_directory(full_path)
        elif os.path.isfile(full_path) and os.path.splitext(entry)[1][1:].lower() in PROCESSED_CODE_EXTENSIONS:
            directory_content[entry] = process_file(full_path)
        else:
            print(f"Ignoring file: {full_path}")
    return directory_content


def build_context(base_path):
    context = {}

    # Check if base_path exists and is a directory
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"The base path {base_path} does not exist or is not a directory")
    
    # Check for required folders and process accordingly
    for folder in PROCESSED_DIRECTORIES:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Required folder {folder} is missing in the base path {base_path}")
        if folder == "images":
            context[folder] = [file for file in os.listdir(folder_path) if os.path.splitext(file)[1][1:].lower() in PROCESSED_IMAGE_EXTENSIONS]
        else:
            context[folder] = process_directory(folder_path)
    
    return context


context = build_context(BASE_DIRECTORY_PATH)


SYSTEM_PROMPT = f"""
You are a software engineer with expertise in the Angular framework. Your mission is to provide design and development support for UpGrade, an open source A/B testing and feature flagging platform for education software.
When offering code suggestions or examples, ensure they align with the best practices and code patterns demonstrated in the provided 'reference' context.
If the 'design' context provides specific requirements or specifications, aim to integrate them seamlessly with the Angular MDC best practices.
Your responses should be focused, relevant, and grounded in the provided context. Avoid making assumptions or providing information not directly supported by the given 'design' and 'reference' materials.

Context includes:
- 'design': detailed JSON structure outlining the design requirements and specifications for UpGrade.
- 'reference': documentation and examples related to Angular MDC, serving as a guideline for implementing Material Design in an Angular project.

Please utilize this context:
{context}
"""

print("\n")
print(SYSTEM_PROMPT)
print("\n")


# From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model=TARGET_MODEL):
    # Number of tokens used
    num_tokens = 0

    # Return the number of tokens used in Claude messages
    if model == CLAUDE_MODEL:
        for message in messages:
            for key in ["role", "content"]:
                if key in message:
                    num_tokens += anthropic_client.count_tokens(message[key])
            return num_tokens

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
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


num_system_prompt_tokens = num_tokens_from_messages([{"role": "system", "content": SYSTEM_PROMPT}])

print(f"Target model: {TARGET_MODEL}")
print(f"The number of tokens used for system prompt: {num_system_prompt_tokens}\n")


def process_conversation_history(conversation_history, bot_user_id):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in conversation_history["messages"][:-1]:
        role = "assistant" if message["user"] == bot_user_id else "user"
        message_text = process_message(message, bot_user_id)
        if message_text:
            messages.append({"role": role, "content": message_text})
    return messages


def process_message(message, bot_user_id):
    message_text = message["text"]
    role = "assistant" if message["user"] == bot_user_id else "user"
    message_text = clean_message_text(message_text, role, bot_user_id)
    return message_text


def clean_message_text(message_text, role, bot_user_id):
    if (f"<@{bot_user_id}>" in message_text) or (role == "assistant"):
        message_text = message_text.replace(f"<@{bot_user_id}>", "").strip()
        return message_text
    return None


def update_chat(app, channel_id, reply_message_ts, response_text):
    app.client.chat_update(
        channel=channel_id,
        ts=reply_message_ts,
        text=response_text
    )

