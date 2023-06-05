import os
import json
import tiktoken


SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_json_context(filename):
    try:
        with open(filename, "r") as file:
            data = json.load(file)

        minified_json_string = json.dumps(data, separators=(",", ":"))
        return minified_json_string

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return ""

design_context = get_json_context("design.json")

SYSTEM_PROMPT = f'''
You are an AI assistant.
Only answer questions related to UpGrade, based on the provided context.
If you are certain that the question is not relevant to UpGrade, say "As an UpGradeBot, I can only answer questions about UpGrade."
If you cannot find the exact information from the context, say "Sorry, I don't have that information."

Context:
UpGrade, backed by the Gates Foundation and Schmidt Futures, is a free, open-source A/B testing platform by Carnegie Learning and PlayPower Labs for educational software in schools. Built with Angular, it enables quick data-driven decisions, addressing educational software concerns like group random assignment and reduced teacher burden. The platform unites teachers, researchers, and edtech companies for improving learning outcomes and has been used in studies with MATHia and Battleship Numberline. Visit upgradeplatform.org, access docs at upgrade-platform.gitbook.io/docs/, and find the repository at github.com/CarnegieLearningWeb/UpGrade
UI Design (Angle brackets represent non-UI element descriptions; curly braces denote formatted variables/placeholders; square brackets indicate input UI elements; labels starting with an asterisk are emphasized):
{design_context}
'''

WAIT_MESSAGE = "Got your request. Please wait..."
N_CHUNKS_TO_CONCAT_BEFORE_UPDATING = 10
MAX_TOKENS = 8192


# From https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


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

