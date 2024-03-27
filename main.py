import os
from threading import Event
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

from utils import (SLACK_APP_TOKEN, SLACK_BOT_TOKEN, openai_client, anthropic_client,
                   GPT_MODEL, GPT_TEMPERATURE, CLAUDE_MODEL, CLAUDE_TEMPERATURE, CLAUDE_MAX_TOKENS, TARGET_MODEL,
                   num_tokens_from_messages, process_conversation_history, update_chat)

app = App(token=SLACK_BOT_TOKEN)

# Settings
N_CHUNKS_TO_CONCAT_BEFORE_UPDATING = 10
WAIT_MESSAGE = "Got your request. Please wait..."
MAX_TOKEN_MESSAGE = "Apologies, but the maximum number of tokens for this thread has been reached. Please start a new thread to continue discussing this topic."

# A dictionary to store an event for each channel
events = {}


def get_conversation_history(channel_id, thread_ts):
    return app.client.conversations_replies(
        channel=channel_id,
        ts=thread_ts,
        inclusive=True
    )


def stream_openai_request(messages, channel_id, reply_message_ts):
    openai_response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        temperature=GPT_TEMPERATURE,
        messages=messages,
        stream=True
    )
    response_text = ""
    ii = 0
    for chunk in openai_response:
        if chunk.choices[0].delta.content is not None:
            ii += 1
            response_text += chunk.choices[0].delta.content
            if ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
                update_chat(app, channel_id, reply_message_ts, response_text)
                ii = 0
        elif chunk.choices[0].finish_reason == "stop":
            update_chat(app, channel_id, reply_message_ts, response_text)
        elif chunk.choices[0].finish_reason == "length":
            update_chat(app, channel_id, reply_message_ts, response_text + "...\n\n" + MAX_TOKEN_MESSAGE)


def stream_anthropic_request(messages, channel_id, reply_message_ts):
    anthropic_response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        temperature=CLAUDE_TEMPERATURE,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=messages[0]["content"],
        messages=messages[1:],
        stream=True,
    )
    response_text = ""
    input_tokens = None
    output_tokens = None
    ii = 0
    for event in anthropic_response:
        ii += 1
        match event.type:
            case "message_start":
                input_tokens = event.message.usage.input_tokens
            case "content_block_start":
                response_text += event.content_block.text
            case "content_block_delta":
                response_text += event.delta.text
            case "message_delta":
                output_tokens = event.usage.output_tokens
            case "content_block_stop" | "message_stop":
                ...
        if ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
            update_chat(app, channel_id, reply_message_ts, response_text)
            ii = 0

    update_chat(app, channel_id, reply_message_ts, response_text)
    print(f"Cost incurred: ${input_tokens * 0.000015 + output_tokens * 0.000075:.2f}")


@app.event("app_mention")
def command_handler(body, context):
    try:
        channel_id = body["event"]["channel"]
        thread_ts = body["event"].get("thread_ts", body["event"]["ts"])
        bot_user_id = context["bot_user_id"]
        slack_resp = app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=WAIT_MESSAGE
        )
        reply_message_ts = slack_resp["message"]["ts"]
        
        # If there's no event for this thread yet, create one and set it
        if thread_ts not in events:
            events[thread_ts] = Event()
            events[thread_ts].set()

        # Wait until the event is set, indicating that the previous message is done processing
        events[thread_ts].wait()

        # Clear the event to indicate that a new message is being processed
        events[thread_ts].clear()

        conversation_history = get_conversation_history(channel_id, thread_ts)
        messages = process_conversation_history(conversation_history, bot_user_id)
        num_tokens = num_tokens_from_messages(messages)
        print(f"Number of tokens: {num_tokens}")
        if TARGET_MODEL == GPT_MODEL:
            stream_openai_request(messages, channel_id, reply_message_ts)
        else:
            stream_anthropic_request(messages, channel_id, reply_message_ts)
    except Exception as e:
        print(f"Error: {e}")
        try:
            app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"I can't provide a response. Encountered an error:\n`\n{e}\n`")
        except Exception as e:
            print(f"Error: {e}")
    finally:
        # Set the event to indicate that this message is done processing
        if "thread_ts" in locals() and thread_ts in events:
            events[thread_ts].set()


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
