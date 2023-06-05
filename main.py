import os
import openai
from threading import Event
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from utils import (N_CHUNKS_TO_CONCAT_BEFORE_UPDATING, OPENAI_API_KEY,
                   MAX_TOKENS, SLACK_APP_TOKEN, SLACK_BOT_TOKEN, WAIT_MESSAGE,
                   num_tokens_from_messages, process_conversation_history,
                   update_chat)

app = App(token=SLACK_BOT_TOKEN)
openai.api_key = OPENAI_API_KEY

# Debug mode
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

MAX_TOKEN_MESSAGE = f"Apologies, but the maximum number of tokens ({format(MAX_TOKENS, ',')}) for this thread has been reached. Please start a new thread to continue discussing this topic."

# A dictionary to store an event for each channel
events = {}


def get_conversation_history(channel_id, thread_ts):
    return app.client.conversations_replies(
        channel=channel_id,
        ts=thread_ts,
        inclusive=True
    )


def make_openai_request(messages, channel_id, reply_message_ts):
    openai_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        stream=True
    )
    response_text = ""
    ii = 0
    for chunk in openai_response:
        if chunk.choices[0].delta.get("content"):
            ii = ii + 1
            response_text += chunk.choices[0].delta.content
            if ii > N_CHUNKS_TO_CONCAT_BEFORE_UPDATING:
                update_chat(app, channel_id, reply_message_ts, response_text)
                ii = 0
        elif chunk.choices[0].finish_reason == "stop":
            update_chat(app, channel_id, reply_message_ts, response_text)
        elif chunk.choices[0].finish_reason == "length":
            update_chat(app, channel_id, reply_message_ts, response_text + "...\n\n" + MAX_TOKEN_MESSAGE)


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
        make_openai_request(messages, channel_id, reply_message_ts)
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
