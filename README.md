# UpGradeBot
A Slack chatbot designed to assist with questions and guidance about UpGrade, the open-source A/B testing platform for educational software.

## Features
- Integrate with OpenAI's GPT or Anthropic's Claude to answer questions
- Maintain conversation context in a threaded format
- Socket mode integration with Slack

## Installation
1. Clone this repository:

```bash
git clone https://github.com/CarnegieLearningWeb/upgrade-bot.git
cd upgrade-bot
```
2. Install the required packages:

```bash
pip install -r requirements.txt
```
3. Create a .env file in the root directory of the project and add your Slack and OpenAI API keys:

```bash
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_APP_TOKEN=your_slack_app_token
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
TARGET_LLM=gpt_or_claude
PROMPT_TYPE=full_or_design_or_reference
```
See below how to get those.

## Configuring Permissions in Slack
Before you can run the UpGradeBot, you need to configure the appropriate permissions. Follow these steps to set up the necessary permissions:

1. Create [Slack App](https://api.slack.com/authentication/basics)
2. Go to your [Slack API Dashboard](https://api.slack.com/apps) and click on the app you created for this bot.
3. In the left sidebar, click on "OAuth & Permissions".
4. In the "Scopes" section, you will find two types of scopes: "Bot Token Scopes" and "User Token Scopes". Add the following scopes under "Bot Token Scopes":
   - `app_mentions:read`: Allows the bot to read mention events.
   - `chat:write`: Allows the bot to send messages.
   - `channels:history`: View messages and other content in public channels the bot has been added to.
   - `groups:history`: View messages and other content in private channels the bot has been added to.
   - `mpim:history`: View messages and other content in group direct messages the bot has been added to.
   - `im:history`: View messages and other content in direct messages the bot has been added to.
5. Scroll up to the "OAuth Tokens for Your Workspace" and click "Install App To Workspace" button. This will generate the `SLACK_BOT_TOKEN`.
6. In the left sidebar, click on "Socket Mode" and enable it. You'll be prompted to "Generate an app-level token to enable Socket Mode". Generate a token named `SLACK_APP_TOKEN` and add the `connections:write` scope.
7. In the "Features affected" section of "Socket Mode" page, click "Event Subscriptions" and toggle "Enable Events" to "On". Add `app_mention` event with the `app_mentions:read` scope in the "Subscribe to bot events" section below the toggle.

## Usage
1. Start the bot:

```
python main.py
```
2. Invite the bot to your desired Slack channel.
3. Mention the bot in a message and ask a question (including any URLs). The bot will respond with an answer, taking into account any extracted content from URLs.

## Based on
This code is based on [Slack GPT Bot](https://github.com/alex000kim/slack-gpt-bot).
