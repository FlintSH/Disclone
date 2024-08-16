<p align="center">
  <img src="https://cdn.fl1nt.dev/GAGE1/ReDIZUKO73.jpg/raw" alt="Disclone Banner">
</p>
<p align="center"><small><i>Banner generated with FLUX.1-Pro</i></small></p>

<h1 align="center">Disclone - Fine-tune OpenAI models with your Discord chat history</h1>

Disclone is a little script I made that helps you create a fine-tuning dataset for OpenAI models based on Discord chat logs.

It allows you to generate an AI version of a specific Discord user by processing their chat messages into a JSONL file that can be plugged into OpenAI's fine-tuning API.

## What it does

1. Processes a CSV file exported from Discord using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).
2. Creates training data from the chat logs, focusing on a specific user.
3. Applies content moderation to filter out inappropriate content - this is a must as OpenAI's fine-tuning API will reject datasets with a high density of inappropriate messages.
 - Content is moderated using the same API that OpenAI uses to moderate content on their platform, so it essentially guarantees that your dataset will be accepted.
4. Generates a JSONL file that can be simply plugged into OpenAI's fine-tuning API, and have a new model tuned on your data.

## Prerequisites

- Python 3.7 or higher
- An OpenAI API key
- A CSV file of your desired Discord chat exported using [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter).

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/disclone.git
   cd disclone
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```
 - Note: While this uses an OpenAI API key, this is just for the moderation API, and is not used to generate any content. OpenAI's moderation API is free to use, and your API key will not be charged.

## Usage

1. Export your Discord chat logs:
   - Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to export a chat as a CSV file.

2. Run the Disclone script:
   ```
   python main.py
   ```

3. Follow the prompts:
   - Enter the path to your CSV file.
   - Specify the Discord username of the target user you want to clone.
   - Provide a system prompt to guide the AI's behavior (optional but recommended).
   - Enter a start date if you want to process messages from a specific date onwards (optional).
   - Set a limit on the number of conversations to include (optional).

4. Wait for the script to process the data. It will:
   - Parse the CSV file
   - Create conversations
   - Moderate content
   - Generate a JSONL file for fine-tuning

5. Once complete, you'll find a JSONL file named `<username>_training_data.jsonl` in the same directory.

6. Use this JSONL file to fine-tune an OpenAI model following the [OpenAI fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning).

## Notes

- The script uses OpenAI's content moderation API to filter out inappropriate content. This helps ensure the training data is suitable for fine-tuning.
- The generated JSONL file follows OpenAI's required format for fine-tuning datasets.
- Be mindful of Discord's terms of service and privacy considerations when exporting and using chat data - use at your own risk.

## Troubleshooting

- If you encounter rate limiting issues, you may need to adjust the `RateLimiter` parameters in the script based on your OpenAI API tier.
- Ensure your CSV file is properly formatted and contains the required columns (ID, Author, Date, Content, Attachments, Reactions).
