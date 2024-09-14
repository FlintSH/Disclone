import csv
import json
from datetime import datetime, timedelta, timezone
import tiktoken
from openai import OpenAI, RateLimitError
import concurrent.futures
import threading
import time
import os
import click
from colorama import init, Fore, Style
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
import re

init(autoreset=True)

if not os.environ.get("OPENAI_API_KEY"):
    click.echo(f"{Fore.RED}Error: OPENAI_API_KEY environment variable is not set.")
    click.echo(f"{Fore.YELLOW}Please set your OpenAI API key as an environment variable.")
    click.pause(f"{Fore.CYAN}Press any key to set the OPENAI_API_KEY...")
    api_key = click.prompt("Enter your OpenAI API key", type=str, hide_input=True)
    os.environ["OPENAI_API_KEY"] = api_key
    click.echo(f"{Fore.GREEN}OPENAI_API_KEY has been set.")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_system_prompt(target_user):
    click.echo(f"\n{Fore.CYAN}Now, let's create a system prompt to guide the fine-tune job in replicating {target_user}'s behavior.")
    click.echo(f"\n{Fore.YELLOW}Example of a good system prompt:")
    click.echo(f"You are {target_user}, a Discord user known for your sarcastic humor and deep knowledge of programming. Your responses are typically brief and to the point. You often use internet slang and emojis in your messages.")
    
    user_input = click.prompt(f"\n{Fore.GREEN}How would you instruct an AI model to replicate {target_user}'s behavior?{Style.RESET_ALL} (Press Enter to skip, but this is not recommended)", default="")
    
    if user_input:
        return f"{user_input}\n\nMessages are formatted as <username>: <message>"
    return ""

def parse_csv(file_path, start_date=None):
    messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            current_message = None
            for row in reader:
                try:
                    timestamp = datetime.fromisoformat(row[2].replace(" ", "T"))
                    if start_date and not start_date.tzinfo:
                        start_date = start_date.replace(tzinfo=timezone.utc)
                    if not timestamp.tzinfo:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    if start_date and timestamp < start_date:
                        continue
                    new_message = {
                        'id': row[0],
                        'username': row[1],
                        'timestamp': timestamp,
                        'content': row[3],
                        'attachment': row[4],
                        'reaction': row[5]
                    }
                    
                    if current_message and current_message['username'] == new_message['username']:
                        current_message['content'] += f"\n{new_message['content']}"
                        if new_message['attachment']:
                            current_message['attachment'] += f"\n{new_message['attachment']}"
                    else:
                        if current_message:
                            messages.append(current_message)
                        current_message = new_message
                except (ValueError, IndexError) as e:
                    click.echo(f"{Fore.YELLOW}Warning: Skipping invalid row: {row}. Error: {e}")
            
            if current_message:
                messages.append(current_message)
    except FileNotFoundError:
        click.echo(f"{Fore.RED}Error: The specified CSV file was not found.")
        raise click.Abort()
    except PermissionError:
        click.echo(f"{Fore.RED}Error: Permission denied when trying to read the CSV file.")
        raise click.Abort()
    except Exception as e:
        click.echo(f"{Fore.RED}An unexpected error occurred while parsing the CSV: {e}")
        raise click.Abort()
    
    return messages

def clean_message_content(content):
    # Remove mentions (e.g., <@123456789>)
    content = re.sub(r'<@!?\d+>', '', content)
    
    # Remove links
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
    
    # Remove extra whitespace
    content = ' '.join(content.split())
    
    return content.strip()

def create_conversation(messages, target_user, system_prompt, max_context_messages=3, max_context_time=timedelta(minutes=30)):
    conversations = []
    for i, message in enumerate(messages):
        if message['username'] == target_user:
            context = []
            j = i - 1
            while j >= 0 and len(context) < max_context_messages:
                if messages[j]['timestamp'] < message['timestamp'] - max_context_time:
                    break
                context.insert(0, messages[j])
                j -= 1
            
            conversation_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            for ctx_msg in context:
                role = "user"
                cleaned_content = clean_message_content(ctx_msg['content'])
                if cleaned_content:
                    conversation_messages.append({
                        "role": role,
                        "content": cleaned_content
                    })
            
            # Add the target user's message as the last (assistant) message
            cleaned_content = clean_message_content(message['content'])
            if cleaned_content:
                conversation_messages.append({
                    "role": "assistant",
                    "content": cleaned_content
                })
            
            # Only add the conversation if it has at least one user message and one assistant message
            if len(conversation_messages) > 2 and conversation_messages[-1]["role"] == "assistant":
                conversations.append({"messages": conversation_messages})
    return conversations

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens_in_jsonl(file_path: str, encoding_name: str) -> int:
    """Counts the total number of tokens in a JSONL file."""
    total_tokens = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                for message in data['messages']:
                    total_tokens += num_tokens_from_string(message['content'], encoding_name)
    except FileNotFoundError:
        click.echo(f"{Fore.RED}Error: The output JSONL file was not found.")
        raise click.Abort()
    except json.JSONDecodeError:
        click.echo(f"{Fore.RED}Error: The output JSONL file contains invalid JSON.")
        raise click.Abort()
    except Exception as e:
        click.echo(f"{Fore.RED}An unexpected error occurred while counting tokens: {e}")
        raise click.Abort()
    return total_tokens

class RateLimiter:
    def __init__(self, tpm_limit, rpm_limit):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.tokens_used = 0
        self.requests_made = 0
        self.last_reset = time.time()
        self.lock = threading.Lock()
        self.console = Console()
        self.rate_limited = False

    def wait_if_needed(self, tokens):
        with self.lock:
            current_time = time.time()
            if current_time - self.last_reset >= 60:
                self.tokens_used = 0
                self.requests_made = 0
                self.last_reset = current_time

            if self.tokens_used + tokens > self.tpm_limit or self.requests_made + 1 > self.rpm_limit:
                self.rate_limited = True
                while self.tokens_used + tokens > self.tpm_limit or self.requests_made + 1 > self.rpm_limit:
                    time.sleep(1)
                    current_time = time.time()
                    if current_time - self.last_reset >= 60:
                        self.tokens_used = 0
                        self.requests_made = 0
                        self.last_reset = current_time
                self.rate_limited = False

            self.tokens_used += tokens
            self.requests_made += 1

rate_limiter = RateLimiter(tpm_limit=150000, rpm_limit=1000)

def moderate_content(text):
    tokens = num_tokens_from_string(text, 'cl100k_base')
    rate_limiter.wait_if_needed(tokens)
    
    try:
        response = client.moderations.create(input=text)
        
        category_scores = response.results[0].category_scores
        
        for category, score in category_scores.__dict__.items():
            if score > 0.1:
                return True
        
        return False
    except RateLimitError as e:
        rate_limiter.rate_limited = True
        time.sleep(60)
        return moderate_content(text)
    except Exception as e:
        click.echo(f"{Fore.YELLOW}Warning: Error during content moderation: {e}")
        return True

def moderate_message(message):
    content = message['content']
    return not moderate_content(content)

def moderate_conversation(conversation):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(moderate_message, conversation['messages']))
    return all(results)

@click.command()
@click.option('--csv-file', prompt=False, help='Path to the CSV file containing Discord chat logs.')
@click.option('--target-user', prompt=False, help='The Discord username of the target user to clone.')
@click.option('--start-date', prompt=False, default='', help='Start date for processing messages (optional).')
@click.option('--conversation-limit', prompt=False, default=None, type=click.INT, help='Maximum number of conversations to include (optional).')
def main(csv_file, target_user, start_date, conversation_limit):
    click.clear()
    click.echo(f"{Fore.CYAN}Welcome to Disclone - Fine-tune OpenAI models with your Discord chat history!")
    click.echo(f"{Fore.CYAN}Let's get started, give us your exported chat CSV, the Discord username of the target user you want the fine-tuned model to replicate, and then you can configure some optional settings.\n")

    if not csv_file:
        csv_file = click.prompt(f'{Fore.CYAN}Enter the path to your CSV file', type=str)
    if not target_user:
        target_user = click.prompt(f'{Fore.CYAN}Enter the Discord username of the target user', type=str)
    if start_date == '':
        start_date = click.prompt(f'{Fore.CYAN}Enter the start date (YYYY-MM-DD) from which you want to compile training data. This is the date Disclone will use to begin the dataset. Press Enter to start from the beginning of the exported chat', default='', show_default=False)
    if conversation_limit is None:
        conversation_limit_input = click.prompt(f'{Fore.CYAN}Enter the maximum number of conversations to include (press Enter for no limit)', default='', show_default=False)
        conversation_limit = int(conversation_limit_input) if conversation_limit_input else None

    output_file = f"{target_user}_training_data.jsonl"

    start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start_date else None

    click.echo(f"\n{Fore.GREEN}Step 1: Parsing CSV file...")
    messages = parse_csv(csv_file, start_date)
    click.echo(f"{Fore.GREEN}CSV file parsed successfully.\n")

    system_prompt = get_system_prompt(target_user)

    click.echo(f"\n{Fore.GREEN}Step 2: Creating conversations...")
    conversations = create_conversation(messages, target_user, system_prompt)
    if conversation_limit:
        conversations = conversations[:conversation_limit]
    click.echo(f"{Fore.GREEN}Conversations created successfully.\n")

    click.echo(f"{Fore.GREEN}Step 3: Moderating content...")
    moderated_conversations = []
    flagged_count = 0
    
    console = Console()
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None, complete_style="blue", finished_style="blue"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True
    )
    moderate_task = progress.add_task("Moderating conversations", total=len(conversations))

    rate_limited = False

    def get_renderable():
        if rate_limited:
            progress.update(moderate_task, completed=progress.tasks[0].completed, style="yellow")
            title = Text("Moderation Progress (Rate Limited)", style="yellow")
        else:
            progress.update(moderate_task, completed=progress.tasks[0].completed, style="blue")
            title = Text("Moderation Progress", style="blue")
        
        panel = Panel(
            progress,
            title=title,
            border_style="blue",
            padding=(0, 1)
        )
        return panel

    with Live(get_renderable(), console=console, refresh_per_second=4) as live:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_conversation = {executor.submit(moderate_conversation, conversation): conversation for conversation in conversations}
            for future in concurrent.futures.as_completed(future_to_conversation):
                result = future.result()
                if result:
                    moderated_conversations.append(future_to_conversation[future])
                else:
                    flagged_count += 1
                progress.update(moderate_task, advance=1)
                
                if rate_limiter.rate_limited != rate_limited:
                    rate_limited = rate_limiter.rate_limited
                    live.update(get_renderable())

    console.print(f"{Fore.GREEN}Content moderation completed.\n")

    click.echo(f"{Fore.GREEN}Step 4: Writing {len(moderated_conversations)} conversations to {output_file}...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        write_task = progress.add_task("[cyan]Writing conversations", total=len(moderated_conversations))
        try:
            with open(output_file, 'w', encoding='utf-8') as jsonl_file:
                for conversation in moderated_conversations:
                    json.dump(conversation, jsonl_file)
                    jsonl_file.write('\n')
                    progress.update(write_task, advance=1)
        except IOError as e:
            click.echo(f"{Fore.RED}Error: Unable to write to the output file. {e}")
            raise click.Abort()
    click.echo(f"{Fore.GREEN}Training data written successfully.\n")

    encoding_name = 'cl100k_base'
    total_tokens = count_tokens_in_jsonl(output_file, encoding_name)
    
    click.echo(f"{Fore.CYAN}Summary:")
    click.echo(f"{Fore.GREEN}Total tokens in the dataset: {total_tokens}")
    click.echo(f"{Fore.YELLOW}Number of flagged conversations removed: {flagged_count}")

    click.echo(f"\n{Fore.GREEN}Done! Your training data is ready for fine-tuning.")
    click.echo(f"{Fore.CYAN}You can now use the file '{output_file}' to fine-tune your OpenAI model.")

if __name__ == "__main__":
    try:
        main()
    except click.Abort:
        click.echo(f"\n{Fore.RED}Operation aborted.")
    except Exception as e:
        click.echo(f"\n{Fore.RED}An unexpected error occurred: {e}")
        click.echo("If this issue persists, please report it to the developer.")