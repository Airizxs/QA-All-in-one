"""
This script checks the spelling of a given URL's content using OpenAI's GPT model.
It accepts a URL as a command-line argument, or prompts the user to enter one.
The results are logged to logs/spellcheck.csv.
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import argparse
import csv
import datetime
from openai import OpenAI
from colorama import Fore, Style, init

init(autoreset=True)
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-mini"
LOG_FILE = os.path.join("logs", "spellcheck.csv")


def setup_logging():
    """Creates the logs directory if it doesn't exist and initializes the log file."""
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create log file if it doesn't exist and write header
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as csvfile:
             writer = csv.writer(csvfile)
             writer.writerow(["Timestamp", "URL", "Misspelled Word", "Count"])
             


def extract_visible_text(url):
    try:
        res = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        })
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text()).strip()
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Failed to fetch content: {e}")
        return ""

def chunk_text(text, max_tokens=1500):
    words = text.split()
    chunks, chunk = [], []
    for word in words:
        chunk.append(word)
        if len(chunk) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def check_spelling(chunk):
    prompt = (
        "Review the following webpage content and return only actual US English spelling mistakes. "
        "Output a markdown table of confirmed errors with this format:\n\n"
        "**Misspelled Word | Correct Spelling**\n"
        "Do not include made-up examples, synonyms, or grammar suggestions.\n\n"
        f"Content: {chunk}"
    )
    
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"{Fore.RED}[ERROR] OpenAI request failed: {e}")
        return ""

def log_results(url, results):
    """Logs the results to the CSV log file with counts."""
    
    timestamp = datetime.datetime.now().isoformat()

    # Count occurrences of each misspelled word
    word_counts = {}
    for word in results:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Check if there are any actual misspellings
        if not word_counts:
           writer.writerow([timestamp, url, "No spelling errors found.", ""])
           return
           
        # Log unique misspelled words and their counts
        for word, count in word_counts.items():
             writer.writerow([timestamp, url, word, count])





def main():
    """Main function to run the spellcheck script."""
    parser = argparse.ArgumentParser(
        description="Spellchecks a given URL's content using OpenAI's GPT model."
    )
    parser.add_argument(
        "url", nargs="?", type=str, help="The URL to spellcheck"
    )
    args = parser.parse_args()

    if args.url:
        url = args.url.strip()
    else:
        url = input("Enter full URL to spellcheck: ").strip()

    print(f"\n{Fore.CYAN + Style.BRIGHT}=== SPELLCHECKING PAGE ===\n")
    print(f"{Fore.YELLOW}URL: {url}\n")

    # Extract text from URL
    text = extract_visible_text(url)
    if not text:
        print(f"{Fore.RED}No content extracted.")
        exit()

   # Split text into chunks for processing
    chunks = chunk_text(text)
    print(f"{Fore.BLUE}Split into {len(chunks)} chunk(s). Sending to OpenAI...\n")

    all_potential_issues = []
    for i, chunk in enumerate(chunks):
        print(f"{Fore.CYAN}→ Checking chunk {i+1}/{len(chunks)}...          ", end="\r")
        result = check_spelling(chunk)
        if result:
            
            for line in result.split("\n"):
                if " | " in line:
                    misspelled_word = line.split(" | ")[0].strip()
                    all_potential_issues.append(misspelled_word)
        time.sleep(1)
    print("                                                        ",end="\r")

    confirmed_misspellings = []
    if all_potential_issues:
        print(f"{Fore.BLUE}Validating potential misspellings...")
        for word in all_potential_issues:
            validation_prompt = (
                f"Is '{word}' a spelling mistake? Respond with 'yes' or 'no'."
            )
            try:
                validation_res = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": validation_prompt}],
                    temperature=0
                )
                if "yes" in validation_res.choices[0].message.content.lower():
                    confirmed_misspellings.append(word)
            except Exception as e:
                print(f"{Fore.RED}[ERROR] OpenAI validation failed for '{word}': {e}")


    if confirmed_misspellings:
        print(f"\n{Fore.GREEN + Style.BRIGHT}=== CONFIRMED SPELLING ISSUES FOUND ===\n")
        for issue in sorted(confirmed_misspellings):
            print(f"{Fore.YELLOW}{issue}")
    else:
         print(f"{Fore.GREEN}No spelling errors found.")
    log_results(url, confirmed_misspellings)


if __name__ == "__main__":
    setup_logging()
    main()


