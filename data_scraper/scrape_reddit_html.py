#!/usr/bin/env python3
"""
Reddit HTML Scraper (No API Required)
======================================
Scrapes Reddit comments using HTML parsing - no API credentials needed!

Setup:
    pip install -r requirements.txt
    python scrape_reddit_html.py

Output:
    - raw_comments.jsonl (one comment per line)
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime
from pathlib import Path
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

# Target subreddits (gaming, casual, teen culture)
SUBREDDITS = [
    'teenagers',
    'gaming',
    'pcmasterrace',
    'dankmemes',
    'memes',
    'Minecraft',
    'leagueoflegends',
    'valorant',
    'GlobalOffensive',
    'discordapp',
]

# Scraping parameters
TARGET_COMMENTS = 50000  # How many comments to scrape
MIN_COMMENT_LENGTH = 10  # Minimum characters
MAX_COMMENT_LENGTH = 300  # Maximum characters
POSTS_PER_SUBREDDIT = 50  # How many posts to check per subreddit
DELAY_BETWEEN_REQUESTS = (2, 5)  # Random delay (min, max) in seconds


# ============================================================================
# HTTP CLIENT
# ============================================================================

class RedditScraper:
    def __init__(self):
        self.session = requests.Session()
        # Randomize user agent to avoid detection
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        ]
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    def get_page(self, url, retries=3):
        """
        Fetch a page with retry logic.
        """
        for attempt in range(retries):
            try:
                # Random delay to avoid rate limiting
                delay = random.uniform(*DELAY_BETWEEN_REQUESTS)
                time.sleep(delay)

                response = self.session.get(url, timeout=10)

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    print(f"  ⚠ Rate limited, waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"  ⚠ HTTP {response.status_code}, retrying...")

            except Exception as e:
                print(f"  ⚠ Request failed: {e}, retrying...")

            # Exponential backoff
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

        return None


# ============================================================================
# LEETSPEAK DETECTION
# ============================================================================

def has_leetspeak_indicators(text):
    """
    Check if text likely contains leetspeak/abbreviations.
    """
    text_lower = text.lower()

    # Pattern 1: Numbers used as letters (1, 3, 4, 7, 0)
    if re.search(r'\b\w*[1347@$]\w*\b', text):
        return True

    # Pattern 2: Common abbreviations
    abbrevs = [
        'idk', 'omg', 'lol', 'brb', 'btw', 'tbh', 'imo', 'afaik',
        'nvm', 'rn', 'fr', 'np', 'g2g', 'w8', 'l8r', 'm8',
        'thx', 'pls', 'plz', 'u', 'ur', 'y', 'gg', 'wp'
    ]
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    if any(abbr in text_words for abbr in abbrevs):
        return True

    # Pattern 3: Repeated characters
    if re.search(r'(\d)\1{2,}|([a-z])\2{3,}', text_lower):
        return True

    return False


def is_valid_comment(text):
    """
    Filter out junk comments.
    """
    # Length filter
    if len(text) < MIN_COMMENT_LENGTH or len(text) > MAX_COMMENT_LENGTH:
        return False

    # Filter deleted/removed
    if text in ['[deleted]', '[removed]', '[deleted by user]']:
        return False

    # Filter URLs
    if 'http://' in text or 'https://' in text:
        return False

    # Filter if mostly markdown/code
    if text.count('`') > 2 or text.count('*') > 4:
        return False

    # Must have leetspeak indicators
    if not has_leetspeak_indicators(text):
        return False

    return True


# ============================================================================
# HTML PARSING
# ============================================================================

def parse_comments_from_post(html):
    """
    Extract comments from a Reddit post HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    comments = []

    # Find all comment divs
    comment_divs = soup.find_all('div', class_='thing')

    for div in comment_divs:
        # Skip if it's a post, not a comment
        if 't1_' not in div.get('class', [''])[0]:
            continue

        # Extract comment text
        comment_body = div.find('div', class_='md')
        if not comment_body:
            continue

        text = comment_body.get_text().strip()

        # Validate
        if is_valid_comment(text):
            # Extract score
            score_elem = div.find('span', class_='score unvoted')
            score = 0
            if score_elem:
                score_text = score_elem.get_text()
                try:
                    score = int(re.findall(r'\d+', score_text)[0])
                except:
                    pass

            comments.append({
                'text': text,
                'score': score,
            })

    return comments


# ============================================================================
# SCRAPER
# ============================================================================

def scrape_reddit_html():
    """
    Main scraping function using HTML parsing.
    """
    print("=" * 80)
    print("REDDIT HTML SCRAPER (No API Required)")
    print("=" * 80)

    scraper = RedditScraper()
    output_file = Path("raw_comments.jsonl")

    # Load existing comments if resuming
    existing_count = 0
    seen_texts = set()
    if output_file.exists():
        print(f"\n⚠ Found existing {output_file}, resuming...")
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                seen_texts.add(data['text'])
                existing_count += 1
        print(f"✓ Loaded {existing_count} existing comments")

    total_scraped = existing_count
    total_checked = 0

    print(f"\nTarget: {TARGET_COMMENTS:,} comments")
    print(f"Subreddits: {', '.join(SUBREDDITS)}")
    print(f"Using: old.reddit.com (easier to parse)")
    print(f"\nStarting scrape...\n")

    with open(output_file, 'a') as f:
        for subreddit_name in SUBREDDITS:
            if total_scraped >= TARGET_COMMENTS:
                break

            print(f"[{subreddit_name}] Scraping...")

            # Get top posts from subreddit
            subreddit_url = f"https://old.reddit.com/r/{subreddit_name}/hot.json?limit=100"

            try:
                # Fetch subreddit page
                html = scraper.get_page(subreddit_url)
                if not html:
                    print(f"  ⚠ Failed to fetch subreddit")
                    continue

                # Parse JSON (old.reddit.com with .json gives JSON)
                try:
                    data = json.loads(html)
                    posts = data['data']['children']
                except:
                    print(f"  ⚠ Failed to parse JSON")
                    continue

                # Process each post
                posts_checked = 0
                for post in posts[:POSTS_PER_SUBREDDIT]:
                    if total_scraped >= TARGET_COMMENTS:
                        break

                    post_id = post['data']['id']
                    post_url = f"https://old.reddit.com/r/{subreddit_name}/comments/{post_id}"

                    # Fetch post comments
                    post_html = scraper.get_page(post_url)
                    if not post_html:
                        continue

                    # Parse comments
                    comments = parse_comments_from_post(post_html)
                    total_checked += len(comments)

                    valid_count = 0
                    for comment in comments:
                        # Skip duplicates
                        if comment['text'] in seen_texts:
                            continue

                        # Save comment
                        data = {
                            'subreddit': subreddit_name,
                            'text': comment['text'],
                            'score': comment['score'],
                            'timestamp': datetime.now().isoformat()
                        }
                        f.write(json.dumps(data) + '\n')
                        f.flush()

                        seen_texts.add(comment['text'])
                        total_scraped += 1
                        valid_count += 1

                        # Progress update
                        if total_scraped % 100 == 0:
                            print(f"  Progress: {total_scraped:,}/{TARGET_COMMENTS:,} "
                                  f"({100*total_scraped/TARGET_COMMENTS:.1f}%) "
                                  f"[checked: {total_checked:,}]")

                        if total_scraped >= TARGET_COMMENTS:
                            break

                    posts_checked += 1
                    print(f"  [Post {posts_checked}/{POSTS_PER_SUBREDDIT}] "
                          f"Parsed: {len(comments)}, Kept: {valid_count} "
                          f"(Total Saved: {total_scraped})")


            except Exception as e:
                print(f"  ⚠ Error: {e}")
                continue

    # Final stats
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETE")
    print("=" * 80)
    print(f"Total comments scraped: {total_scraped:,}")
    print(f"Total comments checked: {total_checked:,}")
    print(f"Hit rate: {100*total_scraped/max(total_checked, 1):.2f}%")
    print(f"\nOutput: {output_file.absolute()}")
    print("\nNext step: Translate these with your LLM to create training pairs!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    scrape_reddit_html()
