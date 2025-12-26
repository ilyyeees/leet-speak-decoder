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

# Search terms to find dense leetspeak threads
SEARCH_TERMS = [
    'leetspeak',
    '1337',
    '1337 speak',
    'h4x0r',
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
    Check if text is REAL leetspeak (must have numbers/symbols as letters).
    Slang/abbreviations alone (idk, lol) are NOT enough.
    """
    text_lower = text.lower()

    # STRICT FILTER: Must have numbers mixed into words
    # Matches: h3ll0, w0rld, l8r, n00b, pwn3d
    # Does not match: "2024", "100", "idk", "lol"
    if re.search(r'[a-z]+[0-9]+[a-z]*|[a-z]*[0-9]+[a-z]+', text_lower):
        return True

    # Also accept heavy symbol usage if it looks like masking
    # Matches: @, $, 1, 3, 4, 5, 7, 0 used inside words
    if re.search(r'\b\w*[@$34570]\w*\b', text):
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

    if not comment_divs:
        print(f"    [DEBUG] No 'div.thing' found! Page Title: {soup.title.string if soup.title else 'No Title'}")
        print(f"    [DEBUG] HTML Length: {len(html)}")
        if "Too Many Requests" in html:
            print("    [DEBUG] BLOCKED: Too Many Requests")
        return []

    print(f"    [DEBUG] Found {len(comment_divs)} 'thing' divs")

    skipped_not_comment = 0
    skipped_no_body = 0
    skipped_validation = 0

    for div in comment_divs:
        # Check if it is a comment (using data-type attribute is more reliable)
        if div.get('data-type') != 'comment':
            skipped_not_comment += 1
            # Debug: print what kind of thing it is
            # print(f"    [DEBUG] Skipped thing with data-type: {div.get('data-type')}")
            continue

        # Extract comment text from the entry (div.entry > form > div.usertext-body > div.md)
        # On old.reddit, sometimes structure is nested differently
        comment_body = div.find('div', class_='usertext-body')
        if not comment_body:
            # Fallback
            comment_body = div.find('div', class_='md')

        if not comment_body:
            skipped_no_body += 1
            continue

        text = comment_body.get_text().strip()

        # Validate
        if is_valid_comment(text):
            # ... score extraction ...
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
        else:
            skipped_validation += 1
            # Print sample of rejected text (first one only)
            if skipped_validation == 1:
                print(f"    [DEBUG] Rejected sample: {text[:50]}...")

    print(f"    [DEBUG] Skipped: {skipped_not_comment} not_t1, {skipped_no_body} no_body, {skipped_validation} invalid")
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

    # Build list of URLs to scrape
    urls_to_scrape = []

    # 1. Search results (high priority)
    for term in SEARCH_TERMS:
        term_encoded = term.replace(' ', '+')
        urls_to_scrape.append({
            'type': 'search',
            'name': f"Search: {term}",
            'url': f"https://old.reddit.com/search.json?q={term_encoded}&sort=relevance&limit=100"
        })

    # 2. Subreddits (hot posts)
    for sub in SUBREDDITS:
        urls_to_scrape.append({
            'type': 'subreddit',
            'name': f"r/{sub}",
            'url': f"https://old.reddit.com/r/{sub}/hot.json?limit=100"
        })

    with open(output_file, 'a') as f:
        for source in urls_to_scrape:
            if total_scraped >= TARGET_COMMENTS:
                break

            print(f"[{source['name']}] Scraping...")

            try:
                # Fetch listing page
                html = scraper.get_page(source['url'])
                if not html:
                    print(f"  ⚠ Failed to fetch listing")
                    continue

                # Parse JSON
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
                    # Construct permalink from ID to ensure we go to old.reddit
                    post_url = f"https://old.reddit.com/comments/{post_id}"

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
                            'subreddit': post['data'].get('subreddit', 'unknown'),
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
