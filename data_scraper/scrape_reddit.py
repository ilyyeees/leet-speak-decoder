#!/usr/bin/env python3
"""
Reddit Leetspeak Scraper
========================
Scrapes Reddit comments from gaming/casual subreddits to find leetspeak examples.

Setup:
    1. Get Reddit API credentials: https://www.reddit.com/prefs/apps
    2. Create new app -> script -> note client_id and client_secret
    3. pip install -r requirements.txt
    4. python scrape_reddit.py

Output:
    - raw_comments.jsonl (one comment per line)
"""

import praw
import json
import re
import time
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

# Reddit API credentials (UPDATE THESE)
REDDIT_CLIENT_ID = "YOUR_CLIENT_ID_HERE"
REDDIT_CLIENT_SECRET = "YOUR_CLIENT_SECRET_HERE"
REDDIT_USER_AGENT = "leetspeak_scraper/1.0"

# Target subreddits (gaming, casual, teen culture)
SUBREDDITS = [
    'teenagers',
    'gaming',
    'pcmasterrace',
    'dankmemes',
    'me_irl',
    'memes',
    'Minecraft',
    'leagueoflegends',
    'Overwatch',
    'discordapp',
]

# Scraping parameters
TARGET_COMMENTS = 50000  # How many comments to scrape
MIN_COMMENT_LENGTH = 10  # Minimum characters
MAX_COMMENT_LENGTH = 300  # Maximum characters
BATCH_SIZE = 1000  # Save progress every N comments


# ============================================================================
# LEETSPEAK DETECTION
# ============================================================================

def has_leetspeak_indicators(text):
    """
    Check if text likely contains leetspeak/abbreviations.
    """
    # Convert to lowercase for checking
    text_lower = text.lower()

    # Pattern 1: Numbers used as letters (1, 3, 4, 7, 0)
    if re.search(r'\b\w*[1347@$]\w*\b', text):
        return True

    # Pattern 2: Common abbreviations
    abbrevs = [
        'idk', 'omg', 'lol', 'brb', 'btw', 'tbh', 'imo', 'afaik',
        'nvm', 'rn', 'fr', 'np', 'g2g', 'w8', 'l8r', 'm8',
        'thx', 'pls', 'plz', 'u', 'ur', 'y'
    ]
    if any(f' {abbr} ' in f' {text_lower} ' or
           f' {abbr},' in f' {text_lower} ' or
           f' {abbr}.' in f' {text_lower} ' or
           f' {abbr}?' in f' {text_lower} '
           for abbr in abbrevs):
        return True

    # Pattern 3: Repeated characters (leet emphasis)
    if re.search(r'(\d)\1{2,}|([a-z])\2{3,}', text_lower):
        return True

    return False


def is_valid_comment(comment):
    """
    Filter out junk comments.
    """
    text = comment.body.strip()

    # Length filter
    if len(text) < MIN_COMMENT_LENGTH or len(text) > MAX_COMMENT_LENGTH:
        return False

    # Filter deleted/removed
    if text in ['[deleted]', '[removed]']:
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
# SCRAPER
# ============================================================================

def scrape_reddit_comments():
    """
    Main scraping function.
    """
    print("=" * 80)
    print("REDDIT LEETSPEAK SCRAPER")
    print("=" * 80)

    # Initialize Reddit API
    print("\nConnecting to Reddit API...")
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    print(f"✓ Connected as: {reddit.user.me() if reddit.read_only else 'Read-Only Mode'}")

    # Output file
    output_file = Path("raw_comments.jsonl")

    # Load existing comments if resuming
    existing_count = 0
    seen_ids = set()
    if output_file.exists():
        print(f"\n⚠ Found existing {output_file}, resuming...")
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                seen_ids.add(data['id'])
                existing_count += 1
        print(f"✓ Loaded {existing_count} existing comments")

    # Stats
    total_scraped = existing_count
    total_checked = 0

    print(f"\nTarget: {TARGET_COMMENTS:,} comments")
    print(f"Subreddits: {', '.join(SUBREDDITS)}")
    print(f"\nStarting scrape...\n")

    # Open output file in append mode
    with open(output_file, 'a') as f:
        for subreddit_name in SUBREDDITS:
            if total_scraped >= TARGET_COMMENTS:
                break

            print(f"[{subreddit_name}] Scraping...")

            try:
                subreddit = reddit.subreddit(subreddit_name)

                # Scrape from hot, new, and top posts
                for post in subreddit.hot(limit=500):
                    if total_scraped >= TARGET_COMMENTS:
                        break

                    # Expand all comments
                    post.comments.replace_more(limit=0)

                    # Process comments
                    for comment in post.comments.list():
                        total_checked += 1

                        # Skip if already seen
                        if comment.id in seen_ids:
                            continue

                        # Validate comment
                        if is_valid_comment(comment):
                            # Save comment
                            data = {
                                'id': comment.id,
                                'subreddit': subreddit_name,
                                'text': comment.body.strip(),
                                'score': comment.score,
                                'timestamp': datetime.fromtimestamp(comment.created_utc).isoformat()
                            }
                            f.write(json.dumps(data) + '\n')
                            f.flush()  # Ensure data is written

                            seen_ids.add(comment.id)
                            total_scraped += 1

                            # Progress update
                            if total_scraped % 100 == 0:
                                print(f"  Progress: {total_scraped:,}/{TARGET_COMMENTS:,} "
                                      f"({100*total_scraped/TARGET_COMMENTS:.1f}%) "
                                      f"[checked: {total_checked:,}]")

                            # Check if done
                            if total_scraped >= TARGET_COMMENTS:
                                break

                    # Rate limit (avoid API throttling)
                    time.sleep(0.1)

            except Exception as e:
                print(f"  ⚠ Error in r/{subreddit_name}: {e}")
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
    # Validate credentials
    if REDDIT_CLIENT_ID == "YOUR_CLIENT_ID_HERE":
        print("⚠ ERROR: Please update REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        print("\nSetup instructions:")
        print("1. Go to: https://www.reddit.com/prefs/apps")
        print("2. Create new app -> script")
        print("3. Copy 'client_id' and 'client_secret'")
        print("4. Update them in this file")
        exit(1)

    scrape_reddit_comments()
