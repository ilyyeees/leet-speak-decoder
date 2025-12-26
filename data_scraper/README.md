# Reddit Leetspeak Scraper

## Quick Setup

### 1. Get Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app" at the bottom
3. Fill in:
   - Name: `leetspeak_scraper`
   - Type: Select "script"
   - Description: `Scraping leetspeak for ML`
   - Redirect URI: `http://localhost:8080` (doesn't matter for scripts)
4. Click "create app"
5. Copy the values:
   - **client_id**: The string under "personal use script" (looks like `xxxxxxxxxxx`)
   - **client_secret**: The "secret" field (looks like `xxxxxxxxxxxxxxxxxx`)

### 2. Configure the Scraper

Open `scrape_reddit.py` and update lines 29-30:

```python
REDDIT_CLIENT_ID = "your_client_id_here"
REDDIT_CLIENT_SECRET = "your_client_secret_here"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Scraper

```bash
python scrape_reddit.py
```

---

## What It Does

- ✅ Scrapes **50,000 comments** from gaming/casual subreddits
- ✅ Filters for leetspeak patterns (numbers, abbreviations, slang)
- ✅ Saves to `raw_comments.jsonl` (one comment per line)
- ✅ Resumes if interrupted (won't re-scrape existing comments)
- ✅ Progress tracking every 100 comments

---

## Configuration

Edit `scrape_reddit.py` to customize:

```python
SUBREDDITS = ['teenagers', 'gaming', ...]  # Which subreddits to scrape
TARGET_COMMENTS = 50000                    # How many to collect
MIN_COMMENT_LENGTH = 10                    # Filter short comments
MAX_COMMENT_LENGTH = 300                   # Filter long comments
```

---

## Output Format

`raw_comments.jsonl` contains one JSON object per line:

```json
{"id": "abc123", "subreddit": "gaming", "text": "idk wh4t 2 d0 tbh", "score": 5, "timestamp": "2024-01-01T12:00:00"}
```

---

## Next Steps

1. **Scrape** → Run `python scrape_reddit.py` on your cloud VM
2. **Download** → Copy `raw_comments.jsonl` to your local machine
3. **Translate** → Use your LLM to create clean versions
4. **Train** → Fine-tune on the (leetspeak, clean) pairs

---

## Troubleshooting

**Error: "Please update REDDIT_CLIENT_ID"**
→ You forgot to update the credentials in the script

**Error: 429 Rate Limit**
→ Reddit is throttling you. Wait 1 minute and try again.

**No comments found**
→ Check your credentials are correct
→ Try increasing `MAX_COMMENT_LENGTH`

**Slow scraping**
→ Normal. Expect ~500-1000 comments/hour (Reddit rate limits)
→ Run for 24-48 hours to hit 50k target
