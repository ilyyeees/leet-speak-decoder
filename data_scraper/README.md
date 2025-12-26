# Reddit Leetspeak Scraper (HTML - No API!)

## ✅ No Reddit API Needed!

This scraper uses HTML parsing, so you don't need any API credentials.

---

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

That's it! No API keys needed.

---

### 2. Run the Scraper

```bash
python scrape_reddit_html.py
```

The scraper will:
- ✅ Scrape 50,000 comments from gaming/casual subreddits
- ✅ Filter for leetspeak patterns automatically
- ✅ Save to `raw_comments.jsonl`
- ✅ Resume if interrupted (won't re-scrape)
- ✅ Use random delays to avoid rate limits

---

## Configuration

Edit `scrape_reddit_html.py` to customize:

```python
SUBREDDITS = ['teenagers', 'gaming', ...]  # Which subreddits
TARGET_COMMENTS = 50000                    # How many to collect
DELAY_BETWEEN_REQUESTS = (2, 5)           # Random delay (seconds)
```

---

## Output Format

`raw_comments.jsonl` contains one JSON object per line:

```json
{"subreddit": "gaming", "text": "idk wh4t 2 d0 tbh", "score": 5, "timestamp": "2024-01-01T12:00:00"}
```

---

## Deployment to Cloud

### Option 1: DigitalOcean Droplet ($6/month)

```bash
# Create droplet (Ubuntu 22.04)
# SSH into it
ssh root@your-droplet-ip

# Clone repo
git clone https://github.com/ilyyeees/leet-speak-decoder.git
cd leet-speak-decoder/data_scraper

# Install Python + deps
apt update && apt install -y python3-pip
pip3 install -r requirements.txt

# Run scraper in background
nohup python3 scrape_reddit_html.py > scraper.log 2>&1 &

# Check progress
tail -f scraper.log

# Download results later
# From your local machine:
scp root@your-droplet-ip:~/leet-speak-decoder/data_scraper/raw_comments.jsonl .
```

---

### Option 2: AWS EC2 Free Tier

```bash
# Launch t2.micro instance (Ubuntu)
# SSH in
ssh -i your-key.pem ubuntu@ec2-instance-ip

# Same steps as above
```

---

### Option 3: Google Colab (Free!)

```python
# In a Colab notebook:
!git clone https://github.com/ilyyeees/leet-speak-decoder.git
%cd leet-speak-decoder/data_scraper
!pip install -r requirements.txt
!python scrape_reddit_html.py

# Download result
from google.colab import files
files.download('raw_comments.jsonl')
```

---

## Performance

- **Speed**: ~500-1000 comments/hour (Reddit rate limits)
- **Time to 50k**: 24-48 hours
- **Hit rate**: ~10-20% (most comments don't have leetspeak)
- **Network**: ~10-50MB total traffic

---

## How It Works

1. Fetches subreddit hot posts from `old.reddit.com`
2. For each post, fetches all comments
3. Filters comments for leetspeak patterns:
   - Numbers as letters (1, 3, 4, 7, 0)
   - Common abbreviations (idk, omg, brb, etc.)
   - Repeated characters
4. Saves valid comments to JSONL
5. Uses random delays + user-agent rotation to avoid blocking

---

## Troubleshooting

**Error: "429 Rate Limited"**
→ Normal. Scraper will auto-retry after 60s wait.

**Very slow progress**
→ Expected. Reddit has aggressive rate limits.
→ Run on cloud for 24-48 hours to hit 50k target.

**No comments found**
→ Reddit might be blocking your IP
→ Try from a different IP (VPN or cloud server)

**Script crashes**
→ Just re-run it. Progress is saved automatically.

---

## Next Steps

1. **Deploy to cloud** → Let it run for 24-48 hours
2. **Download `raw_comments.jsonl`** to your local machine
3. **Use your LLM** to translate to clean English
4. **Create training pairs** → Fine-tune the model
