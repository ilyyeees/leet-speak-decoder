#!/usr/bin/env python3
"""
ELI5 (Explain Like I'm 5) Collection
=====================================
Downloads and extracts clean English sentences from ELI5 subreddit answers.
ELI5 is great because the language is simple, clear, and conversational.

Output: eli5_sentences.jsonl
    {"text": "Water boils because the molecules move faster when heated.", "source": "eli5"}
"""

import json
import re
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets library...")
    import subprocess
    subprocess.check_call(["pip", "install", "datasets"])
    from datasets import load_dataset


def clean_sentence(text: str) -> str:
    """Clean and normalize a sentence."""
    # Remove Reddit-specific artifacts
    text = re.sub(r'/u/\S+', '', text)  # Remove u/username
    text = re.sub(r'/r/\S+', '', text)  # Remove r/subreddit
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    text = re.sub(r'\*+', '', text)  # Remove markdown bold/italic
    text = re.sub(r'#+\s*', '', text)  # Remove markdown headers
    text = re.sub(r'&gt;.*', '', text)  # Remove quotes
    text = re.sub(r'&amp;', '&', text)  # Fix HTML entities
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def is_valid_sentence(text: str, min_words: int = 6, max_words: int = 35) -> bool:
    """Check if sentence is valid for training."""
    words = text.split()
    
    # Length check
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Must have some letters
    if not any(c.isalpha() for c in text):
        return False
    
    # Skip if too many special characters
    special_count = sum(1 for c in text if c in '[]{}()<>@#$%^&*_+=|\\~`')
    if special_count > 2:
        return False
    
    # Skip sentences with existing leetspeak patterns
    if re.search(r'[a-zA-Z]\d[a-zA-Z]', text):  # Like "l33t"
        return False
    
    # Skip very short words average (likely garbage)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3:
        return False
    
    # Skip questions (we want statements for training)
    if text.strip().endswith('?'):
        return False
    
    return True


def extract_sentences(text: str) -> list:
    """Extract individual sentences from a paragraph."""
    # Handle common ELI5 patterns
    text = text.replace('ELI5:', '').replace('ELI5 ', '')
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def collect_eli5(output_path: str, target_count: int = 15000):
    """
    Collect clean sentences from ELI5 dataset.
    
    Args:
        output_path: Path to save JSONL output
        target_count: Number of sentences to collect
    """
    print("=" * 60)
    print("ELI5 (EXPLAIN LIKE I'M 5) COLLECTION")
    print("=" * 60)
    
    print(f"\n[1/4] Loading ELI5 dataset...")
    # Try 1: Main ELI5
    try:
        # Load ELI5 (legacy) without remote code
        dataset = load_dataset("eli5", split="train_eli5")
    except Exception:
        print(f"       Main ELI5 failed, trying alternative...")
        # Try 2: ELI5 Category
        try:
            dataset = load_dataset("eli5_category", split="train")
        except Exception:
            print(f"       ELI5 Category failed. Trying Reddit TIFU...")
            # Try 3: Reddit TIFU (Parquet version if avail, or standard)
            try:
                # Remove trust_remote_code completely
                dataset = load_dataset("reddit_tifu", "short", split="train")
            except Exception:
                print(f"       Reddit datasets failed. Switching to C4 RealNewsLike (reliable)...")
                # Try 4: C4 RealNewsLike (Guaranteed to work, standard parquet)
                try:
                    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
                    dataset = dataset.take(target_count * 2)
                except Exception as e:
                    print(f"[ERROR] All datasets failed. Last error: {e}")
                    return 0
    
    print(f"       Loaded {len(dataset)} entries")
    
    print(f"\n[2/4] Extracting and filtering sentences...")
    
    valid_sentences = set()  # Use set for deduplication
    
    for i, item in enumerate(dataset):
        # Get the answer/text field (varies by dataset)
        answers = item.get('answers', {})
        if isinstance(answers, dict):
            texts = answers.get('text', [])
        elif isinstance(answers, list):
            texts = answers
        else:
            texts = [item.get('text', ''), item.get('selftext', ''), item.get('body', '')]
        
        if not texts:
            texts = [item.get('selftext', ''), item.get('document', '')]
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            
            # Extract sentences
            sentences = extract_sentences(text)
            
            for sent in sentences:
                cleaned = clean_sentence(sent)
                
                if is_valid_sentence(cleaned):
                    valid_sentences.add(cleaned)
        
        # Progress
        if (i + 1) % 5000 == 0:
            print(f"       Processed {i+1} entries, found {len(valid_sentences)} valid sentences")
        
        # Early stop if we have enough
        if len(valid_sentences) >= target_count * 1.5:
            break
    
    print(f"\n[3/4] Found {len(valid_sentences)} unique valid sentences")
    
    # Take only what we need
    sentences_list = list(valid_sentences)[:target_count]
    
    print(f"\n[4/4] Saving {len(sentences_list)} sentences to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sent in sentences_list:
            data = {"text": sent, "source": "eli5"}
            f.write(json.dumps(data) + '\n')
    
    print(f"\n✓ Done! Saved {len(sentences_list)} sentences")
    print(f"  Output: {output_path}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES:")
    print("=" * 60)
    for sent in sentences_list[:5]:
        print(f"  • {sent}")
    
    return len(sentences_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect clean sentences from ELI5")
    parser.add_argument("--output", "-o", default="eli5_sentences.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", "-n", type=int, default=15000,
                        help="Target number of sentences (default: 15000)")
    
    args = parser.parse_args()
    collect_eli5(args.output, args.count)
