#!/usr/bin/env python3
"""
WikiText-103 Collection
========================
Downloads and extracts clean English sentences from WikiText-103.

Output: wikitext_sentences.jsonl
    {"text": "The quick brown fox jumps over the lazy dog.", "source": "wikitext"}
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
    # Remove wiki markup artifacts
    text = re.sub(r'@[^\s]+', '', text)  # Remove @-mentions
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [brackets]
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def is_valid_sentence(text: str, min_words: int = 5, max_words: int = 30) -> bool:
    """Check if sentence is valid for training."""
    words = text.split()
    
    # Length check
    if len(words) < min_words or len(words) > max_words:
        return False
    
    # Must start with capital letter
    if not text[0].isupper():
        return False
    
    # Must end with punctuation
    if text[-1] not in '.!?':
        return False
    
    # No weird characters (allow basic punctuation)
    if re.search(r'[^\w\s.,!?\'"-]', text):
        return False
    
    # Skip sentences that are mostly numbers
    num_count = sum(1 for c in text if c.isdigit())
    if num_count > len(text) * 0.2:
        return False
    
    # Skip sentences with existing leetspeak patterns
    if re.search(r'\d[a-zA-Z]|[a-zA-Z]\d', text):
        # Allow things like "2023" or "5 minutes" but not "l33t"
        # Check if digits are standalone
        words_with_mixed = [w for w in words if re.search(r'\d', w) and re.search(r'[a-zA-Z]', w)]
        if words_with_mixed:
            return False
    
    return True


def extract_sentences(text: str) -> list:
    """Extract individual sentences from a paragraph."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def collect_wikitext(output_path: str, target_count: int = 20000):
    """
    Collect clean sentences from WikiText-103.
    
    Args:
        output_path: Path to save JSONL output
        target_count: Number of sentences to collect
    """
    print("=" * 60)
    print("WIKITEXT-103 COLLECTION")
    print("=" * 60)
    
    print(f"\n[1/4] Loading WikiText-103 dataset...")
    try:
        # Primary: Salesforce/wikitext (Parquet version, no custom code needed)
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    except Exception as e:
        print(f"       Salesforce/wikitext failed ({e}). Trying fallback...")
        try:
            # Fallback: old identifier (might work if cached)
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        except Exception as e2:
            print(f"[ERROR] Could not load WikiText: {e2}")
            return 0
    print(f"       Loaded {len(dataset)} paragraphs")
    
    print(f"\n[2/4] Extracting and filtering sentences...")
    
    valid_sentences = set()  # Use set for deduplication
    
    for i, item in enumerate(dataset):
        text = item.get('text', '')
        
        if not text.strip():
            continue
        
        # Extract sentences
        sentences = extract_sentences(text)
        
        for sent in sentences:
            cleaned = clean_sentence(sent)
            
            if is_valid_sentence(cleaned):
                valid_sentences.add(cleaned)
        
        # Progress
        if (i + 1) % 10000 == 0:
            print(f"       Processed {i+1} paragraphs, found {len(valid_sentences)} valid sentences")
        
        # Early stop if we have enough
        if len(valid_sentences) >= target_count * 1.5:  # Get extras for filtering
            break
    
    print(f"\n[3/4] Found {len(valid_sentences)} unique valid sentences")
    
    # Take only what we need
    sentences_list = list(valid_sentences)[:target_count]
    
    print(f"\n[4/4] Saving {len(sentences_list)} sentences to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sent in sentences_list:
            data = {"text": sent, "source": "wikitext"}
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
    parser = argparse.ArgumentParser(description="Collect clean sentences from WikiText-103")
    parser.add_argument("--output", "-o", default="wikitext_sentences.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", "-n", type=int, default=20000,
                        help="Target number of sentences (default: 20000)")
    
    args = parser.parse_args()
    collect_wikitext(args.output, args.count)
