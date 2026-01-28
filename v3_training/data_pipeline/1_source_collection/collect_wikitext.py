#!/usr/bin/env python3
"""
FineWeb-Edu Collection (Modern Replacement for WikiText)
========================================================
Downloads high-quality educational text from HuggingFaceFW/fineweb-edu.
This dataset is much cleaner and more diverse than WikiText-103.

Output: wikitext_sentences.jsonl (keeping filename for compatibility)
    {"text": "...", "source": "fineweb-edu"}
"""

import json
import re
import argparse
from pathlib import Path
from datasets import load_dataset, disable_progress_bar

disable_progress_bar()


def clean_sentence(text: str) -> str:
    """Clean and normalize a sentence."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_valid_sentence(text: str) -> bool:
    """Check if sentence is valid for training."""
    words = text.split()
    
    # Length check
    if len(words) < 5 or len(words) > 30:
        return False
    
    # Must start with capital letter
    if not text[0].isupper():
        return False
    
    # Must end with punctuation
    if text[-1] not in '.!?':
        return False
    
    # No weird characters
    if re.search(r'[^\w\s.,!?\'"-]', text):
        return False
    
    # Skip sentences with too many numbers
    num_count = sum(1 for c in text if c.isdigit())
    if num_count > len(text) * 0.2:
        return False
    
    return True


def extract_sentences(text: str) -> list:
    """Extract individual sentences from a paragraph."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def collect_wikitext(output_path: str, target_count: int = 20000):
    """
    Collect clean sentences from FineWeb-Edu.
    """
    print("=" * 60)
    print("FINEWEB-EDU COLLECTION (MODERN WIKITEXT)")
    print("=" * 60)
    
    print(f"\n[1/4] Loading FineWeb-Edu dataset (streaming)...")
    try:
        # Use the sample version for speed, streaming
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    except Exception as e:
        print(f"[ERROR] Could not load FineWeb-Edu: {e}")
        return 0
    
    print(f"\n[2/4] Extracting and filtering sentences...")
    
    valid_sentences = set()
    
    # Iterate through streaming dataset
    for i, item in enumerate(dataset):
        text = item.get('text', '')
        
        if not text:
            continue
            
        sentences = extract_sentences(text)
        
        for sent in sentences:
            cleaned = clean_sentence(sent)
            if is_valid_sentence(cleaned):
                valid_sentences.add(cleaned)
        
        if (i + 1) % 100 == 0:
            print(f"       Processed {i+1} documents, found {len(valid_sentences)} valid sentences", end='\r')
            
        if len(valid_sentences) >= target_count * 1.2:
            break
            
    print(f"\n\n[3/4] Found {len(valid_sentences)} unique valid sentences")
    
    # Take only what we need
    sentences_list = list(valid_sentences)[:target_count]
    
    print(f"\n[4/4] Saving to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sent in sentences_list:
            data = {"text": sent, "source": "fineweb-edu"}
            f.write(json.dumps(data) + '\n')
    
    print(f"\n✓ Done! Saved {len(sentences_list)} sentences")
    print(f"  Output: {output_path}")
    
    return len(sentences_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect clean sentences from FineWeb-Edu")
    parser.add_argument("--output", "-o", default="wikitext_sentences.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", "-n", type=int, default=20000,
                        help="Target number of sentences")
    
    args = parser.parse_args()
    collect_wikitext(args.output, args.count)
