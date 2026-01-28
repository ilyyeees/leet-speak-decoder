#!/usr/bin/env python3
"""
UltraChat Collection (Modern Replacement for ELI5)
==================================================
Downloads high-quality conversational text from HuggingFaceH4/ultrachat_200k.
This dataset contains clean Q&A pairs perfect for training conversational models.

Output: eli5_sentences.jsonl (keeping filename for compatibility)
    {"text": "...", "source": "ultrachat"}
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
    # Remove markdown code blocks
    text = re.sub(r'```.*?```', '', text)
    return text.strip()


def is_valid_sentence(text: str) -> bool:
    """Check if sentence is valid for training."""
    words = text.split()
    
    # Length check
    if len(words) < 6 or len(words) > 35:
        return False
    
    # Must have letters
    if not any(c.isalpha() for c in text):
        return False
        
    # Skip if too many special characters (code, math)
    special_count = sum(1 for c in text if c in '{}[]@#$%^&*_=+|\\<>')
    if special_count > 2:
        return False
        
    # Skip questions (we prioritize statements for this part)
    if '?' in text:
        return False
        
    return True


def extract_sentences(text: str) -> list:
    """Extract individual sentences from a message."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def collect_eli5(output_path: str, target_count: int = 15000):
    """
    Collect clean sentences from UltraChat.
    """
    print("=" * 60)
    print("ULTRACHAT COLLECTION (MODERN ELI5)")
    print("=" * 60)
    
    print(f"\n[1/4] Loading UltraChat 200k dataset...")
    try:
        # Standard Parquet dataset, no custom code needed
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    except Exception as e:
        print(f"[ERROR] Could not load UltraChat: {e}")
        return 0
    
    print(f"\n[2/4] Extracting and filtering sentences...")
    
    valid_sentences = set()
    
    for i, item in enumerate(dataset):
        # UltraChat has a 'messages' list
        messages = item.get('messages', [])
        
        for msg in messages:
            content = msg.get('content', '')
            if not content:
                continue
                
            sentences = extract_sentences(content)
            
            for sent in sentences:
                cleaned = clean_sentence(sent)
                if is_valid_sentence(cleaned):
                    valid_sentences.add(cleaned)
        
        if (i + 1) % 500 == 0:
            print(f"       Processed {i+1} dialogues, found {len(valid_sentences)} valid sentences", end='\r')
            
        if len(valid_sentences) >= target_count * 1.5:
            break
            
    print(f"\n\n[3/4] Found {len(valid_sentences)} unique valid sentences")
    
    # Take only what we need
    sentences_list = list(valid_sentences)[:target_count]
    
    print(f"\n[4/4] Saving to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sent in sentences_list:
            data = {"text": sent, "source": "ultrachat"}
            f.write(json.dumps(data) + '\n')
    
    print(f"\n✓ Done! Saved {len(sentences_list)} sentences")
    print(f"  Output: {output_path}")
    
    return len(sentences_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect clean sentences from UltraChat")
    parser.add_argument("--output", "-o", default="eli5_sentences.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", "-n", type=int, default=15000,
                        help="Target number of sentences")
    
    args = parser.parse_args()
    collect_eli5(args.output, args.count)
