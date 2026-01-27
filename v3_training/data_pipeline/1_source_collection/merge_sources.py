#!/usr/bin/env python3
"""
Merge Source Files
===================
Combines sentences from all sources (WikiText, ELI5, LLM-generated)
into a single deduplicated file.

Input:
    - wikitext_sentences.jsonl
    - eli5_sentences.jsonl
    - llm_sentences.jsonl

Output: clean_sentences.jsonl
    {"text": "...", "source": "wikitext|eli5|llm_generated"}
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import Counter


def normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def load_jsonl(file_path: str) -> list:
    """Load sentences from a JSONL file."""
    sentences = []
    path = Path(file_path)
    
    if not path.exists():
        print(f"  [SKIP] {file_path} not found")
        return []
    
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    sentences.append(data)
            except json.JSONDecodeError:
                continue
    
    print(f"  [LOAD] {file_path}: {len(sentences)} sentences")
    return sentences


def merge_sources(
    output_path: str,
    source_files: list = None,
    target_count: int = 50000,
    shuffle: bool = True
) -> int:
    """
    Merge and deduplicate sentences from multiple sources.
    
    Args:
        output_path: Path to save merged JSONL
        source_files: List of input JSONL files (defaults to standard names)
        target_count: Maximum number of sentences to output
        shuffle: Whether to shuffle the final output
    """
    print("=" * 60)
    print("MERGE SOURCE FILES")
    print("=" * 60)
    
    if source_files is None:
        source_files = [
            "wikitext_sentences.jsonl",
            "eli5_sentences.jsonl",
            "llm_sentences.jsonl",
        ]
    
    print(f"\n[1/4] Loading source files...")
    
    all_sentences = []
    source_counts = Counter()
    
    for source_file in source_files:
        sentences = load_jsonl(source_file)
        all_sentences.extend(sentences)
        for s in sentences:
            source_counts[s.get('source', 'unknown')] += 1
    
    print(f"\n       Total loaded: {len(all_sentences)}")
    for source, count in source_counts.items():
        print(f"         - {source}: {count}")
    
    print(f"\n[2/4] Deduplicating...")
    
    seen = set()
    unique_sentences = []
    
    for item in all_sentences:
        text = item.get('text', '')
        normalized = normalize_for_dedup(text)
        
        if normalized not in seen and len(normalized) > 10:
            seen.add(normalized)
            unique_sentences.append(item)
    
    duplicates_removed = len(all_sentences) - len(unique_sentences)
    print(f"       Removed {duplicates_removed} duplicates")
    print(f"       Unique sentences: {len(unique_sentences)}")
    
    print(f"\n[3/4] Balancing sources...")
    
    # Group by source
    by_source = {}
    for item in unique_sentences:
        source = item.get('source', 'unknown')
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(item)
    
    # Calculate how many to take from each source
    num_sources = len(by_source)
    if num_sources > 0:
        per_source = target_count // num_sources
        
        balanced = []
        for source, items in by_source.items():
            # Take up to per_source from each, but if one source has less,
            # other sources can contribute more
            take = min(len(items), per_source + (target_count // num_sources))
            balanced.extend(items[:take])
            print(f"         - {source}: taking {min(len(items), take)}/{len(items)}")
        
        # If we still need more, take from remaining
        if len(balanced) < target_count:
            remaining = [s for s in unique_sentences if s not in balanced]
            balanced.extend(remaining[:target_count - len(balanced)])
        
        unique_sentences = balanced[:target_count]
    
    if shuffle:
        print(f"\n[3.5/4] Shuffling...")
        random.shuffle(unique_sentences)
    
    print(f"\n[4/4] Saving {len(unique_sentences)} sentences to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in unique_sentences:
            f.write(json.dumps(item) + '\n')
    
    # Final stats
    final_counts = Counter()
    for item in unique_sentences:
        final_counts[item.get('source', 'unknown')] += 1
    
    print(f"\n✓ Done! Saved {len(unique_sentences)} sentences")
    print(f"  Output: {output_path}")
    print(f"\n  Final distribution:")
    for source, count in sorted(final_counts.items()):
        pct = count / len(unique_sentences) * 100
        print(f"    - {source}: {count} ({pct:.1f}%)")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE SENTENCES (from each source):")
    print("=" * 60)
    
    for source in by_source.keys():
        source_items = [s for s in unique_sentences if s.get('source') == source]
        if source_items:
            sample = random.choice(source_items)
            print(f"  [{source}] {sample['text']}")
    
    return len(unique_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge sentence sources")
    parser.add_argument("--output", "-o", default="clean_sentences.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", "-n", type=int, default=50000,
                        help="Target number of sentences (default: 50000)")
    parser.add_argument("--sources", "-s", nargs="+",
                        default=["wikitext_sentences.jsonl", "eli5_sentences.jsonl", "llm_sentences.jsonl"],
                        help="Source JSONL files")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the output")
    
    args = parser.parse_args()
    merge_sources(args.output, args.sources, args.count, not args.no_shuffle)
