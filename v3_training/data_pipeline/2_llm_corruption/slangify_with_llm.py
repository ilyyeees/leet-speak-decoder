#!/usr/bin/env python3
"""
LLM Slangification
===================
Converts clean English sentences to internet slang using Ollama.
This does SEMANTIC/PHONETIC corruption only (abbreviations, phonetic spellings).
NO visual leetspeak (numbers for letters) - that's the script's job.

Input: clean_sentences.jsonl
    {"text": "I really do not know what to do.", "source": "wikitext"}

Output: slang_pairs.jsonl
    {"original": "I really do not know what to do.", "slang": "idk wut to do tbh", "persona": "lazy_texter"}

Requirements:
    - Ollama running with qwen2.5:32b (or similar)
    - pip install requests tqdm
"""

import json
import time
import re
import argparse
import random
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from prompts import get_persona_prompt, get_random_persona, PERSONAS

# Default settings
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:32b"


def query_ollama(
    text: str,
    persona: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL
) -> str:
    """Send a single text to Ollama for slangification."""
    prompt = get_persona_prompt(persona)
    full_prompt = f"{prompt}\n\nInput: \"{text}\"\nOutput:"
    
    for attempt in range(3):
        try:
            response = requests.post(
                ollama_url,
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 256,
                    }
                },
                timeout=180  # Increased for 32B model
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            
            # Clean up the result
            result = clean_llm_output(result, text)
            return result
            
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(2)
                continue
        except Exception as e:
            # On other errors or final timeout, return empty
            pass
            
    return ""


def clean_llm_output(output: str, original: str) -> str:
    """Clean and validate LLM output."""
    # Remove quotes if present
    output = output.strip('"\'')
    
    # Remove "Output:" prefix if LLM repeated it
    output = re.sub(r'^Output:\s*', '', output, flags=re.IGNORECASE)
    
    # Remove any numbering
    output = re.sub(r'^\d+[.)]\s*', '', output)
    
    # If output is way too different from input, it might be garbage
    # (LLM sometimes hallucinates completely different text)
    original_words = set(original.lower().split())
    output_words = set(output.lower().split())
    
    # At least 20% of core words should be recognizable
    # (accounting for abbreviations changing words)
    if len(output.split()) < 2:
        return ""
    
    # If output is way longer than input, probably garbage
    if len(output) > len(original) * 2:
        return ""
    
    # If output contains obvious leetspeak (numbers in words), reject
    # The LLM sometimes ignores our instructions
    if re.search(r'[a-zA-Z][0-9][a-zA-Z]', output):  # like "l33t"
        # Try to clean it
        output = re.sub(r'3', 'e', output)
        output = re.sub(r'4', 'a', output)
        output = re.sub(r'1([a-zA-Z])', r'i\1', output)
        output = re.sub(r'0', 'o', output)
    
    return output.strip()


def process_batch(
    items: list,
    model: str,
    ollama_url: str,
    workers: int = 4
) -> list:
    """Process a batch of items in parallel."""
    results = []
    
    def process_single(item):
        text = item.get('text', '')
        source = item.get('source', 'unknown')
        
        # Pick a random persona for variety
        persona = get_random_persona()
        
        slang = query_ollama(text, persona, model, ollama_url)
        
        if slang and slang != text:
            return {
                "original": text,
                "slang": slang,
                "source": source,
                "persona": persona
            }
        return None
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single, item): item for item in items}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    return results


def slangify_sentences(
    input_path: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    batch_size: int = 50,
    workers: int = 4,
    max_samples: int = None
):
    """
    Convert clean sentences to slang using LLM.
    
    Args:
        input_path: Path to clean_sentences.jsonl
        output_path: Path to save slang_pairs.jsonl
        model: Ollama model name
        ollama_url: Ollama API endpoint
        batch_size: Items per batch for progress tracking
        workers: Parallel workers for Ollama calls
        max_samples: Limit processing (for testing)
    """
    print("=" * 60)
    print("LLM SLANGIFICATION")
    print("=" * 60)
    print(f"\n  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Model:  {model}")
    print(f"  Ollama: {ollama_url}")
    
    # Test connection
    print(f"\n[1/4] Testing Ollama connection...")
    test_result = query_ollama("Hello world", "lazy_texter", model, ollama_url)
    if not test_result:
        print("[ERROR] Cannot connect to Ollama!")
        print("        Make sure Ollama is running: ollama serve")
        print(f"        And model is pulled: ollama pull {model}")
        return
    print(f"       OK! Test: 'Hello world' → '{test_result}'")
    
    # Load input
    print(f"\n[2/4] Loading input sentences...")
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return
    
    items = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if max_samples:
        items = items[:max_samples]
    
    print(f"       Loaded {len(items)} sentences")
    
    # Show persona distribution we'll use
    print(f"\n       Using {len(PERSONAS)} personas:")
    for key, persona in PERSONAS.items():
        print(f"         - {persona['name']}")
    
    # Process
    print(f"\n[3/4] Converting to slang...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_success = 0
    
    with open(output_file, 'w') as f_out:
        # Process in batches for progress tracking
        for i in tqdm(range(0, len(items), batch_size), desc="Processing"):
            batch = items[i:i + batch_size]
            results = process_batch(batch, model, ollama_url, workers)
            
            for result in results:
                f_out.write(json.dumps(result) + '\n')
                total_success += 1
            
            total_processed += len(batch)
    
    success_rate = total_success / total_processed * 100 if total_processed > 0 else 0
    
    print(f"\n[4/4] Done!")
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Processed:    {total_processed}")
    print(f"  Successful:   {total_success}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Output:       {output_path}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSIONS:")
    print("=" * 60)
    
    with open(output_file, 'r') as f:
        samples = []
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= 5:
                break
    
    for sample in samples:
        print(f"\n  [{sample.get('persona', '?')}]")
        print(f"  Original: {sample['original']}")
        print(f"  Slang:    {sample['slang']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert clean English to slang")
    parser.add_argument("--input", "-i", default="../1_source_collection/clean_sentences.jsonl",
                        help="Input JSONL file with clean sentences")
    parser.add_argument("--output", "-o", default="slang_pairs.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--model", "-m", default="qwen2.5:32b",
                        help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate",
                        help="Ollama API URL")
    parser.add_argument("--batch-size", "-b", type=int, default=100,
                        help="Batch size for progress tracking")
    parser.add_argument("--workers", "-w", type=int, default=12,
                        help="Parallel workers (default: 12 - 2x RTX 5090 optimized)")
    parser.add_argument("--max", "-n", type=int, default=None,
                        help="Max samples to process (for testing)")
    
    args = parser.parse_args()
    
    slangify_sentences(
        args.input,
        args.output,
        args.model,
        args.ollama_url,
        args.batch_size,
        args.workers,
        args.max
    )
