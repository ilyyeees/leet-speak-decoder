#!/usr/bin/env python3
"""
Ollama Translator - Convert Reddit comments to formal English
==============================================================
Uses local Ollama API (no rate limits!) for maximum speed.

Usage:
    1. Install Ollama and pull model: ollama pull llama3.1:8b
    2. Start Ollama: ollama serve
    3. Run: python3 translate_with_ollama.py
"""

import json
import time
import os
import concurrent.futures
from pathlib import Path

# Try to import requests, fall back to urllib if not available
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    USE_REQUESTS = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Ollama API endpoint (change IP if running on remote server)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Input/Output files
INPUT_FILE = "raw_comments.jsonl"
OUTPUT_FILE = "translated_pairs.jsonl"

# Performance settings (no rate limits with local LLM!)
MAX_WORKERS = 2  # Parallel requests (increase if GPU can handle more)
BATCH_SIZE = 50  # Progress update frequency

# Model to use
MODEL_NAME = "qwen2.5:32b"


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

def translate_to_formal(text: str) -> str:
    """
    Translate a single comment to clean English using Ollama Chat API.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a text cleaner. Fix abbreviations (idk→I don't know, tbh→to be honest, rn→right now, etc), fix leetspeak (h3ll0→hello, w0rld→world, l8r→later, etc), and fix grammar. Keep the casual tone. Output ONLY the cleaned text. Do not say 'Here is the text' or add any explanation."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
        }
    }

    try:
        if USE_REQUESTS:
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json().get("message", {}).get("content", "").strip()
        else:
            req = urllib.request.Request(
                f"{OLLAMA_HOST}/api/chat",
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode()).get("message", {}).get("content", "").strip()

        # Clean up any quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result

    except Exception as e:
        print(f"  Error: {e}")
        return None


def process_single(item):
    """Process a single comment. Used for parallel execution."""
    original_text, idx = item
    formal_text = translate_to_formal(original_text)
    return (original_text, formal_text, idx)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_comments():
    """Process all comments from input file."""

    print("=" * 60)
    print("OLLAMA TRANSLATOR (Local LLM - No Rate Limits!)")
    print("=" * 60)

    # Verify Ollama is running
    try:
        if USE_REQUESTS:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            r.raise_for_status()
        else:
            urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5)
        print(f"✓ Ollama connected at {OLLAMA_HOST}")
    except Exception as e:
        print(f"✗ Cannot connect to Ollama at {OLLAMA_HOST}")
        print(f"  Error: {e}")
        print(f"\nMake sure Ollama is running:")
        print(f"  ollama serve")
        return

    print(f"✓ Model: {MODEL_NAME}")
    print(f"✓ Parallel workers: {MAX_WORKERS}")

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    # Load all comments
    comments = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                if text:
                    comments.append(text)
            except:
                pass

    print(f"✓ Loaded {len(comments)} comments")

    # Load existing progress if resuming
    existing_texts = set()
    if output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_texts.add(data.get('original', ''))
                except:
                    pass
        print(f"✓ Resuming: {len(existing_texts)} already translated")

    # Filter out already translated
    to_translate = [(text, i) for i, text in enumerate(comments) if text not in existing_texts]
    print(f"✓ Remaining: {len(to_translate)} comments to translate")

    if not to_translate:
        print("\nAll comments already translated!")
        return

    # Process with parallel workers
    processed = 0
    errors = 0
    start_time = time.time()

    print(f"\nStarting translation with {MAX_WORKERS} workers...\n")

    with open(output_path, 'a') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            try:
                for original_text, formal_text, idx in executor.map(process_single, to_translate):
                    if formal_text:
                        output_data = {
                            'original': original_text,
                            'formal': formal_text,
                        }
                        f_out.write(json.dumps(output_data) + '\n')
                        f_out.flush()
                        processed += 1

                        if processed % BATCH_SIZE == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed
                            remaining = len(to_translate) - processed
                            eta = remaining / rate if rate > 0 else 0
                            print(f"[{processed}/{len(to_translate)}] "
                                  f"Rate: {rate:.1f}/s | ETA: {eta:.0f}s | Errors: {errors}")
                    else:
                        errors += 1

            except KeyboardInterrupt:
                print("\n\nInterrupted! Progress saved.")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRANSLATION COMPLETE")
    print("=" * 60)
    print(f"Translated: {processed}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.1f}s ({processed/elapsed:.1f} comments/sec)")
    print(f"\nOutput: {output_path.absolute()}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    process_comments()
