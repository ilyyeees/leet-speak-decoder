#!/usr/bin/env python3
"""
Groq Translator - Convert Reddit comments to formal English
==============================================================
Uses Groq's API to translate slang/leetspeak to proper English.

Usage:
    1. Install: pip install groq
    2. Set your API key in the script or as environment variable GROQ_API_KEY
    3. Run: python3 translate_with_gemini.py
"""

import json
import time
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set your API key here OR use environment variable GROQ_API_KEY
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"

# Input/Output files
INPUT_FILE = "raw_comments.jsonl"
OUTPUT_FILE = "translated_pairs.jsonl"

# Rate limiting (Groq free tier allows ~30 requests/min)
DELAY_BETWEEN_REQUESTS = 2  # seconds
BATCH_SIZE = 25  # Save progress every N translations

# Model to use (Groq models - llama3-8b-8192 is fast and good)
MODEL_NAME = "llama-3.1-8b-instant"


# ============================================================================
# GROQ CLIENT
# ============================================================================

def setup_groq():
    """Initialize the Groq client."""
    try:
        from groq import Groq
    except ImportError:
        print("Error: groq not installed")
        print("Run: pip install groq")
        exit(1)

    api_key = GROQ_API_KEY if GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE" else os.getenv("GROQ_API_KEY")

    if not api_key:
        print("Error: No API key found")
        print("Either set GROQ_API_KEY in the script or as environment variable")
        exit(1)

    client = Groq(api_key=api_key)
    return client


def translate_to_formal(client, text: str) -> str:
    """
    Translate a single comment to clean English.
    """
    prompt = f"""Clean up this text by ONLY doing these things:

1. Expand abbreviations: idk → I don't know, tbh → to be honest, rn → right now, omg → oh my god, btw → by the way, etc.
2. Fix leetspeak numbers: h3ll0 → hello, w0rld → world, l8r → later, n00b → noob, etc.
3. Fix basic grammar/punctuation if needed.

IMPORTANT RULES:
- DO NOT paraphrase or use fancier words
- DO NOT change the meaning or tone
- DO NOT make it overly formal
- Keep slang words like "bro", "dude", "lol" if they're not abbreviations
- Keep it casual and natural sounding
- Output ONLY the cleaned text, nothing else

Input: {text}

Cleaned:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )

        result = response.choices[0].message.content.strip()

        # Clean up any quotes if present
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result

    except Exception as e:
        print(f"  Error: {e}")
        return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_comments():
    """Process all comments from input file."""

    print("=" * 60)
    print("GROQ TRANSLATOR")
    print("=" * 60)

    # Setup
    client = setup_groq()
    print(f"✓ Groq client initialized (model: {MODEL_NAME})")

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    # Count total lines
    with open(input_path, 'r') as f:
        total_lines = sum(1 for _ in f)
    print(f"✓ Found {total_lines} comments to translate")

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

    # Process
    processed = 0
    skipped = 0
    errors = 0

    print(f"\nStarting translation...\n")

    with open(input_path, 'r') as f_in, open(output_path, 'a') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                original_text = data.get('text', '')

                # Skip if already translated
                if original_text in existing_texts:
                    skipped += 1
                    continue

                # Translate
                formal_text = translate_to_formal(client, original_text)

                if formal_text:
                    # Save pair
                    output_data = {
                        'original': original_text,
                        'formal': formal_text,
                    }
                    f_out.write(json.dumps(output_data) + '\n')
                    f_out.flush()

                    existing_texts.add(original_text)
                    processed += 1

                    # Progress
                    if processed % BATCH_SIZE == 0:
                        print(f"[{processed + skipped}/{total_lines}] Translated: {processed}, Errors: {errors}")
                        print(f"  Original: {original_text[:50]}...")
                        print(f"  Formal:   {formal_text[:50]}...")
                else:
                    errors += 1

                # Rate limit
                time.sleep(DELAY_BETWEEN_REQUESTS)

            except json.JSONDecodeError:
                errors += 1
                continue
            except KeyboardInterrupt:
                print("\n\nInterrupted! Progress saved.")
                break
            except Exception as e:
                print(f"  Error: {e}")
                errors += 1
                continue

    # Summary
    print("\n" + "=" * 60)
    print("TRANSLATION COMPLETE")
    print("=" * 60)
    print(f"Translated: {processed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nOutput: {output_path.absolute()}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    process_comments()
