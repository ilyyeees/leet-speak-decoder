#!/usr/bin/env python3
"""
Gemini Translator - Convert Reddit comments to formal English
==============================================================
Uses Google's Gemini API to translate slang/leetspeak to proper English.

Usage:
    1. Install: pip install google-genai
    2. Set your API key in the script or as environment variable
    3. Run: python3 translate_with_gemini.py
"""

import json
import time
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set your API key here OR use environment variable GEMINI_API_KEY
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

# Input/Output files
INPUT_FILE = "raw_comments.jsonl"
OUTPUT_FILE = "translated_pairs.jsonl"

# Rate limiting (to avoid API quota issues)
DELAY_BETWEEN_REQUESTS = 0.5  # seconds
BATCH_SIZE = 100  # Save progress every N translations

# Model to use
MODEL_NAME = "gemini-2.5-flash"


# ============================================================================
# GEMINI CLIENT
# ============================================================================

def setup_gemini():
    """Initialize the Gemini client."""
    try:
        from google import genai
    except ImportError:
        print("Error: google-genai not installed")
        print("Run: pip install google-genai")
        exit(1)

    api_key = GEMINI_API_KEY if GEMINI_API_KEY != "YOUR_API_KEY_HERE" else os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: No API key found")
        print("Either set GEMINI_API_KEY in the script or as environment variable")
        exit(1)

    client = genai.Client(api_key=api_key)
    return client


def translate_to_formal(client, text: str) -> str:
    """
    Translate a single comment to formal English.
    """
    prompt = f"""You are a translator that converts internet slang, leetspeak, and casual text into proper formal English.

Rules:
1. Expand ALL abbreviations (idk → I don't know, tbh → to be honest, rn → right now, etc.)
2. Convert ALL leetspeak numbers to letters (h3ll0 → hello, w0rld → world, l8r → later, etc.)
3. Fix grammar and punctuation
4. Keep the original meaning intact
5. Output ONLY the translated text, nothing else

Input: {text}

Formal English:"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        result = response.text.strip()

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
    print("GEMINI TRANSLATOR")
    print("=" * 60)

    # Setup
    client = setup_gemini()
    print(f"✓ Gemini client initialized (model: {MODEL_NAME})")

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
                    if processed % 10 == 0:
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
