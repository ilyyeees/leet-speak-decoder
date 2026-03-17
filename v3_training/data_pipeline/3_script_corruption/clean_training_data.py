#!/usr/bin/env python3
"""
Quick filtering script to remove LLM slop and bad examples from training data.
"""
import json
import re

input_file = "training_data_v3.jsonl"
output_file = "training_data_v3_CLEANED.jsonl"

bad_patterns = [
    "As an AI", "I cannot", "Here is", "Note:", "N0te:", "note:", "Translation:",
    "输出", "翻译"  # Common LLM leakage
]

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    kept = 0
    dropped = 0

    for line in fin:
        try:
            data = json.loads(line)
            inp = data['input_text']
            tgt = data['target_text']

            # 1. Check for LLM Slop / Non-ASCII (Chinese/Russian etc if mostly English)
            # This checks if there are Chinese characters
            if re.search(r'[\u4e00-\u9fff]', inp):
                dropped += 1
                continue

            # 2. Check for Hallucinated Instructions
            if any(p in inp for p in bad_patterns):
                dropped += 1
                continue

            # 3. Check Alignment (Target shouldn't be 3x longer than input)
            if len(tgt) > len(inp) * 3.0:
                dropped += 1
                continue

            # 4. Check for Empty/Too Short
            if len(inp) < 5 or len(tgt) < 5:
                dropped += 1
                continue

            fout.write(line)
            kept += 1

        except json.JSONDecodeError:
            continue

print(f"Cleaned Data: Kept {kept}, Dropped {dropped}")
