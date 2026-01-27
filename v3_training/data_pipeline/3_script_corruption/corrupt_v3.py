#!/usr/bin/env python3
"""
V3 Script Corruption
=====================
Takes slangified text and applies VISUAL leetspeak corruption.
This is the "mechanical" layer that adds character swaps, noise, etc.

Key changes from v2's corrupt_to_leetspeak.py:
1. NO phrase/word replacements (LLM already did that)
2. Focus on char_map intensity
3. Digit protection (won't corrupt existing numbers like "2nite")
4. 10x multiplexing per input (different intensities/seeds)

Input: slang_pairs.jsonl
    {"original": "I really do not know what to do.", "slang": "idk wut to do tbh"}

Output: training_data_v3.jsonl
    {"input_text": "1dk wut 2 d0 tbh", "target_text": "I really do not know what to do."}
"""

import json
import re
import random
import argparse
from pathlib import Path
from typing import Tuple


class V3LeetCorruptor:
    """
    Focused leetspeak corruptor for V3.
    Only does VISUAL corruption (char swaps, noise).
    Word-level slang was already handled by the LLM.
    """
    
    def __init__(self):
        # ================================================================
        # CHARACTER MAPPINGS (visual leetspeak only)
        # ================================================================
        
        # Simple/readable swaps (use more often)
        self.simple_map = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$', 'z'],  # z is common for plural/suffix s (skillz, gamez)
            't': ['7', '+'],
            'l': ['1'],
            'b': ['8'],
            'g': ['9', '6'],
        }
        
        # Complex swaps (use less often, for "heavy" intensity)
        self.complex_map = {
            'a': ['4', '@', '/\\'],
            'e': ['3', '&'],
            'i': ['1', '!', '|'],
            'o': ['0', '()'],
            'u': ['v', 'uu'],
            'b': ['8', '|3'],
            'c': ['(', '<'],
            'd': ['|)', '|]'],
            'f': ['ph'],
            'g': ['9', '6'],
            'h': ['#', '|-|'],
            'k': ['|<'],
            'l': ['1', '|'],
            'n': ['|\\|'],
            's': ['5', '$', 'z'],
            't': ['7', '+'],
            'v': ['|/', '\\/'],
            'w': ['vv', 'uu'],
            'x': ['><', '%'],
            # 'y' left alone - rarely corrupted in real leetspeak
            'z': ['2', '%'],
        }
        # NOTE: No protected_words list!
        # We WANT to corrupt slang like 'thx' -> '7hx', 'idk' -> '1dk'
        # The model needs to learn these variations.
        # Only protection: has_digit() prevents destroying LLM-added numbers like '2nite'
    
    def has_digit(self, word: str) -> bool:
        """Check if word already contains digits (from LLM like '2nite', 'l8r')."""
        return bool(re.search(r'\d', word))
    
    def corrupt_char(self, char: str, intensity: float, use_complex: bool) -> str:
        """
        Replace a single character with leetspeak.
        
        Args:
            char: Character to potentially replace
            intensity: 0.0-1.0, probability of replacement
            use_complex: If True, use complex character map
        """
        if random.random() > intensity:
            return char
        
        lower = char.lower()
        char_map = self.complex_map if use_complex else self.simple_map
        
        if lower in char_map:
            replacement = random.choice(char_map[lower])
            # Preserve case for first char
            if char.isupper() and len(replacement) > 0:
                return replacement[0].upper() + replacement[1:]
            return replacement
        
        return char
    
    def corrupt_word(self, word: str, intensity: float, use_complex: bool) -> str:
        """
        Apply character-level corruption to a word.
        
        Only protects words that already have digits (LLM-added like "2nite").
        We intentionally corrupt slang words (thx -> 7hx, idk -> 1dk).
        """
        # Only skip words that already have digits from LLM (like '2nite', 'l8r')
        if self.has_digit(word):
            return word
        
        # Apply character corruption
        result = []
        for char in word:
            result.append(self.corrupt_char(char, intensity, use_complex))
        
        return ''.join(result)
    
    def add_noise(self, text: str, intensity: float) -> str:
        """Add random character noise (stretching, typos)."""
        if random.random() > intensity * 0.3:  # Less frequent than char swaps
            return text
        
        words = text.split()
        result = []
        
        for word in words:
            # Protect slang terms
            if word.lower() in self.protected_words or self.has_digit(word):
                result.append(word)
                continue
            
            # Random character stretch (10% chance)
            if random.random() < 0.1 and len(word) > 2:
                pos = random.randint(0, len(word) - 1)
                char = word[pos]
                if char.isalpha():
                    stretch = random.randint(2, 4)
                    word = word[:pos] + char * stretch + word[pos+1:]
            
            result.append(word)
        
        return ' '.join(result)
    
    def apply_case_chaos(self, text: str) -> str:
        """Random case changes (rare, for variety)."""
        roll = random.random()
        
        if roll < 0.05:  # 5% ALL CAPS
            return text.upper()
        
        if roll < 0.10:  # 5% aLtErNaTiNg
            result = []
            upper = random.choice([True, False])
            for char in text:
                if char.isalpha():
                    result.append(char.upper() if upper else char.lower())
                    upper = not upper
                else:
                    result.append(char)
            return ''.join(result)
        
        if roll < 0.30:  # 20% all lowercase
            return text.lower()
        
        return text
    
    def corrupt(self, text: str, intensity: float = 0.5) -> str:
        """
        Main corruption function.
        
        Args:
            text: Slang text to corrupt further
            intensity: 0.0-1.0, controls how aggressive the corruption is
                      0.0-0.3 = light (some char swaps)
                      0.3-0.6 = medium (more swaps, some noise)
                      0.6-1.0 = heavy (lots of swaps, noise, case chaos)
        
        Returns:
            Corrupted leetspeak text
        """
        use_complex = intensity > 0.5
        
        # Word-level corruption
        words = text.split()
        corrupted_words = [self.corrupt_word(w, intensity, use_complex) for w in words]
        result = ' '.join(corrupted_words)
        
        # Add noise for medium+ intensity
        if intensity > 0.3:
            result = self.add_noise(result, intensity)
        
        # Add case chaos for high intensity
        if intensity > 0.6 and random.random() < 0.3:
            result = self.apply_case_chaos(result)
        
        return result


def generate_variants(
    slang_text: str,
    corruptor: V3LeetCorruptor,
    num_variants: int = 10
) -> list:
    """
    Generate multiple corruption variants of the same slang text.
    
    This is the "Multiplier Effect" - 1 LLM output → many training examples.
    
    Args:
        slang_text: Text from LLM slangification
        corruptor: Corruptor instance
        num_variants: Number of variants to generate
    
    Returns:
        List of (corrupted_text, intensity) tuples
    """
    variants = []
    
    # Define intensity distribution
    # 40% light, 35% medium, 25% heavy
    intensities = (
        [random.uniform(0.1, 0.3) for _ in range(int(num_variants * 0.4))] +
        [random.uniform(0.3, 0.6) for _ in range(int(num_variants * 0.35))] +
        [random.uniform(0.6, 0.9) for _ in range(num_variants - int(num_variants * 0.75))]
    )
    
    for intensity in intensities:
        corrupted = corruptor.corrupt(slang_text, intensity)
        
        # Skip if corruption didn't change anything
        if corrupted != slang_text:
            variants.append((corrupted, intensity))
    
    # Deduplicate while preserving order
    seen = set()
    unique_variants = []
    for variant, intensity in variants:
        if variant not in seen:
            seen.add(variant)
            unique_variants.append((variant, intensity))
    
    return unique_variants


def process_file(
    input_path: str,
    output_path: str,
    variants_per_sample: int = 10,
    max_samples: int = None
):
    """
    Process slang pairs and generate training data.
    
    Args:
        input_path: Path to slang_pairs.jsonl
        output_path: Path to save training_data_v3.jsonl
        variants_per_sample: Number of corruption variants per slang sample
        max_samples: Limit input samples (for testing)
    """
    print("=" * 60)
    print("V3 SCRIPT CORRUPTION (10x MULTIPLEX)")
    print("=" * 60)
    print(f"\n  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Variants per sample: {variants_per_sample}")
    
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"\n[ERROR] Input file not found: {input_path}")
        return
    
    corruptor = V3LeetCorruptor()
    
    # Load input
    print(f"\n[1/3] Loading slang pairs...")
    items = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    if max_samples:
        items = items[:max_samples]
    
    print(f"       Loaded {len(items)} slang pairs")
    print(f"       Expected output: ~{len(items) * variants_per_sample} training pairs")
    
    # Process
    print(f"\n[2/3] Generating corruption variants...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    intensity_dist = {'light': 0, 'medium': 0, 'heavy': 0}
    
    for i, item in enumerate(items):
        original = item.get('original', '')  # Clean English target
        slang = item.get('slang', '')  # LLM-generated slang
        
        if not original or not slang:
            continue
        
        # Generate variants
        variants = generate_variants(slang, corruptor, variants_per_sample)
        
        for corrupted, intensity in variants:
            pair = {
                'input_text': corrupted,
                'target_text': original,
            }
            all_pairs.append(json.dumps(pair))
            
            # Track intensity distribution
            if intensity < 0.3:
                intensity_dist['light'] += 1
            elif intensity < 0.6:
                intensity_dist['medium'] += 1
            else:
                intensity_dist['heavy'] += 1
        
        # Progress
        if (i + 1) % 5000 == 0:
            print(f"       Processed {i + 1}/{len(items)} samples, {len(all_pairs)} pairs generated")
    
    print(f"\n[3/3] Shuffling and saving...")
    
    random.shuffle(all_pairs)
    
    with open(output_file, 'w') as f:
        for pair in all_pairs:
            f.write(pair + '\n')
    
    # Stats
    total = len(all_pairs)
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Input samples:        {len(items)}")
    print(f"  Output training pairs: {total}")
    print(f"  Multiplier:           {total / len(items):.1f}x")
    print(f"\n  Intensity distribution:")
    print(f"    Light (0.1-0.3):  {intensity_dist['light']} ({intensity_dist['light']/total*100:.1f}%)")
    print(f"    Medium (0.3-0.6): {intensity_dist['medium']} ({intensity_dist['medium']/total*100:.1f}%)")
    print(f"    Heavy (0.6-0.9):  {intensity_dist['heavy']} ({intensity_dist['heavy']/total*100:.1f}%)")
    print(f"\n  Output: {output_path}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING PAIRS:")
    print("=" * 60)
    
    samples = random.sample(all_pairs, min(5, len(all_pairs)))
    for sample_json in samples:
        sample = json.loads(sample_json)
        print(f"\n  Input:  {sample['input_text']}")
        print(f"  Target: {sample['target_text']}")


def demo():
    """Demonstrate the corruption pipeline."""
    corruptor = V3LeetCorruptor()
    
    # Examples of slang text (as if from LLM)
    examples = [
        ("I really don't know what to do about this.", "idk wut to do bout this tbh"),
        ("See you later tonight at the party.", "cya l8r 2nite at da party"),
        ("Thanks for helping me, you are the best.", "thx 4 helpin me ur the best"),
        ("That game was really easy to win.", "that game was ez to win fr"),
        ("I am going to be late, wait for me.", "im gonna be l8 w8 4 me"),
    ]
    
    print("=" * 60)
    print("V3 CORRUPTION DEMO")
    print("=" * 60)
    
    for original, slang in examples:
        print(f"\n  Original: {original}")
        print(f"  LLM Slang: {slang}")
        print(f"  Corrupted variants:")
        
        variants = generate_variants(slang, corruptor, 5)
        for corrupted, intensity in variants:
            level = "light" if intensity < 0.3 else "medium" if intensity < 0.6 else "heavy"
            print(f"    [{level:6s}] {corrupted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Script Corruption")
    parser.add_argument("--input", "-i", default="../2_llm_corruption/slang_pairs.jsonl",
                        help="Input JSONL file with slang pairs")
    parser.add_argument("--output", "-o", default="training_data_v3.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--variants", "-v", type=int, default=10,
                        help="Variants per sample (default: 10)")
    parser.add_argument("--max", "-n", type=int, default=None,
                        help="Max input samples (for testing)")
    parser.add_argument("--demo", "-d", action="store_true",
                        help="Show demo examples")
    
    args = parser.parse_args()
    
    if args.demo:
        demo()
    else:
        process_file(args.input, args.output, args.variants, args.max)
