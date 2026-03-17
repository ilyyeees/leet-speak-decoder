#!/usr/bin/env python3
"""
V3 Script Corruption (Enhanced)
================================
Takes slangified text and applies VISUAL leetspeak corruption + realistic typos.
This is the "mechanical" layer that adds character swaps, noise, typos, etc.

Key features:
1. NO phrase/word replacements (LLM already did that)
2. Intensity-driven char_map corruption
3. Digit protection (won't corrupt existing numbers like "2nite")
4. 15x multiplexing per input (different intensities/seeds)
5. QWERTY adjacency typos (fat finger)
6. Transpositions & character drops (teh, tday)
7. Realistic case patterns (caps lock accidents, shift laziness)
8. Structure protection (URLs, @mentions, emails preserved)
9. Modern internet patterns (repeated chars, missing spaces, punctuation spam)

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
from typing import Tuple, List, Optional


# =============================================================================
# PROTECTION PATTERNS (URLs, emails, @mentions, code)
# =============================================================================
PROTECTED_PATTERNS = [
    re.compile(r'https?://\S+'),                          # URLs
    re.compile(r'www\.\S+'),                              # www links
    re.compile(r'\S+@\S+\.\S+'),                          # Emails
    re.compile(r'@\w+'),                                  # @mentions
    re.compile(r'#\w+'),                                  # #hashtags
    re.compile(r'`[^`]+`'),                               # Inline code
    re.compile(r'\$[\d,.]+'),                             # Money amounts
    re.compile(r'\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\b', re.I),  # Times
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),          # Dates
]


def extract_protected(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract protected patterns and replace with placeholders.
    Returns modified text and list of (placeholder, original) tuples.
    """
    protected = []
    result = text

    for i, pattern in enumerate(PROTECTED_PATTERNS):
        for j, match in enumerate(pattern.finditer(result)):
            placeholder = f"__PROTECTED_{i}_{j}__"
            original = match.group()
            protected.append((placeholder, original))
            result = result.replace(original, placeholder, 1)

    return result, protected


def restore_protected(text: str, protected: List[Tuple[str, str]]) -> str:
    """Restore protected patterns from placeholders (case-insensitive matching)."""
    result = text
    for placeholder, original in protected:
        # Case-insensitive replacement since case chaos might corrupt the placeholder
        pattern = re.compile(re.escape(placeholder), re.IGNORECASE)
        result = pattern.sub(original, result)
    return result


class V3LeetCorruptor:
    """
    Enhanced leetspeak corruptor for V3.
    Handles visual corruption (char swaps), typos, and modern internet patterns.
    Word-level slang was already handled by the LLM.
    """

    def __init__(self):
        # ================================================================
        # QWERTY ADJACENCY MAP (for fat-finger typos)
        # ================================================================
        self.qwerty_adjacent = {
            'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg',
            'y': 'tugh', 'u': 'yijh', 'i': 'uokj', 'o': 'iplk', 'p': 'ol',
            'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
            'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
            'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
            'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
        }

        # ================================================================
        # CHARACTER MAPPINGS (visual leetspeak)
        # ================================================================

        # Simple/readable swaps (use more often)
        self.simple_map = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$', 'z'],
            't': ['7', '+'],
            'l': ['1'],
            'b': ['8'],
            'g': ['9', '6'],
        }

        # Complex swaps (RARE - only at very high intensity > 0.8)
        self.complex_map = {
            'a': ['4', '@'],           # Simplified from /\
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            'b': ['8', '|3'],
            'c': ['(', '<'],
            'd': ['|)'],
            'h': ['|-|'],              # Keep one complex H
            'k': ['|<'],               # Keep one complex K
            'l': ['1', '|'],
            'n': ['|\\|'],             # Keep one complex N
            's': ['5', '$', 'z'],
            't': ['7', '+'],
            'g': ['9', '6'],
        }

        # ================================================================
        # MEME SPELLINGS (modern internet culture)
        # ================================================================
        self.meme_spellings = {
            'boy': 'boi',
            'small': 'smol',
            'thick': 'thicc',
            'good': 'gud',
            'friend': 'fren',
            'dog': 'doggo',
            'cat': 'catto',
            'what': 'wut',
            'yes': 'yus',
            'no': 'nah',
            'the': 'da',
            'though': 'tho',
            'brother': 'bro',
            'sister': 'sis',
            'something': 'smth',
            'anything': 'anythin',
            'nothing': 'nothin',
            'going': 'goin',
            'doing': 'doin',
            'pretty': 'pritty',
            'really': 'rly',
            'about': 'bout',
            'because': 'cuz',
            'probably': 'prolly',
            'definitely': 'def',
            'literally': 'legit',
            'actually': 'actly',
        }

        # ================================================================
        # COMMON DOUBLE LETTERS (for drop simulation)
        # ================================================================
        self.double_letters = ['ll', 'ss', 'ee', 'oo', 'tt', 'ff', 'rr', 'nn', 'mm', 'pp', 'cc', 'dd', 'gg']

        # ================================================================
        # APOSTROPHE CONTRACTIONS (to drop apostrophes)
        # ================================================================
        self.contractions = {
            "don't": "dont",
            "can't": "cant",
            "won't": "wont",
            "wouldn't": "wouldnt",
            "couldn't": "couldnt",
            "shouldn't": "shouldnt",
            "isn't": "isnt",
            "aren't": "arent",
            "wasn't": "wasnt",
            "weren't": "werent",
            "hasn't": "hasnt",
            "haven't": "havent",
            "hadn't": "hadnt",
            "doesn't": "doesnt",
            "didn't": "didnt",
            "i'm": "im",
            "you're": "youre",
            "we're": "were",
            "they're": "theyre",
            "he's": "hes",
            "she's": "shes",
            "it's": "its",
            "that's": "thats",
            "what's": "whats",
            "there's": "theres",
            "here's": "heres",
            "who's": "whos",
            "i've": "ive",
            "you've": "youve",
            "we've": "weve",
            "they've": "theyve",
            "i'll": "ill",
            "you'll": "youll",
            "we'll": "well",
            "they'll": "theyll",
            "he'll": "hell",
            "she'll": "shell",
            "it'll": "itll",
            "i'd": "id",
            "you'd": "youd",
            "we'd": "wed",
            "they'd": "theyd",
            "let's": "lets",
        }

    # ================================================================
    # HELPER METHODS
    # ================================================================

    def has_digit(self, word: str) -> bool:
        """Check if word already contains digits (from LLM like '2nite', 'l8r')."""
        return bool(re.search(r'\d', word))

    def is_placeholder(self, word: str) -> bool:
        """Check if word is a protection placeholder."""
        return word.startswith('__PROTECTED_') and word.endswith('__')

    # ================================================================
    # TYPO METHODS (QWERTY, transposition, drops)
    # ================================================================

    def apply_qwerty_typo(self, word: str, intensity: float) -> str:
        """Apply fat-finger QWERTY adjacency typos."""
        if len(word) < 3 or random.random() > intensity * 0.15:
            return word

        result = list(word)
        # Pick 1-2 positions to typo
        num_typos = 1 if random.random() > 0.3 else 2
        positions = random.sample(range(len(word)), min(num_typos, len(word)))

        for pos in positions:
            char = result[pos].lower()
            if char in self.qwerty_adjacent:
                adjacent = self.qwerty_adjacent[char]
                typo_char = random.choice(adjacent)
                # Preserve case
                if result[pos].isupper():
                    typo_char = typo_char.upper()
                result[pos] = typo_char

        return ''.join(result)

    def apply_transposition(self, word: str, intensity: float) -> str:
        """Swap adjacent characters (teh, hte, etc.)."""
        if len(word) < 3 or random.random() > intensity * 0.12:
            return word

        result = list(word)
        # Pick a position to swap (not first or last usually)
        pos = random.randint(0, len(word) - 2)
        result[pos], result[pos + 1] = result[pos + 1], result[pos]

        return ''.join(result)

    def apply_char_drop(self, word: str, intensity: float) -> str:
        """Drop a character (tday, gona, etc.)."""
        if len(word) < 4 or random.random() > intensity * 0.10:
            return word

        # Prefer dropping vowels or repeated chars
        vowel_positions = [i for i, c in enumerate(word) if c.lower() in 'aeiou' and i > 0]

        if vowel_positions and random.random() < 0.7:
            pos = random.choice(vowel_positions)
        else:
            pos = random.randint(1, len(word) - 1)  # Don't drop first char

        return word[:pos] + word[pos + 1:]

    def apply_double_letter_drop(self, word: str, intensity: float) -> str:
        """Drop one letter from double letters (realy, tomorow)."""
        if random.random() > intensity * 0.15:
            return word

        word_lower = word.lower()
        for double in self.double_letters:
            if double in word_lower:
                # Find position and drop one
                pos = word_lower.find(double)
                return word[:pos] + word[pos + 1:]

        return word

    def apply_vowel_drop(self, word: str, intensity: float) -> str:
        """Drop vowels for texting shorthand (pls, msg, txt)."""
        if len(word) < 4 or random.random() > intensity * 0.08:
            return word

        # Only drop internal vowels, keep first and last
        result = [word[0]]
        for i, char in enumerate(word[1:-1], 1):
            if char.lower() in 'aeiou' and random.random() < 0.5:
                continue
            result.append(char)
        result.append(word[-1])

        return ''.join(result)

    # ================================================================
    # SPACING & PUNCTUATION METHODS
    # ================================================================

    def apply_missing_spaces(self, words: List[str], intensity: float) -> List[str]:
        """Concatenate some adjacent words (imgoing, dontknow)."""
        if len(words) < 2 or random.random() > intensity * 0.08:
            return words

        result = []
        i = 0
        while i < len(words):
            if i < len(words) - 1 and random.random() < 0.15 * intensity:
                # Merge with next word
                merged = words[i] + words[i + 1]
                if len(merged) < 15:  # Don't create super long words
                    result.append(merged)
                    i += 2
                    continue
            result.append(words[i])
            i += 1

        return result

    def apply_extra_spaces(self, text: str, intensity: float) -> str:
        """Add extra spaces (mobile typing artifact)."""
        if random.random() > intensity * 0.05:
            return text

        words = text.split()
        result = []
        for word in words:
            result.append(word)
            if random.random() < 0.1:
                result.append('')  # Creates double space when joined

        return ' '.join(result)

    def apply_punctuation_spacing(self, text: str, intensity: float) -> str:
        """Add space before punctuation (mobile autocorrect artifact)."""
        if random.random() > intensity * 0.06:
            return text

        # Add space before some punctuation
        for punct in ['!', '?', '.', ',']:
            if punct in text and random.random() < 0.3:
                text = text.replace(punct, f' {punct}', 1)

        return text

    def apply_punctuation_spam(self, text: str, intensity: float) -> str:
        """Repeat punctuation for emphasis (!!!, ???, ....)."""
        if random.random() > intensity * 0.10:
            return text

        # Find ending punctuation
        if text.endswith('!'):
            repeats = random.randint(2, 4)
            text = text[:-1] + '!' * repeats
        elif text.endswith('?'):
            repeats = random.randint(2, 3)
            text = text[:-1] + '?' * repeats
        elif text.endswith('.'):
            if random.random() < 0.5:
                text = text[:-1] + '...'

        return text

    def apply_missing_apostrophe(self, text: str, intensity: float) -> str:
        """Remove apostrophes from contractions."""
        if random.random() > intensity * 0.25:
            return text

        text_lower = text.lower()
        for contraction, replacement in self.contractions.items():
            if contraction in text_lower:
                # Case-insensitive replace
                pattern = re.compile(re.escape(contraction), re.IGNORECASE)
                text = pattern.sub(replacement, text, count=1)

        return text

    # ================================================================
    # CHARACTER EMPHASIS & REPETITION
    # ================================================================

    def apply_char_repetition(self, word: str, intensity: float) -> str:
        """Repeat characters for emphasis (yesss, nooo, pleaseee)."""
        if len(word) < 3 or random.random() > intensity * 0.12:
            return word

        # Usually repeat the last char or a vowel
        result = list(word)

        if random.random() < 0.6:
            # Repeat last character
            if result[-1].isalpha():
                repeats = random.randint(2, 4)
                result[-1] = result[-1] * repeats
        else:
            # Repeat a vowel
            vowel_positions = [i for i, c in enumerate(word) if c.lower() in 'aeiou']
            if vowel_positions:
                pos = random.choice(vowel_positions)
                repeats = random.randint(2, 3)
                result[pos] = result[pos] * repeats

        return ''.join(result)

    # ================================================================
    # MEME SPELLING
    # ================================================================

    def apply_meme_spelling(self, word: str, intensity: float) -> str:
        """Apply modern meme spellings."""
        if random.random() > intensity * 0.15:
            return word

        word_lower = word.lower()
        if word_lower in self.meme_spellings:
            replacement = self.meme_spellings[word_lower]
            # Preserve first letter case
            if word[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            return replacement

        return word

    # ================================================================
    # LEETSPEAK CHARACTER CORRUPTION
    # ================================================================

    def corrupt_char(self, char: str, intensity: float, use_complex: bool) -> str:
        """
        Replace a single character with leetspeak.

        Args:
            char: Character to potentially replace
            intensity: 0.0-1.0, probability of replacement
            use_complex: If True, use complex character map (very rare)
        """
        if random.random() > intensity:
            return char

        lower = char.lower()

        # Complex map only at EXTREMELY high intensity (> 0.9) and rarely (15% chance)
        # This makes |\| for N, |-| for H, etc. VERY rare
        if use_complex and intensity > 0.9 and random.random() < 0.15:
            char_map = self.complex_map
        else:
            char_map = self.simple_map

        if lower in char_map:
            replacement = random.choice(char_map[lower])
            # Preserve case for first char
            if char.isupper() and len(replacement) > 0:
                return replacement[0].upper() + replacement[1:]
            return replacement

        return char

    def apply_phonetic_hacks(self, word: str) -> str:
        """Apply simple phonetic shortcuts common in leetspeak."""
        word_lower = word.lower()

        # 'cks' -> 'x' (hacks -> hax, rocks -> rox)
        if word_lower.endswith('cks'):
            return word[:-3] + 'x'

        # ending 's' -> 'z' (skills -> skillz)
        if word_lower.endswith('s') and len(word) > 3:
            if random.random() < 0.5:
                return word[:-1] + 'z'

        return word

    def apply_partial_corruption(self, word: str, intensity: float, use_complex: bool) -> str:
        """
        Corrupt only part of the word (more realistic than uniform corruption).
        """
        if len(word) < 4:
            # Short words: corrupt fully
            return ''.join(self.corrupt_char(c, intensity, use_complex) for c in word)

        # For longer words, randomly choose a corruption strategy
        strategy = random.choice(['full', 'first_half', 'second_half', 'sparse'])

        if strategy == 'full':
            return ''.join(self.corrupt_char(c, intensity, use_complex) for c in word)

        elif strategy == 'first_half':
            mid = len(word) // 2
            corrupted = ''.join(self.corrupt_char(c, intensity, use_complex) for c in word[:mid])
            return corrupted + word[mid:]

        elif strategy == 'second_half':
            mid = len(word) // 2
            corrupted = ''.join(self.corrupt_char(c, intensity, use_complex) for c in word[mid:])
            return word[:mid] + corrupted

        else:  # sparse - only corrupt every 2nd or 3rd char
            result = []
            for i, char in enumerate(word):
                if i % random.randint(2, 3) == 0:
                    result.append(self.corrupt_char(char, intensity, use_complex))
                else:
                    result.append(char)
            return ''.join(result)

    def corrupt_word(self, word: str, intensity: float, use_complex: bool) -> str:
        """
        Apply all word-level corruptions.
        """
        # Skip placeholders
        if self.is_placeholder(word):
            return word

        # Only skip words that already have digits from LLM (like '2nite', 'l8r')
        if self.has_digit(word):
            return word

        # 1. Maybe apply meme spelling first
        word = self.apply_meme_spelling(word, intensity)

        # 2. Phonetic hacks (structure changes)
        if intensity > 0.3:
            word = self.apply_phonetic_hacks(word)

        # 3. Typo layer (pick one type of typo at most)
        typo_roll = random.random()
        if typo_roll < 0.25:
            word = self.apply_qwerty_typo(word, intensity)
        elif typo_roll < 0.45:
            word = self.apply_transposition(word, intensity)
        elif typo_roll < 0.60:
            word = self.apply_char_drop(word, intensity)
        elif typo_roll < 0.72:
            word = self.apply_double_letter_drop(word, intensity)
        elif typo_roll < 0.82:
            word = self.apply_vowel_drop(word, intensity)

        # 4. Character repetition for emphasis
        word = self.apply_char_repetition(word, intensity)

        # 5. Apply leetspeak character corruption (partial for realism)
        if random.random() < 0.7:
            word = self.apply_partial_corruption(word, intensity, use_complex)
        else:
            # Full corruption sometimes
            result = []
            for char in word:
                result.append(self.corrupt_char(char, intensity, use_complex))
            word = ''.join(result)

        return word

    # ================================================================
    # CASE CORRUPTION
    # ================================================================

    def apply_case_chaos(self, text: str, intensity: float) -> str:
        """
        Apply realistic case variations.
        """
        roll = random.random()

        # Caps lock accident (tHANKS, hELLO) - first letter lower, rest upper
        if roll < 0.08:
            words = text.split()
            result = []
            for word in words:
                if len(word) > 1 and word[0].isupper():
                    result.append(word[0].lower() + word[1:].upper())
                else:
                    result.append(word)
            return ' '.join(result)

        # Shift key laziness - all lowercase (very common)
        if roll < 0.35:
            return text.lower()

        # ALL CAPS (angry/emphasis)
        if roll < 0.42:
            return text.upper()

        # aLtErNaTiNg CaSe (spongebob mocking)
        if roll < 0.48:
            result = []
            upper = random.choice([True, False])
            for char in text:
                if char.isalpha():
                    result.append(char.upper() if upper else char.lower())
                    upper = not upper
                else:
                    result.append(char)
            return ''.join(result)

        # Random caps on some words
        if roll < 0.55:
            words = text.split()
            result = []
            for word in words:
                if random.random() < 0.3:
                    result.append(word.upper())
                else:
                    result.append(word.lower())
            return ' '.join(result)

        # No change
        return text

    # ================================================================
    # NOISE
    # ================================================================

    def add_noise(self, text: str, intensity: float) -> str:
        """Add random character noise."""
        if random.random() > intensity * 0.3:
            return text

        words = text.split()
        result = []

        for word in words:
            if self.has_digit(word) or self.is_placeholder(word):
                result.append(word)
                continue

            # Random punctuation insertion (rare)
            if random.random() < 0.03 * intensity:
                pos = random.randint(0, len(word))
                punct = random.choice(".,!?")
                word = word[:pos] + punct + word[pos:]

            result.append(word)

        return ' '.join(result)

    # ================================================================
    # MAIN CORRUPTION FUNCTION
    # ================================================================

    def corrupt(self, text: str, intensity: float = 0.5) -> str:
        """
        Main corruption function.

        Args:
            text: Slang text to corrupt further
            intensity: 0.0-1.0, controls how aggressive the corruption is
                      0.0-0.3 = light (some char swaps, minor typos)
                      0.3-0.6 = medium (more swaps, typos, some spacing issues)
                      0.6-1.0 = heavy (lots of swaps, typos, case chaos)

        Returns:
            Corrupted leetspeak text
        """
        # Extract and protect URLs, emails, mentions
        text, protected = extract_protected(text)

        # Complex leet only at very high intensity
        use_complex = intensity > 0.8

        # 1. Apply missing apostrophes early (affects whole text)
        text = self.apply_missing_apostrophe(text, intensity)

        # 2. Word-level corruption
        words = text.split()
        corrupted_words = [self.corrupt_word(w, intensity, use_complex) for w in words]

        # 3. Apply missing spaces (word concatenation)
        if intensity > 0.4:
            corrupted_words = self.apply_missing_spaces(corrupted_words, intensity)

        result = ' '.join(corrupted_words)

        # 4. Add extra spaces occasionally
        if intensity > 0.3:
            result = self.apply_extra_spaces(result, intensity)

        # 5. Punctuation modifications
        result = self.apply_punctuation_spacing(result, intensity)
        result = self.apply_punctuation_spam(result, intensity)

        # 6. Add noise for medium+ intensity
        if intensity > 0.3:
            result = self.add_noise(result, intensity)

        # 7. Add case chaos for medium+ intensity
        if intensity > 0.4 and random.random() < 0.35:
            result = self.apply_case_chaos(result, intensity)

        # 8. Restore protected patterns
        result = restore_protected(result, protected)

        return result


def generate_variants(
    slang_text: str,
    corruptor: V3LeetCorruptor,
    num_variants: int = 15
) -> list:
    """
    Generate multiple corruption variants of the same slang text.

    This is the "Multiplier Effect" - 1 LLM output → many training examples.

    Args:
        slang_text: Text from LLM slangification
        corruptor: Corruptor instance
        num_variants: Number of variants to generate (default: 15)

    Returns:
        List of (corrupted_text, intensity) tuples
    """
    variants = []

    # Define intensity distribution across the spectrum
    # 50% light, 45% medium, 5% heavy - mostly readable corruption
    num_light = int(num_variants * 0.50)
    num_medium = int(num_variants * 0.45)
    num_heavy = num_variants - num_light - num_medium

    intensities = (
        [random.uniform(0.1, 0.3) for _ in range(num_light)] +
        [random.uniform(0.3, 0.6) for _ in range(num_medium)] +
        [random.uniform(0.6, 0.95) for _ in range(num_heavy)]
    )

    # Shuffle intensities so we don't always process in order
    random.shuffle(intensities)

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
    variants_per_sample: int = 15,
    max_samples: int = None
):
    """
    Process slang pairs and generate training data.

    Args:
        input_path: Path to slang_pairs.jsonl
        output_path: Path to save training_data_v3.jsonl
        variants_per_sample: Number of corruption variants per slang sample (default: 15)
        max_samples: Limit input samples (for testing)
    """
    print("=" * 60)
    print("V3 SCRIPT CORRUPTION (ENHANCED 15x MULTIPLEX)")
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
    """Demonstrate the enhanced corruption pipeline."""
    corruptor = V3LeetCorruptor()

    # Examples of slang text (as if from LLM)
    examples = [
        ("I really don't know what to do about this.", "idk wut to do bout this tbh"),
        ("See you later tonight at the party.", "cya l8r 2nite at da party"),
        ("Thanks for helping me, you are the best.", "thx 4 helpin me ur the best"),
        ("That game was really easy to win.", "that game was ez to win fr"),
        ("I am going to be late, wait for me.", "im gonna be l8 w8 4 me"),
        ("Check out my profile at https://example.com/user", "check out my profile at https://example.com/user"),
        ("Send me a message @username please", "send me a msg @username pls"),
    ]

    print("=" * 60)
    print("V3 ENHANCED CORRUPTION DEMO")
    print("=" * 60)
    print("\nFeatures: QWERTY typos, transpositions, char drops, meme spellings,")
    print("          missing apostrophes, spacing issues, case chaos, URL protection")

    for original, slang in examples:
        print(f"\n{'─' * 60}")
        print(f"  Original: {original}")
        print(f"  LLM Slang: {slang}")
        print(f"  Corrupted variants:")

        variants = generate_variants(slang, corruptor, 8)
        for corrupted, intensity in variants:
            level = "light" if intensity < 0.3 else "medium" if intensity < 0.6 else "heavy"
            print(f"    [{level:6s} {intensity:.2f}] {corrupted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 Enhanced Script Corruption")
    parser.add_argument("--input", "-i", default="../2_llm_corruption/slang_pairs.jsonl",
                        help="Input JSONL file with slang pairs")
    parser.add_argument("--output", "-o", default="training_data_v3.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--variants", "-v", type=int, default=15,
                        help="Variants per sample (default: 15)")
    parser.add_argument("--max", "-n", type=int, default=None,
                        help="Max input samples (for testing)")
    parser.add_argument("--demo", "-d", action="store_true",
                        help="Show demo examples")

    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        process_file(args.input, args.output, args.variants, args.max)
