#!/usr/bin/env python3
"""
Ultimate Leetspeak Corruptor
============================
Transforms normal text into heavy leetspeak.

Usage:
    python corrupt_to_leetspeak.py --input raw_comments.jsonl --output corrupted_comments.jsonl
"""

import json
import random
import re
import argparse
from pathlib import Path


class LeetSpeakCorruptor:
    """
    Sophisticated leetspeak corruptor with multiple intensity levels.
    """

    def __init__(self, intensity: float = 0.7):
        """
        Args:
            intensity: 0.0 to 1.0, how aggressive the corruption is
        """
        self.intensity = intensity

        # ================================================================
        # CHARACTER MAPPINGS (extensive)
        # ================================================================
        self.char_map = {
            # Vowels
            'a': ['4', '@', '/\\', '^', 'α', '∂', 'Д'],
            'e': ['3', '€', '£', '&', 'ε', 'ë'],
            'i': ['1', '!', '|', 'ï', '¡', 'ı'],
            'o': ['0', '()', 'ø', 'Ø', 'ö', '°'],
            'u': ['v', 'µ', 'ü', 'ù', 'û'],

            # Consonants
            'b': ['8', '|3', 'ß', '|>', '13'],
            'c': ['(', '{', '[', '¢', '©'],
            'd': ['|)', '|]', 'đ', 'Ð'],
            'f': ['ph', '|=', 'ƒ'],
            'g': ['9', '6', 'gee', '&'],
            'h': ['#', '|-|', '}{', ')-(', '/-/'],
            'j': ['_|', '_/', ';'],
            'k': ['|<', '|{', '|('],
            'l': ['1', '|', '|_', '£', '¬'],
            'm': ['/\\/\\', '|v|', 'em', '|\\/|', '/|\\'],
            'n': ['|\\|', '/\\/', 'И', '|v'],
            'p': ['|*', '|>', '|D', '¶'],
            'q': ['9', '()_', 'kw'],
            'r': ['|2', '12', '®', 'Я'],
            's': ['5', '$', '§', 'z'],
            't': ['7', '+', '†', '|'],
            'v': ['\\/', '|/', '√'],
            'w': ['\\/\\/', 'vv', '\\^/', 'uu', 'Ш'],
            'x': ['><', '%', '}{', '×'],
            'y': ['`/', '¥', 'j', '`('],
            'z': ['2', '%', '7_', 'ž'],
        }

        # Simpler substitutions (more readable)
        self.simple_char_map = {
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$'],
            't': ['7', '+'],
            'l': ['1'],
            'b': ['8'],
            'g': ['9', '6'],
        }

        # ================================================================
        # WORD-LEVEL SUBSTITUTIONS
        # ================================================================
        self.word_map = {
            # === CRITICAL PHRASE-LEVEL SLANG (TEST SUITE FAILURES) ===
            'right now': ['rn'],
            'to be honest': ['tbh'],
            "i don't know": ['idk'],
            'i dont know': ['idk'],
            'as fuck': ['af'],
            'got to go': ['g2g', 'gtg'],
            'by the way': ['btw'],
            'oh my god': ['omg'],
            'no problem': ['np'],
            'in my opinion': ['imo'],
            'as far as i know': ['afaik'],
            'nevermind': ['nvm'],
            'cause': ['cuz', 'bc', 'cos'],
            'seriously': ['srsly'],
            'serious': ['srs'],
            'are you': ['r u', 'ru'],
            'see you': ['c u', 'cya', 'cu'],
            'good game': ['gg'],
            'well played': ['wp'],
            'easy': ['ez', '3z'],
            'real life': ['irl'],
            'for real': ['fr'],
            'no cap': ['nc'],
            'not gonna lie': ['ngl'],
            'good game well played': ['gg wp', 'ggwp'],
            # =========================================================

            # Common words
            'you': ['u', 'yu', 'yoo', 'j00'],
            'your': ['ur', 'yr', 'yer'],
            "you're": ['ur', 'youre', 'u r'],
            'are': ['r', 'ar', 'r'],
            'see': ['c', 'sea', 's33'],
            'be': ['b', 'bee', 'b3'],
            'to': ['2', 'too', '2'],
            'too': ['2', '2', 'tu'],
            'two': ['2', 'tu'],
            'for': ['4', 'fore', 'fr'],
            'fore': ['4', '4'],
            'before': ['b4', 'befor', 'bf'],
            'wait': ['w8', 'w4it'],
            'late': ['l8', 'l4te'],
            'later': ['l8r', 'l8er', 'latr'],
            'great': ['gr8', 'gre8t'],
            'hate': ['h8', 'h8te'],
            'mate': ['m8', 'm8te'],
            'ate': ['8', 'ayt'],
            'and': ['&', 'n', 'nd'],
            'the': ['da', 'teh', 'th3'],
            'this': ['dis', 'th1s', 'ths'],
            'that': ['dat', 'th4t', 'tht'],
            'what': ['wut', 'wat', 'wh4t'],
            'with': ['wit', 'w/', 'w1th'],
            'have': ['hav', 'hv', 'h4v3'],
            'has': ['haz', 'h4s'],
            'was': ['wuz', 'w4s'],
            'were': ['wer', 'wr'],
            'been': ['ben', 'b33n'],
            'being': ['bein', 'b31ng'],
            'just': ['jst', 'jus', 'ju5t'],
            'like': ['lyk', 'lik', 'l1k3'],
            'know': ['kno', 'no', 'kn0w'],
            'think': ['thnk', 'th1nk'],
            'because': ['cuz', 'coz', 'bc', 'bcuz'],
            'please': ['pls', 'plz', 'plox', 'pl34s3'],
            'thanks': ['thx', 'thnx', 'ty', 'th4nks'],
            'thank': ['thx', 'thnk', 'ty'],
            'okay': ['ok', 'k', 'okey'],
            'right': ['rite', 'r1ght', 'r1t3'],
            'night': ['nite', 'n1ght', 'n1t3'],
            'people': ['ppl', 'peeps', 'p30pl3'],
            'something': ['smth', 'sumthin', 's0m3th1ng'],
            'someone': ['sum1', 's0m30n3'],
            'everyone': ['every1', 'evry1', '3v3ry0n3'],
            'anyone': ['any1', 'ne1', '4ny0n3'],
            'nothing': ['nuthin', 'nothin', 'n0th1ng'],
            'everything': ['evrthing', 'evrythng', '3v3ryth1ng'],
            'about': ['bout', 'abt', '4b0ut'],
            'really': ['rly', 'rlly', 'r34lly'],
            'probably': ['prob', 'prolly', 'prly'],
            'actually': ['actly', 'acc', '4ctu4lly'],
            'definitely': ['def', 'deffo', 'd3f1n1t3ly'],
            'maybe': ['mayb', 'mby', 'm4yb3'],
            'sorry': ['sry', 'srry', 's0rry'],
            'hello': ['helo', 'hllo', 'h3ll0'],
            'hey': ['hay', 'h3y'],
            'good': ['gud', 'g00d', 'gd'],
            'cool': ['kool', 'c00l', 'cl'],
            'nice': ['n1c3', 'nyce', 'nic3'],
            'awesome': ['awsum', 'aw3s0m3', 'awsm'],
            'amazing': ['amzing', 'amzng', '4m4z1ng'],
            'love': ['luv', 'lov', 'l0v3'],
            'want': ['wnt', 'wan', 'w4nt'],
            'need': ['ned', 'nd', 'n33d'],
            'going': ['goin', 'gng', 'g01ng'],
            'come': ['cum', 'com', 'c0m3'],
            'coming': ['comin', 'cmng', 'c0m1ng'],
            'time': ['tym', 'tme', 't1m3'],
            'money': ['$', 'munny', 'm0n3y'],
            'work': ['wrk', 'werk', 'w0rk'],
            'game': ['gam', 'gaem', 'g4m3'],
            'play': ['pley', 'plai', 'pl4y'],
            'player': ['playr', 'plyr', 'pl4y3r'],
            'world': ['wrld', 'w0rld'],
            'never': ['nvr', 'nevr', 'n3v3r'],
            'always': ['alwys', 'alwayz', '4lw4ys'],
            'first': ['1st', 'frst', 'f1rst'],
            'second': ['2nd', 'scnd', 's3c0nd'],
            'third': ['3rd', 'thrd', 'th1rd'],

            # Gaming/internet terms
            'hacker': ['h4x0r', 'h4ck3r', 'haxor'],
            'hack': ['h4ck', 'h4x'],
            'noob': ['n00b', 'newb', 'nub', 'n0ob'],
            'newbie': ['n00b', 'newb', 'nub'],
            'owned': ['pwned', 'pwn3d', '0wn3d', 'ownd'],
            'own': ['pwn', '0wn'],
            'kill': ['k1ll', 'kil', 'pwn'],
            'killed': ['k1ll3d', 'pwned', 'rekt'],
            'wrecked': ['rekt', 'rek7', 'r3kt'],
            'elite': ['l33t', '1337', '3l1t3'],
            'leet': ['l33t', '1337'],
            'skills': ['sk1llz', 'skillz', 'sk1lls'],
            'skill': ['sk1ll', 'skil'],
            'pro': ['pr0', 'proe'],
            'gg': ['gg', 'gege'],

            # Abbreviations to keep/enhance
            'lol': ['l0l', 'lawl', 'lulz'],
            'rofl': ['r0fl', 'roflmao'],
            'omg': ['0mg', 'ohmygod'],
            'wtf': ['w7f', 'wut'],
        }

        # ================================================================
        # SUFFIX TRANSFORMATIONS
        # ================================================================
        self.suffix_map = {
            'ing': ['1ng', 'in', "in'", 'ng'],
            'tion': ['t10n', 'shun', 'shn'],
            'ness': ['n3ss', 'nes', 'nss'],
            'ment': ['m3nt', 'mnt'],
            'ight': ['1ght', 'ite', '1t3'],
            'ould': ['ud', 'ood'],
            'ough': ['uff', 'uf'],
            'ious': ['10us', 'ius'],
            'eous': ['30us', 'eus'],
            'ness': ['n3ss', 'nes'],
            'less': ['l3ss', 'les'],
            'able': ['4bl3', 'abl'],
            'ible': ['1bl3', 'ibl'],
        }

        # ================================================================
        # RANDOM NOISE INSERTIONS
        # ================================================================
        self.noise_chars = ['x', 'z', '_', '-', '.']

    def corrupt_char(self, char: str, use_simple: bool = False) -> str:
        """Replace a single character with leetspeak equivalent."""
        lower = char.lower()

        if random.random() > self.intensity:
            return char

        if use_simple and lower in self.simple_char_map:
            replacement = random.choice(self.simple_char_map[lower])
        elif lower in self.char_map:
            replacement = random.choice(self.char_map[lower])
        else:
            return char

        # Preserve case for first character of replacement
        if char.isupper() and len(replacement) > 0:
            return replacement[0].upper() + replacement[1:]
        return replacement

    def corrupt_word(self, word: str) -> str:
        """Corrupt a single word."""
        lower = word.lower()

        # Check for word-level replacement
        if lower in self.word_map and random.random() < self.intensity:
            replacement = random.choice(self.word_map[lower])
            # Preserve capitalization
            if word[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        # Check for suffix replacement
        for suffix, replacements in self.suffix_map.items():
            if lower.endswith(suffix) and random.random() < self.intensity * 0.5:
                new_suffix = random.choice(replacements)
                return word[:-len(suffix)] + new_suffix

        # Character-level corruption
        use_simple = random.random() < 0.6  # 60% chance of using simpler substitutions
        result = []
        for char in word:
            result.append(self.corrupt_char(char, use_simple))

        return ''.join(result)

    def add_noise(self, text: str) -> str:
        """Add random noise like repeated chars, random insertions."""
        words = text.split()
        result = []

        for word in words:
            # Randomly extend characters (10% chance)
            if random.random() < 0.1 * self.intensity:
                pos = random.randint(0, max(0, len(word) - 1))
                char = word[pos]
                if char.isalpha():
                    word = word[:pos] + char * random.randint(2, 4) + word[pos+1:]

            result.append(word)

        return ' '.join(result)

    def apply_case_chaos(self, text: str) -> str:
        """Apply random case changes."""
        if random.random() < 0.1:  # 10% chance of ALL CAPS
            return text.upper()

        if random.random() < 0.15:  # 15% chance of aLtErNaTiNg
            result = []
            upper = random.choice([True, False])
            for char in text:
                if char.isalpha():
                    result.append(char.upper() if upper else char.lower())
                    upper = not upper
                else:
                    result.append(char)
            return ''.join(result)

        return text

    def corrupt(self, text: str) -> str:
        """
        Main corruption function. Transforms clean text into leetspeak.

        Args:
            text: Clean English text

        Returns:
            Corrupted leetspeak version
        """
        result = text

        # 1. PHRASE-LEVEL REPLACEMENTS FIRST (multi-word -> abbreviation)
        text_lower = result.lower()
        for phrase, replacements in self.word_map.items():
            if ' ' in phrase:  # Only multi-word phrases
                if phrase in text_lower and random.random() < self.intensity:
                    replacement = random.choice(replacements)
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    result = pattern.sub(replacement, result, count=1)
                    text_lower = result.lower()  # Update for next iteration

        # 2. WORD-LEVEL CORRUPTIONS
        words = result.split()
        corrupted_words = [self.corrupt_word(word) for word in words]
        result = ' '.join(corrupted_words)

        # 3. ADD NOISE
        result = self.add_noise(result)

        # 4. APPLY CASE CHAOS
        result = self.apply_case_chaos(result)

        return result


def process_file(input_path: str, output_path: str, intensity: float = 0.7):
    """
    Process a JSONL file and corrupt all text entries.
    """
    corruptor = LeetSpeakCorruptor(intensity=intensity)

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"Error: {input_path} not found")
        return

    processed = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line.strip())
                original_text = data.get('text', '')

                # Corrupt the text
                corrupted_text = corruptor.corrupt(original_text)

                # Save corrupted version
                output_data = {
                    'original': original_text,
                    'corrupted': corrupted_text,
                    'subreddit': data.get('subreddit', ''),
                }
                f_out.write(json.dumps(output_data) + '\n')

                processed += 1
                if processed % 500 == 0:
                    print(f"Processed {processed} comments...")

            except json.JSONDecodeError:
                continue

    print(f"\n✓ Done! Processed {processed} comments")
    print(f"Output: {output_file}")


def demo():
    """Show some examples of corruption."""
    corruptor = LeetSpeakCorruptor(intensity=0.8)

    examples = [
        "Hello world, how are you doing today?",
        "I dont know what to do to be honest",
        "See you later tonight at the party",
        "Thanks for the help, you are the best!",
        "The hacker owned that noob so hard",
        "Please wait for me before you leave",
        "I think this is really awesome",
        "Good game well played everyone",
    ]

    print("=" * 60)
    print("LEETSPEAK CORRUPTOR DEMO")
    print("=" * 60)

    for text in examples:
        corrupted = corruptor.corrupt(text)
        print(f"\nOriginal:  {text}")
        print(f"Corrupted: {corrupted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Corrupt text to leetspeak")
    parser.add_argument('--input', '-i', help="Input JSONL file")
    parser.add_argument('--output', '-o', help="Output JSONL file")
    parser.add_argument('--intensity', '-n', type=float, default=0.7,
                        help="Corruption intensity 0.0-1.0 (default: 0.7)")
    parser.add_argument('--demo', '-d', action='store_true',
                        help="Show demo examples")

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.input and args.output:
        process_file(args.input, args.output, args.intensity)
    else:
        print("Usage:")
        print("  Demo:    python corrupt_to_leetspeak.py --demo")
        print("  Process: python corrupt_to_leetspeak.py -i input.jsonl -o output.jsonl")
