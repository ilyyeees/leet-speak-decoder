#!/usr/bin/env python3
"""
================================================================================
V3 COMPREHENSIVE TEST SUITE
================================================================================

Evaluate the v3 model with extensive test cases covering:
- Basic leetspeak decoding
- Edge cases (2/to/too, number preservation)
- Different intensity levels
- Gaming/Gen-Z slang
- Adversarial examples
- Regression tests

Run after training to validate model quality.

Usage:
    python test_suite_v3.py --model ./byt5_leetspeak_v3
    python test_suite_v3.py --model ./byt5_leetspeak_v3 --verbose
    python test_suite_v3.py --model ./byt5_leetspeak_v3 --export results.json

================================================================================
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


# ============================================================================
# TEST CASES
# ============================================================================

@dataclass
class TestCase:
    """Single test case with input, expected output, and metadata."""
    input: str
    expected: str
    category: str
    difficulty: str  # easy, medium, hard, very_hard
    description: str = ""

    def __hash__(self):
        return hash((self.input, self.expected))


# Comprehensive test cases organized by category
TEST_CASES: List[TestCase] = [
    # =========================================================================
    # BASIC LEETSPEAK (Character Substitutions)
    # =========================================================================
    TestCase("h3ll0 w0rld", "Hello world", "basic_leet", "easy", "Simple vowel substitutions"),
    TestCase("th4nk y0u", "Thank you", "basic_leet", "easy", "Common greetings"),
    TestCase("g00d m0rn1ng", "Good morning", "basic_leet", "easy", "Morning greeting"),
    TestCase("h0w 4r3 y0u", "How are you", "basic_leet", "easy", "Basic question"),
    TestCase("th15 15 c00l", "This is cool", "basic_leet", "easy", "Simple sentence"),
    TestCase("1 l0v3 y0u", "I love you", "basic_leet", "easy", "Emotional phrase"),
    TestCase("pl34s3 h3lp m3", "Please help me", "basic_leet", "easy", "Request"),
    TestCase("y3s 0r n0", "Yes or no", "basic_leet", "easy", "Simple question"),
    TestCase("g00d n1ght", "Good night", "basic_leet", "easy", "Evening greeting"),
    TestCase("s33 y0u l4t3r", "See you later", "basic_leet", "easy", "Farewell"),

    # =========================================================================
    # WORD-LEVEL SUBSTITUTIONS
    # =========================================================================
    TestCase("thx 4 the help", "Thanks for the help", "word_subs", "easy", "Common abbreviations"),
    TestCase("c u l8r", "See you later", "word_subs", "easy", "Classic SMS style"),
    TestCase("r u ok", "Are you okay", "word_subs", "easy", "Simple question"),
    TestCase("b4 u go", "Before you go", "word_subs", "easy", "Before abbreviation"),
    TestCase("gr8 job m8", "Great job mate", "word_subs", "medium", "8 suffix words"),
    TestCase("w8 4 me", "Wait for me", "word_subs", "medium", "Wait abbreviation"),
    TestCase("wut r u doin", "What are you doing", "word_subs", "medium", "Phonetic spelling"),
    TestCase("cya 2nite", "See you tonight", "word_subs", "medium", "Tonight abbreviation"),
    TestCase("gonna b l8", "Going to be late", "word_subs", "medium", "Multiple contractions"),
    TestCase("prolly wont make it", "Probably won't make it", "word_subs", "medium", "Phonetic spelling"),

    # =========================================================================
    # THE 2/TO/TOO EDGE CASE (Critical!)
    # =========================================================================
    TestCase("1t5 2 l8", "It is too late", "2_to_too", "hard", "Too late pattern"),
    TestCase("thats 2 much", "That's too much", "2_to_too", "hard", "Too much pattern"),
    TestCase("im going 2 sleep", "I'm going to sleep", "2_to_too", "hard", "Going to pattern"),
    TestCase("i have 2 go", "I have to go", "2_to_too", "hard", "Have to pattern"),
    TestCase("want 2 play", "Want to play", "2_to_too", "hard", "Want to pattern"),
    TestCase("need 2 talk 2 u", "Need to talk to you", "2_to_too", "hard", "Multiple 'to'"),
    TestCase("2 hot 2 handle", "Too hot to handle", "2_to_too", "very_hard", "Mixed too/to"),
    TestCase("trying 2 b 2 cool", "Trying to be too cool", "2_to_too", "very_hard", "Multiple mixed"),
    TestCase("not 2 bad", "Not too bad", "2_to_too", "hard", "Not too pattern"),
    TestCase("way 2 ez", "Way too easy", "2_to_too", "hard", "Way too pattern"),

    # =========================================================================
    # NUMBER PRESERVATION (Critical!)
    # =========================================================================
    TestCase("I have 2 cats", "I have 2 cats", "number_preserve", "medium", "Standalone 2"),
    TestCase("Meet at 3 PM", "Meet at 3 PM", "number_preserve", "easy", "Time format"),
    TestCase("I got 100 points", "I got 100 points", "number_preserve", "easy", "Score"),
    TestCase("Born in 1990", "Born in 1990", "number_preserve", "easy", "Year"),
    TestCase("Version 2.0", "Version 2.0", "number_preserve", "easy", "Version number"),
    TestCase("Room 404", "Room 404", "number_preserve", "easy", "Room number"),
    TestCase("Page 42", "Page 42", "number_preserve", "easy", "Page number"),
    TestCase("25 years old", "25 years old", "number_preserve", "easy", "Age"),
    TestCase("at 5 o'clock", "At 5 o'clock", "number_preserve", "easy", "Time with o'clock"),
    TestCase("$100 dollars", "$100 dollars", "number_preserve", "easy", "Price"),

    # =========================================================================
    # MIXED NUMBER/LEET CASES (Very Hard!)
    # =========================================================================
    TestCase("1 h4v3 2 c4ts", "I have 2 cats", "mixed_numbers", "very_hard", "Leet + number 2"),
    TestCase("m33t m3 4t 3 pm", "Meet me at 3 PM", "mixed_numbers", "very_hard", "Leet + time"),
    TestCase("1 g0t 100 p01nts", "I got 100 points", "mixed_numbers", "very_hard", "Leet + large number"),
    TestCase("v3rs10n 2.0 15 0ut", "Version 2.0 is out", "mixed_numbers", "very_hard", "Leet + version"),
    TestCase("p4g3 42 0f th3 b00k", "Page 42 of the book", "mixed_numbers", "very_hard", "Leet + page number"),
    TestCase("sh3 1s 25 y34rs 0ld", "She is 25 years old", "mixed_numbers", "very_hard", "Leet + age"),
    TestCase("th3 3v3nt 1s 4t 9 4m", "The event is at 9 AM", "mixed_numbers", "very_hard", "Leet + time AM"),

    # =========================================================================
    # INTERNET SLANG / ABBREVIATIONS
    # =========================================================================
    TestCase("idk what to do", "I don't know what to do", "slang", "medium", "IDK expansion"),
    TestCase("tbh it was boring", "To be honest it was boring", "slang", "medium", "TBH expansion"),
    TestCase("ngl thats fire", "Not gonna lie that's fire", "slang", "medium", "NGL expansion"),
    TestCase("imo ur wrong", "In my opinion you're wrong", "slang", "medium", "IMO expansion"),
    TestCase("brb need food", "Be right back need food", "slang", "medium", "BRB expansion"),
    TestCase("g2g cya", "Got to go see you", "slang", "medium", "G2G CYA expansion"),
    TestCase("omg thats crazy", "Oh my god that's crazy", "slang", "medium", "OMG expansion"),
    TestCase("btw i forgot", "By the way I forgot", "slang", "medium", "BTW expansion"),
    TestCase("smh at this", "Shaking my head at this", "slang", "medium", "SMH expansion"),
    TestCase("fr fr no cap", "For real for real no cap", "slang", "hard", "Gen Z emphasis"),

    # =========================================================================
    # GAMING CULTURE
    # =========================================================================
    TestCase("ez game gg", "Easy game good game", "gaming", "medium", "Basic gaming"),
    TestCase("skill diff", "Skill difference", "gaming", "hard", "Gaming slang"),
    TestCase("get rekt noob", "Get wrecked noob", "gaming", "hard", "Trash talk"),
    TestCase("ur trash ngl", "You're trash not gonna lie", "gaming", "hard", "Toxic gaming"),
    TestCase("skill issue fr", "Skill issue for real", "gaming", "hard", "Meme phrase"),
    TestCase("gg ez no re", "Good game easy no rematch", "gaming", "hard", "Post-game"),
    TestCase("1v1 me bro", "One versus one me bro", "gaming", "medium", "Challenge"),
    TestCase("he got diffed", "He got outperformed", "gaming", "hard", "Outplay slang"),
    TestCase("clutch or kick", "Clutch or kick", "gaming", "medium", "Gaming phrase"),
    TestCase("nice aim hacks", "Nice aim hacks", "gaming", "medium", "Accusation"),

    # =========================================================================
    # GEN Z SLANG
    # =========================================================================
    TestCase("thats bussin", "That's really good", "gen_z", "hard", "Bussin = delicious/good"),
    TestCase("no cap this slaps", "No lie this is great", "gen_z", "hard", "Slaps = great"),
    TestCase("lowkey fire", "Kind of great", "gen_z", "hard", "Lowkey fire"),
    TestCase("its giving main character", "It's giving main character energy", "gen_z", "very_hard", "Gen Z phrase"),
    TestCase("periodt", "Period (emphatic)", "gen_z", "hard", "Emphasis"),
    TestCase("slay queen", "Slay queen", "gen_z", "medium", "Compliment"),
    TestCase("based take", "Based take", "gen_z", "hard", "Based = good opinion"),
    TestCase("thats mid ngl", "That's mediocre not gonna lie", "gen_z", "hard", "Mid = average"),
    TestCase("rent free in my head", "Rent free in my head", "gen_z", "medium", "Meme phrase"),
    TestCase("caught in 4k", "Caught in 4k", "gen_z", "medium", "Caught red-handed"),

    # =========================================================================
    # HEAVY LEETSPEAK (Complex Patterns)
    # =========================================================================
    TestCase("|337 5p34k", "Leet speak", "heavy_leet", "very_hard", "Classic 1337"),
    TestCase("|-|4ck3r", "Hacker", "heavy_leet", "very_hard", "H with pipe pattern"),
    TestCase("n00b", "Noob", "heavy_leet", "medium", "Classic noob"),
    TestCase("pwn3d", "Pwned", "heavy_leet", "medium", "Owned variant"),
    TestCase("|<1||", "Kill", "heavy_leet", "very_hard", "K with pipe"),
    TestCase("ph34r m3", "Fear me", "heavy_leet", "hard", "Ph for F"),
    TestCase("3p1c w1n", "Epic win", "heavy_leet", "hard", "Gaming phrase"),
    TestCase("1337 h4x0r", "Leet hacker", "heavy_leet", "very_hard", "Elite hacker"),
    TestCase("r0x0rz", "Rocks", "heavy_leet", "hard", "Xor suffix"),
    TestCase("k3wb01", "Coolboy", "heavy_leet", "very_hard", "Creative spelling"),

    # =========================================================================
    # CASE VARIATIONS
    # =========================================================================
    TestCase("HELLO WORLD", "Hello world", "case", "easy", "All caps"),
    TestCase("hElLo WoRlD", "Hello world", "case", "medium", "Alternating case"),
    TestCase("hello world", "Hello world", "case", "easy", "All lowercase"),
    TestCase("H3LL0 W0RLD", "Hello world", "case", "medium", "Caps + leet"),
    TestCase("ThAnKs FoR tHe HeLp", "Thanks for the help", "case", "medium", "Spongebob case"),

    # =========================================================================
    # PHONETIC SPELLINGS
    # =========================================================================
    TestCase("kewl", "Cool", "phonetic", "medium", "Kewl spelling"),
    TestCase("wut", "What", "phonetic", "easy", "Wut spelling"),
    TestCase("gud", "Good", "phonetic", "easy", "Gud spelling"),
    TestCase("prolly", "Probably", "phonetic", "medium", "Prolly spelling"),
    TestCase("gonna", "Going to", "phonetic", "easy", "Gonna contraction"),
    TestCase("wanna", "Want to", "phonetic", "easy", "Wanna contraction"),
    TestCase("gotta", "Got to", "phonetic", "easy", "Gotta contraction"),
    TestCase("dunno", "Don't know", "phonetic", "medium", "Dunno contraction"),
    TestCase("lemme", "Let me", "phonetic", "medium", "Lemme contraction"),
    TestCase("gimme", "Give me", "phonetic", "medium", "Gimme contraction"),

    # =========================================================================
    # COMPLEX SENTENCES (Multi-pattern)
    # =========================================================================
    TestCase("idk wh4t 2 d0 tbh ngl", "I don't know what to do to be honest not gonna lie",
             "complex", "very_hard", "Multiple patterns"),
    TestCase("c u l8r m8 gonna b gr8", "See you later mate going to be great",
             "complex", "very_hard", "Multiple word subs"),
    TestCase("thx 4 th3 h31p ur 4w3s0m3", "Thanks for the help you're awesome",
             "complex", "very_hard", "Mixed patterns"),
    TestCase("1 d0nt kn0w wut 2 d0 4 r34l", "I don't know what to do for real",
             "complex", "very_hard", "Heavy mixed"),
    TestCase("ur prolly r1ght tbh n0 c4p", "You're probably right to be honest no cap",
             "complex", "very_hard", "Slang + leet"),

    # =========================================================================
    # PRESERVATION (Should NOT change)
    # =========================================================================
    TestCase("The quick brown fox jumps over the lazy dog.",
             "The quick brown fox jumps over the lazy dog.",
             "preservation", "easy", "Clean sentence"),
    TestCase("Hello, how are you doing today?",
             "Hello, how are you doing today?",
             "preservation", "easy", "Clean question"),
    TestCase("I appreciate your help.",
             "I appreciate your help.",
             "preservation", "easy", "Polite sentence"),
    TestCase("Have a great day!",
             "Have a great day!",
             "preservation", "easy", "Greeting"),
    TestCase("The meeting is scheduled for 3 PM.",
             "The meeting is scheduled for 3 PM.",
             "preservation", "easy", "Formal sentence"),

    # =========================================================================
    # EDGE CASES / ADVERSARIAL
    # =========================================================================
    TestCase("", "", "edge", "easy", "Empty string"),
    TestCase("a", "A", "edge", "easy", "Single char"),
    TestCase("42", "42", "edge", "easy", "Just numbers"),
    TestCase("???", "???", "edge", "easy", "Just punctuation"),
    TestCase("1111", "1111", "edge", "hard", "Ambiguous 1s - could be IIII or 1111"),
    TestCase("2222", "2222", "edge", "hard", "Ambiguous 2s"),
    TestCase("@@@", "@@@", "edge", "hard", "Just @ symbols"),
    TestCase("i <3 u", "I love you", "edge", "hard", "Heart symbol"),
    TestCase(":-)", ":-)", "edge", "easy", "Emoticon"),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

@dataclass
class TestResult:
    """Result of a single test case."""
    test_case: TestCase
    prediction: str
    passed: bool
    similarity: float
    inference_time_ms: float


class TestSuite:
    """Test suite for evaluating leetspeak models."""

    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[TEST] Loading model from: {model_path}")
        print(f"[TEST] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print("[TEST] Model loaded successfully")

    def _compute_similarity(self, pred: str, expected: str) -> float:
        """Compute normalized similarity between prediction and expected."""
        pred_lower = pred.strip().lower()
        exp_lower = expected.strip().lower()

        if pred_lower == exp_lower:
            return 1.0

        # Character-level similarity
        if not pred_lower or not exp_lower:
            return 0.0

        # Simple Levenshtein-ish ratio
        common = sum(1 for a, b in zip(pred_lower, exp_lower) if a == b)
        max_len = max(len(pred_lower), len(exp_lower))
        return common / max_len

    def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case."""
        start_time = time.time()

        # Handle empty input
        if not test_case.input.strip():
            return TestResult(
                test_case=test_case,
                prediction="",
                passed=test_case.expected == "",
                similarity=1.0 if test_case.expected == "" else 0.0,
                inference_time_ms=0.0
            )

        # Tokenize and generate
        inputs = self.tokenizer(
            test_case.input,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )

        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time_ms = (time.time() - start_time) * 1000

        similarity = self._compute_similarity(prediction, test_case.expected)
        passed = similarity >= 0.9  # Allow small differences

        return TestResult(
            test_case=test_case,
            prediction=prediction,
            passed=passed,
            similarity=similarity,
            inference_time_ms=inference_time_ms
        )

    def run_all_tests(self, verbose: bool = False) -> Dict:
        """Run all test cases and return results."""
        results = []
        category_stats = defaultdict(lambda: {"passed": 0, "total": 0})
        difficulty_stats = defaultdict(lambda: {"passed": 0, "total": 0})

        print(f"\n[TEST] Running {len(TEST_CASES)} test cases...")

        for test_case in tqdm(TEST_CASES, desc="Testing"):
            result = self.run_single_test(test_case)
            results.append(result)

            # Update stats
            category_stats[test_case.category]["total"] += 1
            difficulty_stats[test_case.difficulty]["total"] += 1

            if result.passed:
                category_stats[test_case.category]["passed"] += 1
                difficulty_stats[test_case.difficulty]["passed"] += 1

            # Verbose output
            if verbose:
                status = "✓" if result.passed else "✗"
                print(f"\n  {status} [{test_case.category}/{test_case.difficulty}]")
                print(f"    IN:  {test_case.input}")
                print(f"    EXP: {test_case.expected}")
                print(f"    OUT: {result.prediction}")
                print(f"    SIM: {result.similarity:.2%}")

        # Compute summary
        total_passed = sum(1 for r in results if r.passed)
        total_tests = len(results)
        avg_inference_time = sum(r.inference_time_ms for r in results) / total_tests

        # Category breakdown
        category_breakdown = {}
        for cat, stats in sorted(category_stats.items()):
            pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            category_breakdown[cat] = {
                "passed": stats["passed"],
                "total": stats["total"],
                "percentage": pct
            }

        # Difficulty breakdown
        difficulty_breakdown = {}
        for diff, stats in sorted(difficulty_stats.items()):
            pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            difficulty_breakdown[diff] = {
                "passed": stats["passed"],
                "total": stats["total"],
                "percentage": pct
            }

        # Failed tests
        failed_tests = [r for r in results if not r.passed]

        summary = {
            "model_path": self.model_path,
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_tests - total_passed,
            "accuracy": total_passed / total_tests * 100,
            "avg_inference_time_ms": avg_inference_time,
            "category_breakdown": category_breakdown,
            "difficulty_breakdown": difficulty_breakdown,
            "failed_tests": [
                {
                    "input": r.test_case.input,
                    "expected": r.test_case.expected,
                    "prediction": r.prediction,
                    "category": r.test_case.category,
                    "difficulty": r.test_case.difficulty,
                    "similarity": r.similarity,
                }
                for r in failed_tests
            ],
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print a formatted summary of test results."""
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)

        print(f"\nModel: {summary['model_path']}")
        print(f"\nOverall: {summary['passed']}/{summary['total_tests']} passed ({summary['accuracy']:.1f}%)")
        print(f"Average inference time: {summary['avg_inference_time_ms']:.1f}ms")

        # Category breakdown
        print("\n" + "-" * 70)
        print("RESULTS BY CATEGORY")
        print("-" * 70)
        for cat, stats in sorted(summary['category_breakdown'].items()):
            bar = "█" * int(stats['percentage'] / 5) + "░" * (20 - int(stats['percentage'] / 5))
            print(f"  {cat:20s} {bar} {stats['passed']:3d}/{stats['total']:3d} ({stats['percentage']:5.1f}%)")

        # Difficulty breakdown
        print("\n" + "-" * 70)
        print("RESULTS BY DIFFICULTY")
        print("-" * 70)
        for diff in ["easy", "medium", "hard", "very_hard"]:
            if diff in summary['difficulty_breakdown']:
                stats = summary['difficulty_breakdown'][diff]
                bar = "█" * int(stats['percentage'] / 5) + "░" * (20 - int(stats['percentage'] / 5))
                print(f"  {diff:12s} {bar} {stats['passed']:3d}/{stats['total']:3d} ({stats['percentage']:5.1f}%)")

        # Failed tests
        if summary['failed_tests']:
            print("\n" + "-" * 70)
            print(f"FAILED TESTS ({len(summary['failed_tests'])})")
            print("-" * 70)
            for fail in summary['failed_tests'][:20]:  # Show first 20
                print(f"\n  [{fail['category']}/{fail['difficulty']}]")
                print(f"    IN:  {fail['input']}")
                print(f"    EXP: {fail['expected']}")
                print(f"    OUT: {fail['prediction']}")
                print(f"    SIM: {fail['similarity']:.2%}")

            if len(summary['failed_tests']) > 20:
                print(f"\n  ... and {len(summary['failed_tests']) - 20} more failures")

        # Final grade
        print("\n" + "=" * 70)
        accuracy = summary['accuracy']
        if accuracy >= 95:
            grade = "A+ (Excellent!)"
        elif accuracy >= 90:
            grade = "A (Great)"
        elif accuracy >= 85:
            grade = "B (Good)"
        elif accuracy >= 80:
            grade = "C (Acceptable)"
        elif accuracy >= 70:
            grade = "D (Needs work)"
        else:
            grade = "F (Major issues)"

        print(f"FINAL GRADE: {grade}")
        print("=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="V3 Test Suite")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to model directory or HuggingFace model ID")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for each test")
    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Run tests
    suite = TestSuite(args.model, args.device)
    summary = suite.run_all_tests(verbose=args.verbose)

    # Print summary
    suite.print_summary(summary)

    # Export if requested
    if args.export:
        with open(args.export, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[TEST] Results exported to: {args.export}")

    # Exit with code based on pass rate
    if summary['accuracy'] >= 80:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
