#!/usr/bin/env python3
"""
Smart Test Suite for ByT5 Leetspeak Decoder
============================================
Ignores capitalization, punctuation, and minor formatting differences.
Focuses on semantic correctness.
"""

import re

# ============================================================================
# TEST CASES (150 examples)
# ============================================================================

test_cases = [
    # === BASIC LEETSPEAK ===
    ("1 c4n s33 y0u", "i can see you"),
    ("th1s 1s 4 t3st", "this is a test"),
    ("h3ll0 w0rld", "hello world"),
    ("g00d m0rn1ng 3v3ry0n3", "good morning everyone"),

    # === TIME/WAITING ===
    ("c u l8r t0n1ght", "see you later tonight"),
    ("w8 4 m3", "wait for me"),
    ("1ll b th3r3 1n 5 m1ns", "ill be there in 5 mins"),
    ("s0rry 1m runn1ng l8", "sorry im running late"),
    ("brb 1n 10", "be right back in 10"),
    ("brb", "be right back"),
    ("w8 up 4 m3", "wait up for me"),
    ("g2g l8r m8", "got to go later mate"),
    ("g2g", "got to go"),
    ("c u @ 8pm", "see you at 8pm"),

    # === ABBREVIATIONS ===
    ("thx 4 th3 h3lp br0", "thanks for the help bro"),
    ("np m8", "no problem mate"),
    ("idk wh4t 2 d0 tbh", "i dont know what to do to be honest"),
    ("omg r u s3r10us rn", "oh my god are you serious right now"),
    ("lol th4ts s0 funny", "lol thats so funny"),
    ("btw d1d u s33 th4t", "by the way did you see that"),
    ("nvm 1 f0und 1t", "nevermind i found it"),
    ("imo th1s 1s b3tt3r", "in my opinion this is better"),
    ("tbh 1 d0nt c4r3", "to be honest i dont care"),
    ("afaik th4ts c0rr3ct", "as far as i know thats correct"),

    # === QUESTIONS ===
    ("y d1d u d0 th4t", "why did you do that"),
    ("wh3r3 r u g01ng", "where are you going"),
    ("h0w l0ng w1ll 1t t4k3", "how long will it take"),
    ("wh0 1s c0m1ng 2 th3 p4rty", "who is coming to the party"),
    ("c4n u s3nd m3 th3 l1nk", "can you send me the link"),
    ("wh4t r u d01ng 2n1ght", "what are you doing tonight"),

    # === TECH/WORK ===
    ("th3 c0d3 1s br0k3n 4g41n", "the code is broken again"),
    ("n33d 2 d3pl0y b4 3pm", "need to deploy before 3pm"),
    ("th3 s3rv3r 1s d0wn rn", "the server is down right now"),
    ("ch3ck th3 l0gs 4 3rr0rs", "check the logs for errors"),
    ("push1ng 2 pr0d 2m0rr0w", "pushing to prod tomorrow"),
    ("g1t pull b4 u st4rt", "git pull before you start"),
    ("run th3 t3sts b4 c0mm1t", "run the tests before commit"),
    ("d3bug th1s 4s4p", "debug this asap"),

    # === INTENSE LEETSPEAK ===
    ("7h15 15 cr4zy h4rd 2 r34d", "this is crazy hard to read"),
    ("1 c4n7 b3l13v3 7h15 w0rk5", "i cant believe this works"),
    ("y0u 4r3 7h3 b357", "you are the best"),
    ("w3 g0774 f1x 7h15", "we gotta fix this"),
    ("1337 h4x0r sk1llz", "leet hacker skills"),
    ("pwn3d th4t n00b", "pwned that noob"),
    ("7h3 c4k3 1s 4 l13", "the cake is a lie"),
    ("3p1c w1n br0", "epic win bro"),

    # === GAMING ===
    ("gg wp", "good game well played"),
    ("gg wp 3z g4m3", "good game well played easy game"),
    ("n1c3 clutch br0", "nice clutch bro"),
    ("r3kt th4t t34m", "rekt that team"),
    ("c4rry1ng th3 t34m rn", "carrying the team right now"),
    ("s0 m4ny n00bs 1n th1s l0bby", "so many noobs in this lobby"),
    ("1v1 m3 br0", "1v1 me bro"),
    ("3z g4m3", "easy game"),

    # === MIXED ===
    ("s3nd m3 th3 d3t41ls l8r", "send me the details later"),
    ("g0t 2 g0 2 th3 st0r3 b4 1t cl0s3s", "got to go to the store before it closes"),
    ("d1dnt m34n 2 b0th3r u", "didnt mean to bother you"),
    ("th4nks 4 w41t1ng s0 l0ng", "thanks for waiting so long"),
    ("c4nt w8 2 s33 u", "cant wait to see you"),
    ("1m s0 t1r3d 4ft3r w0rk", "im so tired after work"),
    ("l3ts m33t @ 7", "lets meet at 7"),
    ("n0 w4y th4ts 1ns4n3", "no way thats insane"),
    ("th1s 1s b3tt3r th4n b4", "this is better than before"),

    # === SOCIAL ===
    ("wh4ts g00d br0", "whats good bro"),
    ("ch1ll1ng @ h0m3 rn", "chilling at home right now"),
    ("w4nn4 gr4b s0m3 f00d", "wanna grab some food"),
    ("y34h 1m d0wn 4 th4t", "yeah im down for that"),
    ("s0rry g2g my m0m 1s c4ll1ng", "sorry got to go my mom is calling"),
    ("txt m3 wh3n u g3t th3r3", "text me when you get there"),
    ("1ll h1t u up l8r", "ill hit you up later"),
    ("n1c3 2 m33t u", "nice to meet you"),
    ("l3ts c4tch up s00n", "lets catch up soon"),

    # === SLANG ===
    ("th4ts l1t af", "thats lit as fuck"),
    ("n0 c4p", "no cap"),
    ("str41ght f1r3", "straight fire"),
    ("l0w k3y g00d", "low key good"),
    ("h1gh k3y 4m4z1ng", "high key amazing"),

    # === FEELINGS ===
    ("1m s0 3xc1t3d", "im so excited"),
    ("f33l1ng s4d 2d4y", "feeling sad today"),
    ("th1s m4k3s m3 s0 h4ppy", "this makes me so happy"),
    ("d0nt b m4d @ m3", "dont be mad at me"),
    ("pr0ud 0f wh4t w3 d1d", "proud of what we did"),

    # === TRAVEL ===
    ("m33t @ th3 tr41n st4t10n", "meet at the train station"),
    ("g01ng 2 NYC n3xt w33k", "going to nyc next week"),
    ("b00k3d fl1ght 2 LA", "booked flight to la"),
    ("@ th3 4irp0rt n0w", "at the airport now"),

    # === NUMBERS (should preserve) ===
    ("1 h4v3 2 c4ts", "i have 2 cats"),
    ("m33t m3 @ 3 PM", "meet me at 3 pm"),
    ("v3rs10n 2.0 1s 0ut", "version 2.0 is out"),
    ("1 g0t 100 p01nts", "i got 100 points"),
    ("p4g3 42 0f th3 b00k", "page 42 of the book"),

    # === EDGE CASES ===
    ("1ts 2 l8", "its too late"),
    ("1t5 2 l8", "its too late"),
    ("2 b or n0t 2 b", "to be or not to be"),
    ("ur 2 sl0w 2 b g00d", "your too slow to be good"),
    ("1ts r41n1ng h4rd rn", "its raining hard right now"),
    ("c0ld w34th3r 2d4y", "cold weather today"),

    # === LONG SENTENCES ===
    ("1 w4s th1nk1ng w3 c0uld m33t up @ th3 c4f3", "i was thinking we could meet up at the cafe"),
    ("th3 pr0j3ct 1s 4lm0st d0n3 but w3 n33d 2 t3st 1t", "the project is almost done but we need to test it"),
    ("c4n u s3nd m3 th3 f1l3s y0u pr0m1s3d", "can you send me the files you promised"),
    ("1 d0nt kn0w 1f 1 c4n m4k3 1t 2n1ght", "i dont know if i can make it tonight"),
    ("th4nk y0u s0 much 4 4ll y0ur h3lp", "thank you so much for all your help"),

    # === COMMON WORDS ===
    ("h3ll0", "hello"),
    ("th4nks", "thanks"),
    ("pl34s3", "please"),
    ("s0rry", "sorry"),
    ("y34h", "yeah"),
    ("n1c3", "nice"),
    ("c00l", "cool"),
    ("4w3s0m3", "awesome"),
    ("4m4z1ng", "amazing"),
    ("g00d", "good"),

    # === PUNCTUATION HANDLING ===
    ("wh4t!!", "what"),
    ("n0 w4y!!!", "no way"),
    ("r34lly???", "really"),
    ("OMG!!!", "omg"),

    # === MORE PATTERNS ===
    ("sh0uld0f d0n3 1t", "should have done it"),
    ("c0uld0f b33n b3tt3r", "could have been better"),
    ("w0uld0f h3lp3d", "would have helped"),
    ("n33d 2 f1x th1s", "need to fix this"),
    ("w4nt 2 g0 h0m3", "want to go home"),
    ("g0nn4 b l8", "gonna be late"),
    ("w4nn4 h4ng 0ut", "wanna hang out"),
    ("g0tt4 g0 n0w", "gotta go now"),
    ("l3mm3 kn0w", "let me know"),
    ("g1mm3 4 s3c", "give me a sec"),
]

# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize(text: str) -> str:
    """Normalize text for comparison - ignore case, punctuation, extra spaces."""
    text = text.lower()
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", "", text)
    # Normalize contractions
    text = text.replace("'", "")
    text = text.replace("dont", "don't").replace("dont", "dont")
    text = text.replace("cant", "can't").replace("cant", "cant")
    text = text.replace("wont", "won't").replace("wont", "wont")
    text = text.replace("im ", "i'm ").replace("im ", "im ")
    text = text.replace("ill ", "i'll ").replace("ill ", "ill ")
    text = text.replace("youre", "you're").replace("youre", "youre")
    text = text.replace("thats", "that's").replace("thats", "thats")
    text = text.replace("its ", "it's ").replace("its ", "its ")
    text = text.replace("didnt", "didn't").replace("didnt", "didnt")
    # Normalize back (we don't care about apostrophes)
    text = text.replace("'", "")
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_smart_tests(translate_function):
    """Run tests with smart normalization."""
    passed = 0
    failed = 0
    failures = []

    print("=" * 70)
    print("SMART LEETSPEAK DECODER TEST SUITE")
    print("(ignores capitalization, punctuation, and minor formatting)")
    print("=" * 70)
    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = translate_function(input_text)

        result_norm = normalize(result)
        expected_norm = normalize(expected)

        if result_norm == expected_norm:
            passed += 1
            if i % 25 == 0:
                print(f"[{i:3d}/{len(test_cases)}] ✓ {passed} passed so far")
        else:
            failed += 1
            failures.append({
                'input': input_text,
                'expected': expected,
                'expected_norm': expected_norm,
                'got': result,
                'got_norm': result_norm
            })
            print(f"\n[{i:3d}/{len(test_cases)}] ✗ FAIL")
            print(f"  Input:    {input_text}")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    # Summary
    accuracy = (passed / len(test_cases)) * 100
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total:    {len(test_cases)}")
    print(f"Passed:   {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed:   {failed} ({failed/len(test_cases)*100:.1f}%)")
    print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 70)

    if accuracy >= 90:
        print("🎉 EXCELLENT! Model is production-ready!")
    elif accuracy >= 80:
        print("✅ Good! Minor improvements possible.")
    elif accuracy >= 70:
        print("⚠️ Acceptable, but needs more training.")
    else:
        print("❌ Needs significant improvement.")

    return {
        'total': len(test_cases),
        'passed': passed,
        'failed': failed,
        'accuracy': accuracy,
        'failures': failures
    }

# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
To use this test suite:

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('./byt5_leetspeak_model_v2')
tokenizer = AutoTokenizer.from_pretrained('./byt5_leetspeak_model_v2')
model = model.cuda()

def translate(text):
    inputs = tokenizer(text, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

from smart_test_suite import run_smart_tests
results = run_smart_tests(translate)
""")
