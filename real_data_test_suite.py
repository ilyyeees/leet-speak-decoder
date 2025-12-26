"""
Comprehensive Test Suite for ByT5 Leetspeak Decoder
====================================================
Run this in Google Colab to evaluate model performance across diverse leetspeak patterns.

Usage:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder")
    tokenizer = AutoTokenizer.from_pretrained("ilyyeees/byt5-leetspeak-decoder")

    def translate(text):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Then run the tests below
"""

# ============================================================================
# COMPREHENSIVE TEST SUITE
# ============================================================================

test_cases = [
    # ========== BASIC NUMBER SUBSTITUTIONS ==========
    ("1 c4n s33 y0u", "I can see you"),
    ("th1s 1s 4 t3st", "this is a test"),
    ("h3ll0 w0rld", "hello world"),
    ("1 4m s0 h4ppy 2d4y", "I am so happy today"),
    ("g00d m0rn1ng 3v3ry0n3", "good morning everyone"),

    # ========== TIME AND WAITING ==========
    ("c u l8r t0n1ght", "see you later tonight"),
    ("w8 4 m3 @ th3 c0rn3r", "wait for me at the corner"),
    ("1ll b th3r3 1n 5 m1ns", "Ill be there in 5 mins"),
    ("s0rry 1m runn1ng l8", "sorry Im running late"),
    ("brb 1n 10", "brb in 10"),
    ("w8 up 4 m3", "wait up for me"),
    ("g2g l8r m8", "got to go later mate"),
    ("c u @ 8pm", "see you at 8pm"),

    # ========== COMMON ABBREVIATIONS WITH LEETSPEAK ==========
    ("thx 4 th3 h3lp br0", "thanks for the help bro"),
    ("np m8 anyt1m3", "no problem mate anytime"),
    ("idk wh4t 2 d0 tbh", "I dont know what to do to be honest"),
    ("omg r u s3r10us rn??", "oh my god are you serious right now??"),
    ("lol th4ts s0 funny", "lol thats so funny"),
    ("btw d1d u s33 th4t?", "by the way did you see that?"),
    ("nvm 1 f0und 1t", "nevermind I found it"),
    ("imo th1s 1s b3tt3r", "in my opinion this is better"),
    ("tbh 1 d0nt c4r3", "to be honest I dont care"),
    ("afaik th4ts c0rr3ct", "as far as I know thats correct"),

    # ========== QUESTIONS ==========
    ("y d1d u d0 th4t?", "why did you do that?"),
    ("wh3r3 r u g01ng 2 b?", "where are you going to be?"),
    ("h0w l0ng w1ll 1t t4k3?", "how long will it take?"),
    ("wh0 1s c0m1ng 2 th3 p4rty?", "who is coming to the party?"),
    ("wh3n 4r3 w3 m33t1ng up?", "when are we meeting up?"),
    ("c4n u s3nd m3 th3 l1nk?", "can you send me the link?"),
    ("d0 u kn0w wh4t t1m3 1t 1s?", "do you know what time it is?"),
    ("wh4t r u d01ng 2n1ght?", "what are you doing tonight?"),
    ("wh3r3 d1d u put 1t?", "where did you put it?"),

    # ========== WORK/TECH RELATED ==========
    ("th3 c0d3 1s br0k3n 4g41n", "the code is broken again"),
    ("n33d 2 d3pl0y b4 3pm", "need to deploy before 3pm"),
    ("th3 s3rv3r 1s d0wn rn", "the server is down right now"),
    ("ch3ck th3 l0gs 4 3rr0rs", "check the logs for errors"),
    ("push1ng 2 pr0d 2m0rr0w", "pushing to prod tomorrow"),
    ("r3v13w my PR pl0x", "review my PR please"),
    ("g1t pull b4 u st4rt", "git pull before you start"),
    ("m3rg3 c0nfl1ct 1n m41n", "merge conflict in main"),
    ("run th3 t3sts b4 c0mm1t", "run the tests before commit"),
    ("d3bug th1s 4s4p", "debug this asap"),
    ("r3f4ct0r th4t c0d3", "refactor that code"),
    ("f1x th3 typ0s 1n th3 d0cs", "fix the typos in the docs"),

    # ========== INTENSE LEETSPEAK ==========
    ("7h15 15 cr4zy h4rd 2 r34d", "this is crazy hard to read"),
    ("1 c4n7 b3l13v3 7h15 w0rk5", "I cant believe this works"),
    ("y0u 4r3 7h3 b357", "you are the best"),
    ("w3 g0774 f1x 7h15 4$4p", "we gotta fix this asap"),
    ("1337 h4x0r sk1llz", "leet hacker skills"),
    ("pwn3d th4t n00b", "pwned that noob"),
    ("7h3 c4k3 15 4 l13", "the cake is a lie"),
    ("3p1c w1n br0", "epic win bro"),
    ("n1c3 7ry m8", "nice try mate"),

    # ========== MIXED INTENSITY ==========
    ("s3nd m3 th3 d3t41ls l8r", "send me the details later"),
    ("g0t 2 g0 2 th3 st0r3 b4 1t cl0s3s", "got to go to the store before it closes"),
    ("d1dnt m34n 2 b0th3r u", "didnt mean to bother you"),
    ("th4nks 4 w41t1ng s0 l0ng", "thanks for waiting so long"),
    ("c4nt w8 2 s33 u @ th3 c0nc3rt", "cant wait to see you at the concert"),
    ("1m s0 t1r3d 4ft3r w0rk", "Im so tired after work"),
    ("l3ts m33t @ 7 0cl0ck", "lets meet at 7 oclock"),
    ("n0 w4y th4ts 1ns4n3!!", "no way thats insane!!"),
    ("th1s 1s b3tt3r th4n b4", "this is better than before"),

    # ========== SOCIAL/CASUAL ==========
    ("wh4ts g00d br0?", "whats good bro?"),
    ("ch1ll1ng @ h0m3 rn", "chilling at home right now"),
    ("w4nn4 gr4b s0m3 f00d?", "wanna grab some food?"),
    ("y34h 1m d0wn 4 th4t", "yeah Im down for that"),
    ("s0rry g2g my m0m 1s c4ll1ng", "sorry got to go my mom is calling"),
    ("txt m3 wh3n u g3t th3r3", "text me when you get there"),
    ("1ll h1t u up l8r", "Ill hit you up later"),
    ("n1c3 2 m33t u 2d4y", "nice to meet you today"),
    ("h4ng0ut @ my pl4c3?", "hangout at my place?"),
    ("l3ts c4tch up s00n", "lets catch up soon"),

    # ========== URGENT/EMPHATIC ==========
    ("OMG!!! 1 f0rg0t!!!", "OMG!!! I forgot!!!"),
    ("HURRY!! w3 r l8 4 th3 tr41n", "HURRY!! we are late for the train"),
    ("PLS PLS PLS h3lp m3 n0w", "PLEASE PLEASE PLEASE help me now"),
    ("N0000 th1s c4nt b h4pp3n1ng", "NOOOO this cant be happening"),
    ("Y3SSS w3 d1d 1t!!!", "YESSS we did it!!!"),
    ("WTF 1s g01ng 0n h3r3??", "WTF is going on here??"),
    ("ST0P R1GHT N0W!!!", "STOP RIGHT NOW!!!"),

    # ========== LONGER SENTENCES ==========
    ("1 w4s th1nk1ng w3 c0uld m33t up @ th3 c4f3 4r0und 3pm 1f ur fr33",
     "I was thinking we could meet up at the cafe around 3pm if your free"),
    ("th3 pr0j3ct 1s 4lm0st d0n3 but w3 n33d 2 t3st 1t b4 th3 d34dl1n3",
     "the project is almost done but we need to test it before the deadline"),
    ("c4n u pl34s3 s3nd m3 th3 f1l3s y0u pr0m1s3d 2 s3nd y3st3rd4y?",
     "can you please send me the files you promised to send yesterday?"),
    ("1 d0nt kn0w 1f 1 c4n m4k3 1t 2n1ght c0z 1m st1ll @ w0rk",
     "I dont know if I can make it tonight cause Im still at work"),
    ("th4nk y0u s0 much 4 4ll y0ur h3lp w1th th3 pr3s3nt4t10n",
     "thank you so much for all your help with the presentation"),

    # ========== EDGE CASES AND VARIATIONS ==========
    ("1111 s33 u 4444 th3 g4m3", "Ill see you after the game"),
    ("b3333n w41t1ng 4 ag3333s", "been waiting for ages"),
    ("ur 2 sl0w 2 b 2 g00d", "your too slow to be too good"),
    ("444 sur3", "for sure"),
    ("3v3ry1 n33ds 2 kn0w th1s", "everyone needs to know this"),
    ("2 b or n0t 2 b", "to be or not to be"),
    ("th3r3s n0 pl4c3 l1k3 h0m3", "theres no place like home"),

    # ========== NUMBERS AND LETTERS MIXED HEAVILY ==========
    ("4dd m3 0n d1sc0rd @ us3r#1234", "add me on discord at user#1234"),
    ("my p4ssw0rd 1s n0t 12345", "my password is not 12345"),
    ("c4ll m3 @ 555-0123", "call me at 555-0123"),
    ("v3rs10n 2.0 1s b3tt3r th4n v1", "version 2.0 is better than v1"),
    ("ch4pt3r 7 p4g3 42", "chapter 7 page 42"),
    ("r00m numb3r 303", "room number 303"),

    # ========== COMMON TYPOS/INTENTIONAL MISSPELLINGS ==========
    ("1 h4v n0 1d34 wh4t ur s4y1ng", "I have no idea what your saying"),
    ("y0ur g0nn4 l0v3 th1s", "your gonna love this"),
    ("th3y d0nt kn0w wh4t th3yr d01ng", "they dont know what theyre doing"),
    ("1ts 2 l8 2 f1x 1t n0w", "its too late to fix it now"),
    ("w3r3 g01ng 2 b th3r3 s00n", "were going to be there soon"),
    ("sh0uld0f d0n3 1t b3tt3r", "shouldve done it better"),
    ("c0uld0f w41t3d l0ng3r", "couldve waited longer"),

    # ========== GAMING/INTERNET CULTURE ==========
    ("gg wp 3z g4m3", "good game well played easy game"),
    ("n1c3 clutch br0", "nice clutch bro"),
    ("r3kt th4t t34m", "wrecked that team"),
    ("c4rry1ng th3 t34m rn", "carrying the team right now"),
    ("s0 m4ny n00bs 1n th1s l0bby", "so many noobs in this lobby"),
    ("lag sp1k3 k1ll3d m3", "lag spike killed me"),
    ("1v1 m3 br0", "1v1 me bro"),

    # ========== FOOD/LIFESTYLE ==========
    ("p1zz4 4 d1nn3r?", "pizza for dinner?"),
    ("gr4bb1ng c0ff33 @ 9", "grabbing coffee at 9"),
    ("l0v3 th1s r3st4ur4nt", "love this restaurant"),
    ("ch34p3r 0pt10n 4v41l4bl3", "cheaper option available"),
    ("h34lthy ch01c3s 0nly", "healthy choices only"),

    # ========== FEELINGS/EMOTIONS ==========
    ("1m s0 3xc1t3d!!!", "Im so excited!!!"),
    ("f33l1ng s4d 2d4y", "feeling sad today"),
    ("th1s m4k3s m3 s0 h4ppy", "this makes me so happy"),
    ("d0nt b m4d @ m3", "dont be mad at me"),
    ("w0rr13d 4b0ut th3 r3sults", "worried about the results"),
    ("pr0ud 0f wh4t w3 4cc0mpl1sh3d", "proud of what we accomplished"),

    # ========== TRAVEL/LOCATION ==========
    ("m33t @ th3 tr41n st4t10n", "meet at the train station"),
    ("g01ng 2 NYC n3xt w33k", "going to NYC next week"),
    ("b00k3d fl1ght 2 LA", "booked flight to LA"),
    ("@ th3 4irp0rt n0w", "at the airport now"),
    ("l0st 1n th3 c1ty", "lost in the city"),

    # ========== SLANG VARIATIONS ==========
    ("th4ts l1t af", "thats lit as fuck"),
    ("n0 c4p", "no cap"),
    ("str41ght f1r3", "straight fire"),
    ("d34d d3adb3ats", "dead deadbeats"),
    ("l0w k3y g00d", "low key good"),
    ("h1gh k3y 4m4z1ng", "high key amazing"),
    ("f4x n0 pr1nt3r", "facts no printer"),

    # ========== WEATHER/TIME ==========
    ("1ts r41n1ng h4rd rn", "its raining hard right now"),
    ("c0ld w34th3r 2d4y", "cold weather today"),
    ("sunn7 4ft3rn00n", "sunny afternoon"),
    ("w1nt3r 1s c0m1ng", "winter is coming"),
    ("spr1ng br34k pl4ns?", "spring break plans?"),
]

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_tests(translate_function):
    """
    Run all test cases and report results.

    Args:
        translate_function: Your model's translate function

    Returns:
        dict with results summary
    """
    passed = 0
    failed = 0
    failures = []

    print("=" * 80)
    print("LEETSPEAK DECODER TEST SUITE")
    print("=" * 80)
    print(f"\\nRunning {len(test_cases)} test cases...\\n")

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = translate_function(input_text)

        # Normalize for comparison (lowercase, strip)
        result_normalized = result.lower().strip()
        expected_normalized = expected.lower().strip()

        if result_normalized == expected_normalized:
            passed += 1
            # Show progress every 20 passed tests
            if i % 20 == 0:
                print(f"[{i:3d}/{len(test_cases)}] ✓ PASS")
        else:
            failed += 1
            failures.append({
                'input': input_text,
                'expected': expected,
                'got': result
            })
            # ALWAYS show failures immediately with full comparison
            print(f"\\n[{i:3d}/{len(test_cases)}] ✗ FAIL")
            print(f"  Input:    {input_text}")
            print(f"  Expected: {expected}")
            print(f"  Got:      {result}")

    # Summary
    accuracy = (passed / len(test_cases)) * 100
    print("\\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tests:  {len(test_cases)}")
    print(f"Passed:       {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed:       {failed} ({failed/len(test_cases)*100:.1f}%)")
    print(f"Accuracy:     {accuracy:.2f}%")

    # Note about failures
    if failures:
        print("\\n" + "=" * 80)
        print(f"FAILED TEST COUNT: {len(failures)}")
        print("=" * 80)
        print("(All failures shown inline above with Expected vs Got)")

    return {
        'total': len(test_cases),
        'passed': passed,
        'failed': failed,
        'accuracy': accuracy,
        'failures': failures
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
    To use this test suite in Google Colab:

    1. Load your model:
       from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

       model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder")
       tokenizer = AutoTokenizer.from_pretrained("ilyyeees/byt5-leetspeak-decoder")

       def translate(text):
           inputs = tokenizer(text, return_tensors="pt")
           outputs = model.generate(**inputs, max_length=256)
           return tokenizer.decode(outputs[0], skip_special_tokens=True)

    2. Run the tests:
       results = run_comprehensive_tests(translate)

    3. Analyze failures to identify improvement areas!
    """)
