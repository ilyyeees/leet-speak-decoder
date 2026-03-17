# byt5 leetspeak decoder v3

[![hugging face](https://img.shields.io/badge/%F0%9F%A4%97%20model-100%25%20accuracy-brightgreen)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)
[![eval loss](https://img.shields.io/badge/eval%20loss-0.3812-blue)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

the definitive byte-level translator for leetspeak, internet slang, and visual character obfuscation.
built on `google/byt5-base`, v3 uses **curriculum learning** and **adversarial filtering** to finally solve the number ambiguity problem.

> **v3 is currently broken — use v2 for now.**
> the v3 model on `main` has known issues and is being reworked. for a working model, load from the `v2-legacy` branch:
> ```python
> model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder", revision="v2-legacy")
> tokenizer = AutoTokenizer.from_pretrained("ilyyeees/byt5-leetspeak-decoder", revision="v2-legacy")
> ```

**try it online**: [huggingface.co/ilyyeees/byt5-leetspeak-decoder](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

---

## the number problem: solved

v3 is the first model in this series to perfectly distinguish between numbers used as letters and numbers used as quantities within the same sentence.

```
input:  1t5 2 l8 4 2 people
v2 out: It's to late for to people.  (fail)
v3 out: It is too late for 2 people. (pass)
```

---

## current version: v3 (100% accuracy)

| metric               | v2 (legacy)  | v3 (current)   |
| -------------------- | ------------ | -------------- |
| mixed-number context | ~74%         | **100%**       |
| basic leet decoding  | 85%          | **100%**       |
| visual obfuscation   | moderate     | **high**       |
| eval loss            | 0.84         | **0.3812**     |
| output style         | casual/slang | formal english |

| version | accuracy | training data                   | status      |
| ------- | -------- | ------------------------------- | ----------- |
| **v3**  | **100%** | 740k curriculum + adversarial   | `main`      |
| v2      | 85%      | real reddit + qwen translations | `v2-legacy` |
| v1      | 71%      | wikitext + synthetic            | `v1-legacy` |

---

## quick start

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder")
tokenizer = AutoTokenizer.from_pretrained("ilyyeees/byt5-leetspeak-decoder")

def decode(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# test cases
print(decode("1t5 2 l8 4 th4t"))         # It is too late for that.
print(decode("1 g0t 100 p01nt5 0n 1t"))  # I got 100 points on it.
print(decode("idk wh4t 2 d0 tbh"))       # I don't know what to do to be honest.
```

### using older versions

```python
# v2 (85% accuracy)
model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder", revision="v2-legacy")

# v1 (71% accuracy)
model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder", revision="v1-legacy")
```

---

## training methodology

v3 was trained on **2x nvidia rtx 5090s** using a custom reverse-corruption pipeline:

### 1. clean base

high-quality english from wikitext and eli5 to ground the model in correct grammar.

### 2. llm adversarial corruption

qwen 2.5 72b generated "hard negatives"—specific leetspeak patterns that v2 failed to decode.

### 3. curriculum learning

trained in phases of increasing difficulty:

- phase 1: simple character swaps (`h3ll0` → `hello`)
- phase 2: internet slang expansion (`tbh` → `to be honest`)
- phase 3: mixed-number ambiguity (`1 h4v3 2 c4t5` → `i have 2 cats`)

### training stats

```
dataset:        740,712 training samples
epochs:         3
batch size:     384 effective (96 × 2 gpus × 2 accumulation)
training time:  5.4 hours
final loss:     2.5367 (train) / 0.3812 (eval)
```

---

## project structure

```
leetspeak/
├── v1_training/          # original (wikitext + synthetic)
├── v2_training/          # reddit + qwen translations
├── v3_training/          # curriculum + adversarial (current)
│   ├── train_model_v3.py # production training script
│   └── data_pipeline/    # llm corruption + script corruption
└── requirements.txt
```

---

## limitations

### formalization bias

v3 was trained on high-quality datasets (wiki/eli5), so it has a bias toward formal english:

- `ngl` → `not going to lie` (not "not gonna lie")
- `idk` → `i don't know` (not "i dunno")

### short inputs

extremely short inputs (1-2 chars) may be interpreted as standard english due to conservative decoding threshold.

---

## model architecture

- **base**: `google/byt5-base` (580m params)
- **tokenizer**: byte-level (handles any unicode/leetspeak chars)
- **precision**: bf16 mixed precision
- **inference**: ~100ms per sentence on gpu

---

## installation

```bash
pip install transformers torch sentencepiece
```

---

## links

- **model**: [ilyyeees/byt5-leetspeak-decoder](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)
- **dataset**: [ilyyeees/leetspeak-to-english](https://huggingface.co/datasets/ilyyeees/leetspeak-to-english)
- **github**: [ilyyeees/leet-speak-decoder](https://github.com/ilyyeees/leet-speak-decoder)

---

## license

mit
