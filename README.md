# byt5 leetspeak decoder

[![hugging face](https://img.shields.io/badge/%F0%9F%A4%97%20model-85%25%20accuracy-green)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

a context-aware ai model that translates leetspeak back into clean english.
built on google's `byt5-base` architecture to handle character-level noise without vocabulary limitations.

**try it online**: [huggingface.co/ilyyeees/byt5-leetspeak-decoder](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

---

## current version: v2 (85% accuracy)

| version | accuracy | training data | branch |
|---------|----------|---------------|--------|
| **v2** | **85%** | real reddit + qwen translations | `main` |
| v1 | 71% | wikitext + synthetic | `v1-legacy` |

---

## quick start

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder")
tokenizer = AutoTokenizer.from_pretrained("ilyyeees/byt5-leetspeak-decoder")

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate("h3110 w0r1d"))  # hello world
print(translate("idk wh4t 2 d0 tbh"))  # i don't know what to do to be honest
```

### using older versions

```python
# v1 (71% accuracy)
model = AutoModelForSeq2SeqLM.from_pretrained("ilyyeees/byt5-leetspeak-decoder", revision="v1-legacy")
```

---

## project structure

```
leetspeak/
├── v1_training/          # original training (wikitext + synthetic)
├── v2_training/          # improved training (real reddit data)
├── v3_training/          # upcoming: hybrid llm + script corruption
└── requirements.txt
```

---

## training approaches

### v1: synthetic training
trained on ~40k examples from wikitext-2 + samsum conversations, corrupted with synthetic leetspeak.
- handles basic leetspeak (`h3ll0` → `hello`)
- struggles with real-world slang (`tbh`, `rn`, `ngl`)

### v2: real-world training
uses real reddit comments translated by qwen 2.5 32b.
- scrape 5k real reddit comments
- translate to formal english using qwen 32b
- corrupt originals further (3x multiplexing)
- continue training v1 model on new data

**result:** 85% accuracy on real-world slang (up from 71%).

---

## model architecture

- **base**: `google/byt5-base` (580m params)
- **tokenizer**: byte-level (handles any unicode/leetspeak chars)
- **inference**: ~100ms per sentence on gpu

---

## installation

```bash
pip install transformers torch sentencepiece
```

---

## license

mit
