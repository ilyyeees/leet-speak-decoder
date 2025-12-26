# byt5 leetspeak decoder

[![hugging face v1](https://img.shields.io/badge/%F0%9F%A4%97%20v1-71%25-blue)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)
[![hugging face v2](https://img.shields.io/badge/%F0%9F%A4%97%20v2-85%25-green)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder-v2)

a context-aware ai model that translates leetspeak back into clean english.
built on google's `byt5-base` architecture to handle character-level noise without vocabulary limitations.

**try it online**: [v2 (recommended)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder-v2) | [v1](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

---

## versions

| version | accuracy | training data | status |
|---------|----------|---------------|--------|
| **v1** | 71% | wikitext + synthetic | released |
| **v2** | **85%** | real reddit + qwen translations | ✅ **released** |

---

## quick start

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "ilyyeees/byt5-leetspeak-decoder-v2"  # or v1: byt5-leetspeak-decoder
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate("h3110 w0r1d"))  # hello world
print(translate("idk wh4t 2 d0 tbh"))  # i don't know what to do to be honest
```

---

## project structure

```
leetspeak/
├── v1_training/          # original training (wikitext + synthetic)
│   ├── byt5_leetspeak_decoder.py
│   ├── byt5_leetspeak_decoder.ipynb
│   └── finetune_*.py
│
├── v2_training/          # improved training (real reddit data)
│   ├── train_model_v2.py
│   ├── finetune_edge_cases.py
│   ├── smart_test_suite.py
│   └── data_pipeline/
│       ├── scraping/     # reddit comment scraper
│       └── processing/   # translation + corruption
│
├── comprehensive_test_suite.py
└── requirements.txt
```

---

## v1: synthetic training

trained on ~40k examples from wikitext-2 + samsum conversations, corrupted with synthetic leetspeak.

**strengths:**
- handles basic leetspeak (`h3ll0` → `hello`)
- context-aware numbers (`2 cats` preserved vs `2 late` → `too late`)

**weaknesses:**
- struggles with real-world reddit slang (`tbh`, `rn`, `ngl`)
- 71% accuracy on real comments

see [v1_training/README.md](v1_training/README.md) for details.

---

## v2: real-world training

uses real reddit comments translated by qwen 2.5 32b (via tensordock cloud gpu).

**pipeline:**
1. scrape 5k real reddit comments
2. translate to formal english using qwen 32b on cloud
3. corrupt originals further (3x multiplexing)
4. continue training v1 model on new data
5. fine-tune edge cases (brb, g2g, etc.)

**result:** **85% accuracy** on real-world slang (up from 71%).

see [v2_training/README.md](v2_training/README.md) for details.

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
