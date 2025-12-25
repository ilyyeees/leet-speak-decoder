# byt5 leetspeak decoder

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

a context-aware ai model that translates leetspeak back into clean english.
built on google's `byt5-base` architecture to handle character-level noise without vocabulary limitations.

**try it online**: [ilyyeees/byt5-leetspeak-decoder](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)

## features

- **context aware**: distinguishes between "2" as a number ("i have 2 cats") and "2" as a word ("i need 2 go").
- **robustness**: handles 1337 substitutions, slang (`thx`, `gr8`, `l8r`), and varying levels of corruption.
- **accuracy**: achieves ~98.2% accuracy on the test suite.

### performance

- **bleu**: 94.8
- **cer**: 0.7%

## files

- `byt5_leetspeak_decoder.ipynb`: the main notebook for training, usage, and testing.
- `byt5_leetspeak_decoder.py`: pure python export of the training logic.
- `finetune_safe.py`: (failed) attempted safe micro-fine-tuning for edge cases (caused regressions).
- `finetune_weak_patterns.py`: script for fixing specific weak patterns like `8->ate` or `thx->thanks`.

## usage

### using the pre-trained model (easiest)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "ilyyeees/byt5-leetspeak-decoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(translate("h3110 w0r1d"))
```

### training from scratch

1. install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. run the notebook `byt5_leetspeak_decoder.ipynb` in google colab or locally.

3. or use the python script:
   ```bash
   python byt5_leetspeak_decoder.py
   ```

## model architecture

- **base**: `google/byt5-base`
- **tokenizer**: byte-level (no oov issues with weird leetspeak chars)
- **training data**: 40k+ examples from wikitext-2 (clean) + samsum (conversational) + synthetic corruption.
