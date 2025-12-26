# v1 training

the original byt5 leetspeak decoder, trained on synthetic data.

---

## overview

v1 was trained from scratch using:
- **wikitext-2**: ~30k clean english sentences
- **samsum**: ~10k conversational utterances
- **synthetic corruption**: programmatic leetspeak generator

total: ~40k training examples

---

## performance

| metric | score |
|--------|-------|
| bleu | 94.8 |
| cer | 0.7% |
| test accuracy | ~71% (on real reddit) |

the model performs well on synthetic test cases but struggles with real-world reddit slang.

---

## files

| file | description |
|------|-------------|
| `byt5_leetspeak_decoder.py` | main training script (40k samples, 3 epochs) |
| `byt5_leetspeak_decoder.ipynb` | colab notebook version |
| `finetune_weak_patterns.py` | fixes specific weak patterns (`8→ate`, `thx→thanks`) |
| `finetune_too_edge_case.py` | attempted fix for `2→too` edge case |
| `finetune_safe.py` | (failed) safe micro-finetuning attempt |

---

## training

### on google colab (recommended)

1. upload `byt5_leetspeak_decoder.ipynb` to colab
2. enable gpu runtime (t4 or better)
3. run all cells

### locally

```bash
# install dependencies
pip install transformers datasets torch sentencepiece evaluate jiwer sacrebleu

# run training
python byt5_leetspeak_decoder.py
```

requires ~8gb vram (rtx 3060 or better).

---

## corruption engine

the synthetic leetspeak generator includes:

**character substitutions:**
- `a→4`, `e→3`, `i→1`, `o→0`, `s→5`, `t→7`
- complex: `h→#`, `w→\/\/`, `m→|\/|`

**word substitutions:**
- `you→u`, `are→r`, `to→2`, `for→4`
- `thanks→thx`, `great→gr8`, `later→l8r`

**noise injection:**
- vowel dropping (`text→txt`)
- case chaos (`tHiS iS lEeT`)
- random insertions/deletions

---

## limitations

v1 struggles with:
- real reddit slang (`tbh`, `rn`, `ngl`, `idk`)
- gaming abbreviations (`gg`, `wp`, `ez`)
- context-heavy translations

these are addressed in [v2](../v2_training/README.md).

---

## model

released on huggingface: [ilyyeees/byt5-leetspeak-decoder](https://huggingface.co/ilyyeees/byt5-leetspeak-decoder)
