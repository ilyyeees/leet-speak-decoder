# V2 Training Pipeline

## Why V2?

The V1 model (trained on WikiText + synthetic leetspeak) achieved **71% accuracy**. While decent, it struggles with:
- **Real-world slang context** (e.g., `rn` → "run" vs "right now")
- **Gaming/Reddit abbreviations** (`tbh`, `idk`, `ngl`)
- **Heavy leetspeak combinations** (`1 h4v3 2 g0 2 th3 5t0r3`)

**V2 Goal**: Train on **real Reddit comments** to achieve **95%+ accuracy**.

---

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. SCRAPING    │ ──► │  2. TRANSLATION  │ ──► │  3. CORRUPTION  │ ──► │  4. TRAINING    │
│  (Reddit HTML)  │     │  (Qwen on Cloud) │     │  (Synthetic)    │     │  (Continue V1)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Step 1: Scraping (`data_pipeline/scraping/`)

**Script**: `scrape_reddit_html.py`
**Output**: `raw_comments.jsonl` (~5,000 comments)

Scrapes real comments from gaming/meme subreddits where leetspeak and slang are common.

---

## Step 2: Translation (`data_pipeline/processing/`)

**Scripts**: `translate_with_ollama.py` (primary) or `translate_with_gemini.py` (deprecated)

### Why Not Gemini?
We initially tried using the Gemini API (`translate_with_gemini.py`), but it failed due to:
- **Rate limits**: Only 5 requests/minute on free tier
- **Processing time**: Would take 16+ hours for 5,000 comments
- **Cost**: Paid tier too expensive for this volume

**Solution**: Self-host **Qwen 2.5 32B** on a cloud GPU via TensorDock.

### Cloud Setup (TensorDock)
We use **TensorDock** to rent a GPU server for fast LLM inference:

1. **Rent an RTX 3090** (~$0.25/hour)
2. **Install Ollama** + pull **Qwen 2.5 32B** model
3. **Run translation**: Raw Reddit → Clean Formal English
4. **Download results** and **destroy server**

**Why Cloud?**
- Qwen 32B needs 20GB VRAM (doesn't fit on consumer GPUs)
- Processing 5,000 comments takes ~30 minutes on cloud vs hours locally

**Output**: `translated_pairs.jsonl`
```json
{"original": "idk wh4t 2 d0 tbh", "formal": "I don't know what to do, to be honest."}
```

---

## Step 3: Corruption (`data_pipeline/processing/`)

**Script**: `corrupt_to_leetspeak.py`

Takes the **original Reddit comments** and corrupts them FURTHER to create harder training examples:

- **Dynamic Intensity**: 50% Light / 30% Medium / 20% Heavy
- **3x Multiplexing**: Each comment generates 3 variants (5k → 15k pairs)
- **Global Shuffle**: Variants are scattered to prevent overfitting

**Output**: `training_data.jsonl`
```json
{"input_text": "1dk wh47 2 d0 7bh", "target_text": "I don't know what to do, to be honest."}
```

---

## Step 4: Training

**Script**: `train_model_v2.py`

Continues training from the **existing V1 model** (`ilyyeees/byt5-leetspeak-decoder` on HuggingFace):

- **Lower learning rate** (5e-5) to prevent forgetting
- **Preservation examples** (clean English → clean English)
- **Sanity check callback** (logs sample translations each epoch)
- **Early stopping** (prevents overtraining)

### Cloud Training (TensorDock)
For faster training, rent an **RTX 4090** (~$0.40/hour):

```bash
pip install transformers datasets torch
python3 train_model_v2.py --data training_data.jsonl
```

**Output**: `./byt5_leetspeak_model_v2/`

---

## Quick Start

```bash
# 1. Generate training data (after getting translated_pairs.jsonl)
cd data_pipeline/processing
python3 corrupt_to_leetspeak.py -i translated_pairs.jsonl -o training_data.jsonl

# 2. Train (on GPU machine)
cd ../..
python3 train_model_v2.py --data data_pipeline/processing/training_data.jsonl
```

---

## File Structure

```
v2_training/
├── train_model_v2.py           # Main training script
├── data_pipeline/
│   ├── scraping/
│   │   ├── scrape_reddit_html.py
│   │   └── raw_comments.jsonl
│   └── processing/
│       ├── translate_with_ollama.py
│       ├── translate_with_gemini.py
│       └── corrupt_to_leetspeak.py
└── README.md                   # This file
```
