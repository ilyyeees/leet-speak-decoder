#!/usr/bin/env python3
"""
Train ByT5 V2 - Continue Training on Real Reddit Data
======================================================
Continues training the EXISTING byt5_leetspeak_model on new synthetic data
generated from real Reddit comments (via Qwen + corruption).

This script:
1. Loads the EXISTING trained model (not from scratch!)
2. Fine-tunes it on the new 15k+ synthetic pairs
3. Uses optimized settings from the original training
4. Includes sanity-check callbacks and preservation examples

Requirements:
    pip install transformers datasets torch sentencepiece evaluate

Usage:
    python3 train_model_v2.py --data training_data.jsonl
"""

import json
import argparse
import random
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)

# ============================================================================
# CONFIGURATION (matched to original byt5_leetspeak_decoder.py settings)
# ============================================================================

# CRITICAL: Load from existing model ONLY!
# Uses HuggingFace model ID - will download automatically
EXISTING_MODEL_PATH = "ilyyeees/byt5-leetspeak-decoder"

# Output
OUTPUT_DIR = "./byt5_leetspeak_model_v2"

# Training hyperparameters (optimized for continuing training)
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8            # Middle ground - should fit with gradient checkpointing
GRADIENT_ACCUM_STEPS = 4  # Effective batch: 32
LEARNING_RATE = 5e-5      # LOWER than original (3e-4) to avoid forgetting!
NUM_EPOCHS = 3
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01

# Seed
SEED = 42
set_seed(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")
if torch.cuda.is_available():
    print(f"[info] GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# SANITY CHECK CALLBACK
# ============================================================================

class LogSampleCallback(TrainerCallback):
    """
    Log a few sample translations at the end of every epoch
    to verify the model isn't breaking.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.test_sentences = [
            "idk wh4t 2 d0 tbh",       # Abbreviation test
            "1 h4v3 2 c4t5",            # Number preservation (2 cats)
            "1t5 2 l8",                 # The "too late" edge case
            "c u l8r m8",               # Multiple patterns
            "thx 4 th3 h31p",           # thanks for the help
        ]

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n{'='*50}")
        print(f"[Epoch {state.epoch:.0f} SANITY CHECK - Sample Translations]")
        print('='*50)
        model.eval()

        # Move inputs to device (GPU)
        device = next(model.parameters()).device
        inputs = self.tokenizer(self.test_sentences, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for input_text, output in zip(self.test_sentences, decoded):
            print(f"  IN:  {input_text}")
            print(f"  OUT: {output}")
            print()

        model.train()
        print('='*50 + "\n")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl_data(file_path: str):
    """
    Load training data from JSONL file.
    Expects format: {"input_text": "...", "target_text": "..."}
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # Handle both formats from our scripts
                inp = item.get('input_text') or item.get('input') or item.get('corrupted', '')
                tgt = item.get('target_text') or item.get('target') or item.get('formal', '')

                if inp and tgt:
                    data.append({'input': inp, 'target': tgt})
            except json.JSONDecodeError:
                continue

    print(f"[info] Loaded {len(data)} examples from {file_path}")
    return data


def prepare_dataset(data, tokenizer, train_split=0.9):
    """
    Prepare HuggingFace dataset with train/val split.

    WARNING: If your data has multiplexed variants (3x per comment),
    you should split BEFORE multiplexing in corrupt_to_leetspeak.py
    to avoid validation leakage. This split is a fallback.
    """
    random.shuffle(data)
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # =========================================================================
    # PRESERVATION EXAMPLES (Anti-Forgetting Insurance)
    # These remind the model: "Don't change text that is already clean!"
    # =========================================================================
    preservation_examples = [
        # Clean English -> Clean English (identity mapping)
        ("The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog."),
        ("I have 2 cats and 3 dogs.", "I have 2 cats and 3 dogs."),
        ("Please meet me at 5 PM.", "Please meet me at 5 PM."),
        ("Version 2.0 is released.", "Version 2.0 is released."),
        ("She is 25 years old.", "She is 25 years old."),
        ("The meeting is at 3 o'clock.", "The meeting is at 3 o'clock."),
        ("I need 5 minutes to finish.", "I need 5 minutes to finish."),
        ("Page 42 of the book.", "Page 42 of the book."),
        ("There are 10 items in the list.", "There are 10 items in the list."),
        ("We have 100 users online.", "We have 100 users online."),

        # Key number context examples (from original training)
        ("I have 2 cats", "I have 2 cats"),
        ("Meet me at 3 PM", "Meet me at 3 PM"),
        ("Version 2.0 is out", "Version 2.0 is out"),
        ("I got 100 points", "I got 100 points"),
        ("Born in 1990", "Born in 1990"),

        # "too" vs "to" vs "2" edge cases
        ("It is too late", "It is too late"),
        ("I want to go home", "I want to go home"),
        ("That is too much", "That is too much"),
    ]

    # Add preservation examples 50 times for emphasis
    preservation_data = []
    for _ in range(50):
        for inp, tgt in preservation_examples:
            preservation_data.append({'input': inp, 'target': tgt})

    # Add to training set ONLY (not validation!)
    train_data.extend(preservation_data)
    random.shuffle(train_data)

    print(f"[info] Added {len(preservation_data)} preservation examples to train set")
    print(f"[info] Final Train: {len(train_data)}, Validation: {len(val_data)}")

    def tokenize_function(batch):
        model_inputs = tokenizer(
            batch['input'],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=False,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch['target'],
                max_length=MAX_TARGET_LENGTH,
                truncation=True,
                padding=False,
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenize with multiprocessing
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'target'],
        desc="Tokenizing train",
        num_proc=4,
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'target'],
        desc="Tokenizing val",
        num_proc=4,
    )

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })


# ============================================================================
# TRAINING
# ============================================================================

def train(data_path: str):
    print("=" * 60)
    print("ByT5 LEETSPEAK V2 - CONTINUED TRAINING")
    print("=" * 60)

    # Load data
    data = load_jsonl_data(data_path)

    # Load model from HuggingFace or local path
    model_path = EXISTING_MODEL_PATH
    print(f"[info] Loading model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model = model.to(device)
        model.gradient_checkpointing_enable()  # Save VRAM
        print(f"[info] Successfully loaded model (with gradient checkpointing)!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[ERROR] Make sure the model path or HuggingFace ID is correct.")
        exit(1)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Total parameters: {total_params:,}")

    # Prepare dataset
    dataset = prepare_dataset(data, tokenizer)

    # Training arguments (matched to original script style)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch settings
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,

        # Learning rate (LOWER for continued training!)
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,

        # Epochs
        num_train_epochs=NUM_EPOCHS,

        # Evaluation & Saving
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,

        # Performance
        bf16=torch.cuda.is_available(),  # Use bf16 on modern GPUs
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # Logging
        logging_steps=100,
        report_to="none",

        # Generation (for predict_with_generate if needed)
        predict_with_generate=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
    )

    # Trainer with callbacks
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            LogSampleCallback(tokenizer),  # Sanity check every epoch!
        ],
    )

    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60 + "\n")

    trainer.train()

    # Save final model
    print(f"\n[info] Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run your accuracy test on the new model")
    print("  2. If accuracy is good, replace byt5_leetspeak_model with v2")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training ByT5 on new data")
    parser.add_argument("--data", required=True, help="Path to JSONL training data")
    args = parser.parse_args()

    train(args.data)
