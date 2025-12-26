"""
WARNING: THIS SCRIPT FAILED.
attempting to fix the "too" edge case caused regressions even with these safety measures.
the model was restored to the previous checkpoint.

finetune_safe.py - ULTRA-SAFE micro-finetune for edge cases

This script is designed to gently nudge the model without catastrophic forgetting.
Key principles:
1. 95% preservation examples (things model already does well)
2. 5% new edge cases (the specific fixes we want)
3. VERY low learning rate (5e-6, 6x lower than before)
4. Long warmup period
5. Only 1 epoch
6. Gradient accumulation for stable updates

ALWAYS BACKUP YOUR MODEL BEFORE RUNNING THIS!
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import random
import os
from datetime import datetime

# ====================
# CONFIGURATION
# ====================
MODEL_PATH = "./byt5_leetspeak_model"
LEARNING_RATE = 5e-6          # VERY low - 6x lower than the script that broke it
NUM_EPOCHS = 1                # Only 1 epoch
BATCH_SIZE = 8                # Smaller batches
GRADIENT_ACCUMULATION = 4     # Effectively batch size 32 but more stable
WARMUP_RATIO = 0.2            # 20% warmup - gentle start
PRESERVATION_RATIO = 0.95     # 95% preservation, 5% new patterns

# ====================
# AUTO-BACKUP
# ====================
def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{MODEL_PATH}_backup_{timestamp}"
    if os.path.exists(MODEL_PATH):
        import shutil
        print(f"Creating automatic backup: {backup_path}")
        shutil.copytree(MODEL_PATH, backup_path)
        print(f"✓ Backup created!")
        return backup_path
    return None

# ====================
# EXAMPLE DATASETS
# ====================

# PRESERVATION EXAMPLES - Things the model already does well
PRESERVATION_EXAMPLES = [
    ("H3110 W0r1d!", "Hello World!"),
    ("1 l0v3 pr0gr4mm1ng", "I love programming"),
    ("Th15 15 4w350m3", "This is awesome"),
    ("1 4m h3r3", "I am here"),
    ("1 th1nk 50", "I think so"),
    ("1 d0n7 kn0w", "I dont know"),
    ("1 h4v3 2 c4t5", "I have 2 cats"),
    ("M33t m3 4t 3 PM", "Meet me at 3 PM"),
    ("1 n33d 2 g0", "I need to go"),
    ("1 h4v3 2 g0", "I have to go"),
    ("L8r m8", "Later mate"),
    ("gr8 j0b", "great job"),
    ("ur c00l", "your cool"),
    ("thx 4 h3lp1ng", "thanks for helping"),
]

# NEW PATTERNS - The edge cases we want to teach
NEW_EDGE_CASES = [
    ("1t5 2 l8", "its too late"),
    ("2 l8", "too late"),
    ("w4y 2 l8", "way too late"),
    ("th4t5 2 b4d", "thats too bad"),
    ("2 h4rd", "too hard"),
    ("2 345y", "too easy"),
    ("m3 2", "me too"),
    ("y0u 2", "you too"),
]

def main():
    backup_path = create_backup()

    print(f"\nLoading model from: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}!\n")

    print("Building balanced dataset...")
    examples = []

    # Calculate repetitions to achieve 95/5 split
    total_target = 2000
    preservation_count = int(total_target * PRESERVATION_RATIO)
    new_count = total_target - preservation_count

    preservation_reps = preservation_count // len(PRESERVATION_EXAMPLES)
    new_reps = new_count // len(NEW_EDGE_CASES)

    for _ in range(preservation_reps):
        for inp, tgt in PRESERVATION_EXAMPLES:
            examples.append({"input": inp, "target": tgt})

    for _ in range(new_reps):
        for inp, tgt in NEW_EDGE_CASES:
            examples.append({"input": inp, "target": tgt})

    random.shuffle(examples)
    print(f"  Total dataset size: {len(examples)} examples")

    def tokenize_fn(batch):
        model_inputs = tokenizer(batch["input"], max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(text_target=batch["target"], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_list(examples).map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    training_args = Seq2SeqTrainingArguments(
        output_dir="./safe_finetune_checkpoints",
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=25,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        report_to="none",
        lr_scheduler_type="cosine",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("\n🚀 Starting SAFE fine-tuning...")
    trainer.train()

    print("\n💾 Saving updated model...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"\n✅ Done! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
