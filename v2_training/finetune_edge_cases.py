#!/usr/bin/env python3
"""
Edge Case Fine-Tuning for ByT5 Leetspeak V2
============================================
Quick targeted fine-tuning for specific patterns that failed in testing.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

# Load from v2, save back to v2
MODEL_PATH = "./byt5_leetspeak_model_v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")

# Edge cases that failed - repeat each 50x for emphasis
EDGE_CASES = [
    # brb patterns
    ("brb in 10", "be right back in 10"),
    ("brb 5 mins", "be right back 5 minutes"),
    ("brb", "be right back"),

    # g2g patterns
    ("g2g", "got to go"),
    ("g2g l8r m8", "got to go later mate"),
    ("g2g now", "got to go now"),
    ("sorry g2g", "sorry got to go"),
    ("g2g my mom is calling", "got to go my mom is calling"),

    # 1v1 patterns (preserve as gaming term)
    ("1v1 m3 br0", "1v1 me bro"),
    ("1v1 me", "1v1 me"),
    ("lets 1v1", "lets 1v1"),

    # Gaming abbreviations to preserve
    ("gg wp", "good game well played"),
    ("gg wp ez", "good game well played easy"),
    ("ez gam3", "easy game"),
    ("3z g4m3", "easy game"),

    # rekt patterns
    ("r3kt", "rekt"),
    ("r3kt th4t t34m", "rekt that team"),
    ("g3t r3kt", "get rekt"),

    # af patterns
    ("th4ts l1t af", "thats lit as fuck"),
    ("c00l af", "cool as fuck"),
    ("h4rd af", "hard as fuck"),

    # Cake is a lie (iconic)
    ("7h3 c4k3 1s 4 l13", "the cake is a lie"),
    ("th3 c4k3 15 4 l13", "the cake is a lie"),

    # nvm patterns
    ("nvm", "nevermind"),
    ("nvm 1 f0und 1t", "nevermind I found it"),
    ("nvm 1ts f1n3", "nevermind its fine"),

    # shouldof/couldof (common misspellings)
    ("sh0uld0f", "should have"),
    ("c0uld0f", "could have"),
    ("w0uld0f", "would have"),

    # Cold weather fix
    ("c0ld w34th3r", "cold weather"),
    ("c0ld 2d4y", "cold today"),
]

def main():
    print("=" * 60)
    print("EDGE CASE FINE-TUNING")
    print("=" * 60)

    # Load model
    print(f"[info] Loading model from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.gradient_checkpointing_enable()

    # Create dataset (repeat each case 50x)
    data = []
    for inp, tgt in EDGE_CASES:
        for _ in range(50):
            data.append({'input': inp, 'target': tgt})

    print(f"[info] Training on {len(data)} examples ({len(EDGE_CASES)} unique patterns)")

    # Tokenize
    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch['input'],
            max_length=128,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            batch['target'],
            max_length=128,
            truncation=True,
            padding=False,
        )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    dataset = Dataset.from_list(data)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['input', 'target'])

    # Training args - very gentle fine-tuning
    training_args = Seq2SeqTrainingArguments(
        output_dir="./edge_case_temp",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,  # VERY LOW to not break existing knowledge
        num_train_epochs=3,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="no",  # Don't save checkpoints
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # Train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    print("\n[info] Starting edge case fine-tuning...")
    trainer.train()

    # Save back to original location (overwrite)
    print(f"\n[info] Saving to {MODEL_PATH} (overwriting)...")
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    # Quick test
    print("\n" + "=" * 60)
    print("QUICK SANITY CHECK")
    print("=" * 60)

    test_inputs = ["brb in 10", "g2g l8r m8", "1v1 m3 br0", "th4ts l1t af", "nvm 1 f0und 1t"]
    model.eval()
    for text in test_inputs:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  {text} -> {result}")

    print("\n✓ Done! Model saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
