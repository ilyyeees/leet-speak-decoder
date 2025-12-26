"""
WARNING: THIS SCRIPT FAILED.
it caused catastrophic forgetting (loss of capital I, etc.) when run.
see notebook for details.

finetune_too_edge_case.py - micro-finetune for the '2 -> too' edge case

this script fixes the edge case where "2 l8" should become "too late"
but the model was outputting "to late".

includes preservation examples to prevent forgetting 'to' and 'two'.

usage: python finetune_too_edge_case.py
"""
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# load the current model
model_path = "./byt5_leetspeak_model"
print(f"loading model from: {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"model loaded on {device}!\n")

# === THE DATASET ===
# focus heavily on 'too' cases, but mix in others to keep balance
examples = [
    # TARGET: 2 -> too (the problem area)
    ("1t5 2 l8", "its too late"),
    ("2 l8", "too late"),
    ("w4y 2 l8", "way too late"),
    ("th4t5 2 b4d", "thats too bad"),
    ("2 h4rd", "too hard"),
    ("2 345y", "too easy"),
    ("m3 2", "me too"),
    ("y0u 2", "you too"),
    ("1t 15 2 much", "it is too much"),
    ("d0n7 g0 2 f4r", "dont go too far"),
    ("2 c0ld", "too cold"),
    ("2 h0t", "too hot"),
    ("2 g00d", "too good"),
    ("2 f4st", "too fast"),
    ("2 sl0w", "too slow"),
    ("2 34rly", "too early"),
    ("2 l0ng", "too long"),
    ("2 sh0rt", "too short"),

    # PRESERVATION: 2 -> to (remind it of the preposition)
    ("g0 2 th3 st0r3", "go to the store"),
    ("1 w4nt 2 sl33p", "I want to sleep"),
    ("n1c3 2 m33t u", "nice to meet you"),
    ("1 n33d 2 g0", "I need to go"),
    ("1 h4v3 2 g0", "I have to go"),
    ("t1m3 2 g0", "time to go"),

    # PRESERVATION: 2 -> two (remind it of the number)
    ("1 h4v3 2 d0g5", "I have 2 dogs"),
    ("ju5t 2 m1nu735", "just 2 minutes"),
    ("ph453 2", "phase 2"),
    ("v3rs10n 2.0", "version 2.0"),
    ("ch4pt3r 2", "chapter 2"),
]

# duplicate the list to create a proper batch
train_data = examples * 50  # ~1500 examples

def format_data(data):
    inputs = [x[0] for x in data]
    targets = [x[1] for x in data]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = Dataset.from_dict({"input": [x[0] for x in train_data], "target": [x[1] for x in train_data]})
tokenized_dataset = dataset.map(format_data, batched=True)

# low learning rate to gently nudge the weights
training_args = Seq2SeqTrainingArguments(
    output_dir="./finetune_checkpoints",
    per_device_train_batch_size=8,
    learning_rate=3e-5,  # gentle learning rate
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="no",
    bf16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
)

print("starting micro-finetune for 'too' context...")
print("-" * 50)
trainer.train()

print("\nsaving updated model...")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"done! model saved to {model_path}")
