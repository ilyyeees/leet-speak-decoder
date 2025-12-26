"""
finetune_weak_patterns.py - quick fine-tune script for weak patterns

this script loads an already-trained model and fine-tunes it further
on specific patterns that the model struggles with, like:
- 8 -> -ate words (l8r, w8, gr8, m8, h8)
- u -> you, r -> are
- thx -> thanks, ur -> your
- w/o -> without

usage:
    python finetune_weak_patterns.py

this is much faster than training from scratch because we're only
teaching the model the specific patterns it missed.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import random

# configuration
MODEL_PATH = "./byt5_leetspeak_model"  # path to your trained model
OUTPUT_PATH = "./byt5_leetspeak_model"  # overwrite or use _v2 for new version
LEARNING_RATE = 5e-5  # lower than initial training
NUM_EPOCHS = 2
BATCH_SIZE = 16

def main():
    # load the existing trained model
    print(f"loading trained model from: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"model loaded on {device}!\n")

    # targeted examples for weak patterns
    weak_pattern_examples = [
        # === 8 -> -ate patterns ===
        ("l8r", "later"),
        ("l8r m8", "later mate"),
        ("c u l8r", "see you later"),
        ("w8", "wait"),
        ("w8 4 m3", "wait for me"),
        ("w8 h3r3", "wait here"),
        ("gr8", "great"),
        ("gr8 j0b", "great job"),
        ("th4t5 gr8", "thats great"),
        ("m8", "mate"),
        ("h3y m8", "hey mate"),
        ("th4nk5 m8", "thanks mate"),
        ("h8", "hate"),
        ("1 h8 th15", "I hate this"),
        ("d0n7 h8 m3", "dont hate me"),
        ("sk8", "skate"),
        ("sk8b04rd", "skateboard"),
        ("cr8", "create"),
        ("cr8 4 n3w f1l3", "create a new file"),
        ("st8", "state"),
        ("un1t3d st8s", "united states"),
        ("d8", "date"),
        ("wh4t5 th3 d8?", "whats the date?"),
        ("r8", "rate"),
        ("1 r8 th15 10/10", "I rate this 10/10"),
        ("pl8", "plate"),
        ("g8", "gate"),
        ("f8", "fate"),

        # === u -> you ===
        ("u", "you"),
        ("u r c00l", "you are cool"),
        ("1 l0v3 u", "I love you"),
        ("th4nk u", "thank you"),
        ("h0w r u", "how are you"),
        ("wh3r3 r u", "where are you"),
        ("r u 0k", "are you ok"),
        ("r u th3r3", "are you there"),
        ("r u c0m1ng", "are you coming"),
        ("c u l8r", "see you later"),
        ("c u 2m0rr0w", "see you tomorrow"),
        ("c u s00n", "see you soon"),
        ("m155 u", "miss you"),
        ("n33d u", "need you"),
        ("w4nt u", "want you"),

        # === r -> are ===
        ("r", "are"),
        ("th3y r h3r3", "they are here"),
        ("w3 r r34dy", "we are ready"),
        ("u r 4w350m3", "you are awesome"),
        ("th1ng5 r g00d", "things are good"),
        ("wh0 r u", "who are you"),
        ("wh4t r u d01ng", "what are you doing"),
        ("h0w r th1ng5", "how are things"),

        # === ur -> your ===
        ("ur", "your"),
        ("ur c00l", "your cool"),
        ("ur 4w350m3", "your awesome"),
        ("wh4t5 ur n4m3", "whats your name"),
        ("1 l1k3 ur 5tyl3", "I like your style"),
        ("ur th3 b35t", "your the best"),
        ("ur fr13nd", "your friend"),

        # === thx -> thanks ===
        ("thx", "thanks"),
        ("thx m8", "thanks mate"),
        ("thx 4 h3lp1ng", "thanks for helping"),
        ("thx 4 3v3ryth1ng", "thanks for everything"),
        ("thx 4 th3 h31p", "thanks for the help"),
        ("thx 4 c0m1ng", "thanks for coming"),
        ("thx 4 th4t", "thanks for that"),
        ("thx 4 l3tt1ng m3 kn0w", "thanks for letting me know"),

        # === w/o -> without ===
        ("w/o", "without"),
        ("w/o u", "without you"),
        ("w/o h31p", "without help"),
        ("1 c4n7 d0 1t w/o u", "I cant do it without you"),

        # === combined patterns ===
        ("thx m8, c u l8r", "thanks mate, see you later"),
        ("ur gr8 m8", "your great mate"),
        ("r u fr33 l8r?", "are you free later?"),
        ("w8 4 m3, 1ll b r1ght th3r3", "wait for me, ill be right there"),
        ("1 h8 2 w8", "I hate to wait"),
        ("th4t w45 gr8, thx!", "that was great, thanks!"),
        ("wh3r3 r u m8?", "where are you mate?"),
        ("c u l8r m8, thx 4 3v3ryth1ng", "see you later mate, thanks for everything"),
    ]

    # multiply examples for better learning
    all_examples = []
    for _ in range(20):  # repeat 20x for emphasis
        for inp, tgt in weak_pattern_examples:
            all_examples.append({"input": inp, "target": tgt})

    random.shuffle(all_examples)
    print(f"created {len(all_examples)} targeted training examples\n")

    # tokenize function
    def tokenize_fn(batch):
        inputs = tokenizer(batch["input"], max_length=128, truncation=True, padding=False)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target"], max_length=128, truncation=True, padding=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    # create dataset
    dataset = Dataset.from_list(all_examples)
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    # training arguments - lower learning rate for fine-tuning
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_PATH + "_checkpoint",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_steps=50,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    # train
    print("starting fine-tuning on weak patterns...")
    print("-" * 50)
    trainer.train()

    # save updated model
    print("\nsaving improved model...")
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"done! model saved to: {OUTPUT_PATH}")
    print("run your accuracy test again to see improvements!")


if __name__ == "__main__":
    main()
