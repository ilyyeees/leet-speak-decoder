#!/usr/bin/env python3
"""
================================================================================
ByT5 LEETSPEAK DECODER V3 - PRODUCTION TRAINING SCRIPT
================================================================================

The definitive training script for v3, optimized for 2x RTX 5090.

Key improvements over v2:
1. Continues from v2 model (not from scratch) for faster convergence
2. Reverse Pipeline: Clean -> LLM Corrupt -> Train (not Dirty -> LLM Clean)
3. Hybrid corruption: LLM handles semantics, script handles visuals
4. 15x multiplexing: 50k LLM calls -> 750k training pairs
5. Adversarial filtering: Keeps only hard examples v2 can't decode
6. Multi-phase training: Curriculum learning from easy to hard
7. Better base data: WikiText + ELI5 + LLM-generated content

Hardware Configuration:
    - Optimized for 2x RTX 5090 (32GB VRAM each)
    - Model loaded to RAM first (for limited storage instances)
    - BF16 mixed precision for speed + stability
    - Gradient checkpointing enabled by default

Training Pipeline:
    1. Download v2 model to RAM (avoids disk space issues)
    2. Load 500k+ synthetic pairs from data_pipeline
    3. (Optional) Filter using v2 model for hard negatives
    4. Curriculum learning: easy -> medium -> hard intensity
    5. Progressive training with periodic validation

Usage:
    # Basic training (continues from v2)
    python train_model_v3.py --data training_data_v3.jsonl

    # Train from scratch (not recommended)
    python train_model_v3.py --data training_data_v3.jsonl --from-scratch

    # With adversarial filtering
    python train_model_v3.py --data training_data_v3.jsonl --adversarial

    # Resume from checkpoint
    python train_model_v3.py --data training_data_v3.jsonl --resume checkpoint-5000

    # Multi-GPU with accelerate
    accelerate launch train_model_v3.py --data training_data_v3.jsonl

================================================================================
"""

import os
import json
import argparse
import random
import gc
import math
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
import numpy as np
from tqdm.auto import tqdm

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import snapshot_download

import evaluate

# Optional: accelerate for multi-GPU
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


# ============================================================================
# V2 MODEL LOADING (TO RAM)
# ============================================================================

V2_MODEL_HUB = "ilyyeees/byt5-leetspeak-decoder"
RAM_CACHE_DIR = "/dev/shm/hf_cache"  # Linux shared memory (RAM-backed)


def get_ram_cache_dir() -> str:
    """
    Get a RAM-backed cache directory for model storage.
    Uses /dev/shm on Linux (shared memory), falls back to tempfile.
    """
    # Try /dev/shm first (Linux RAM disk)
    if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
        cache_dir = RAM_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[CACHE] Using RAM-backed storage: {cache_dir}")
        return cache_dir

    # Fallback to system temp (might be RAM on some systems)
    cache_dir = tempfile.mkdtemp(prefix="hf_leetspeak_")
    print(f"[CACHE] Using temp directory: {cache_dir}")
    return cache_dir


def download_v2_model_to_ram(force_download: bool = False) -> str:
    """
    Download the v2 model to RAM-backed storage.

    This avoids disk space issues on instances with limited storage
    but plenty of RAM.

    Returns:
        Path to the downloaded model in RAM
    """
    cache_dir = get_ram_cache_dir()
    model_dir = os.path.join(cache_dir, "byt5-leetspeak-v2")

    # Check if already downloaded
    if os.path.exists(model_dir) and not force_download:
        config_file = os.path.join(model_dir, "config.json")
        if os.path.exists(config_file):
            print(f"[V2 MODEL] Already cached in RAM: {model_dir}")
            return model_dir

    print(f"\n{'='*60}")
    print("DOWNLOADING V2 MODEL TO RAM")
    print('='*60)
    print(f"  Source: {V2_MODEL_HUB}")
    print(f"  Destination: {model_dir}")
    print(f"  (This keeps disk usage low)")
    print()

    try:
        # Download to RAM-backed directory
        snapshot_download(
            repo_id=V2_MODEL_HUB,
            local_dir=model_dir,
            local_dir_use_symlinks=False,  # Copy files, don't symlink
        )
        print(f"[V2 MODEL] Downloaded successfully to RAM!")
        return model_dir

    except Exception as e:
        print(f"[V2 MODEL] Download failed: {e}")
        print(f"[V2 MODEL] Falling back to direct HuggingFace loading...")
        return V2_MODEL_HUB


def cleanup_ram_cache():
    """Clean up RAM cache when done."""
    if os.path.exists(RAM_CACHE_DIR):
        try:
            shutil.rmtree(RAM_CACHE_DIR)
            print(f"[CACHE] Cleaned up RAM cache: {RAM_CACHE_DIR}")
        except Exception as e:
            print(f"[CACHE] Failed to clean up: {e}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Comprehensive configuration for v3 training.
    Optimized for 2x RTX 5090 (32GB VRAM each).
    """

    # === Model Configuration ===
    base_model: str = V2_MODEL_HUB  # Continue from v2 by default!
    continue_from: Optional[str] = None  # Path to checkpoint to continue from
    use_v2_for_adversarial: str = V2_MODEL_HUB  # v2 model for filtering
    load_model_to_ram: bool = True  # Download to RAM to save disk space

    # === Data Configuration ===
    max_input_length: int = 384  # Increased for longer sequences
    max_target_length: int = 384
    train_split: float = 0.95  # More training data, less validation

    # === Training Hyperparameters (2x RTX 5090 Optimized) ===
    # RTX 5090 has 32GB VRAM, similar compute to A100 40GB
    # Batch settings for 2x RTX 5090
    per_device_train_batch_size: int = 24  # 32GB VRAM allows this
    per_device_eval_batch_size: int = 48
    gradient_accumulation_steps: int = 3  # Effective batch: 24*2*3 = 144

    # Learning rate schedule (lower for fine-tuning from v2)
    learning_rate: float = 5e-5  # Lower than fresh training since we have v2
    lr_scheduler_type: str = "cosine"  # Cosine annealing works well
    warmup_ratio: float = 0.03  # 3% warmup for fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training duration
    num_train_epochs: int = 3  # Fewer epochs since we're fine-tuning
    max_steps: int = -1  # Set to positive number to override epochs

    # === Mixed Precision ===
    bf16: bool = True  # BF16 is great on RTX 5090 (Blackwell architecture)
    fp16: bool = False
    tf32: bool = True  # Enable TF32 for faster matrix ops

    # === Memory Optimization ===
    gradient_checkpointing: bool = True  # Saves VRAM at cost of speed
    optim: str = "adamw_torch_fused"  # Fused AdamW is faster

    # === Evaluation & Saving ===
    eval_strategy: str = "steps"
    eval_steps: int = 1500
    save_strategy: str = "steps"
    save_steps: int = 1500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # === Logging ===
    logging_steps: int = 50
    logging_first_step: bool = True
    report_to: str = "tensorboard"  # Options: tensorboard, wandb, none

    # === Paths ===
    output_dir: str = "./byt5_leetspeak_v3"
    logging_dir: str = "./logs_v3"
    cache_dir: str = "./.cache"

    # === Advanced Options ===
    seed: int = 42
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4

    # Curriculum learning phases
    curriculum_phases: int = 3  # Easy -> Medium -> Hard

    # Adversarial filtering
    adversarial_threshold: float = 0.8  # Keep examples where v2 score < this


def get_model_path(config: TrainingConfig) -> str:
    """
    Get the model path, downloading to RAM if needed.
    """
    model_source = config.continue_from or config.base_model

    # If it's the HuggingFace v2 model and we want RAM loading
    if config.load_model_to_ram and model_source == V2_MODEL_HUB:
        return download_v2_model_to_ram()

    return model_source


# ============================================================================
# PRESERVATION EXAMPLES
# ============================================================================

# These examples teach the model to preserve clean text
PRESERVATION_EXAMPLES = [
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
    ("The event starts at 9 AM sharp.", "The event starts at 9 AM sharp."),
    ("I scored 95 out of 100.", "I scored 95 out of 100."),
    ("Flight 747 departs at 6 PM.", "Flight 747 departs at 6 PM."),

    # Key number context (standalone numbers)
    ("I have 2 cats", "I have 2 cats"),
    ("Meet me at 3 PM", "Meet me at 3 PM"),
    ("Version 2.0 is out", "Version 2.0 is out"),
    ("I got 100 points", "I got 100 points"),
    ("Born in 1990", "Born in 1990"),
    ("Room 404 not found", "Room 404 not found"),

    # "too" vs "to" vs "2" edge cases
    ("It is too late", "It is too late"),
    ("I want to go home", "I want to go home"),
    ("That is too much", "That is too much"),
    ("I need to sleep", "I need to sleep"),
    ("The soup is too hot", "The soup is too hot"),

    # Common phrases that should stay clean
    ("How are you doing today?", "How are you doing today?"),
    ("Thank you very much!", "Thank you very much!"),
    ("I appreciate your help.", "I appreciate your help."),
    ("Have a great day!", "Have a great day!"),
]

# Edge case examples for specific patterns the model struggles with
EDGE_CASE_EXAMPLES = [
    # 2/to/too disambiguation
    ("1t5 2 l8", "It is too late"),
    ("im going 2 sleep", "I am going to sleep"),
    ("thats 2 much 4 me", "That is too much for me"),
    ("i have 2 go 2 work", "I have to go to work"),
    ("u need 2 c this", "You need to see this"),

    # Heavy leetspeak
    ("1 h4v3 2 c4t5 4nd 3 d0g5", "I have 2 cats and 3 dogs"),
    ("wh4t t1m3 15 th3 m33t1ng", "What time is the meeting"),
    ("th4nk5 4 y0ur h3lp", "Thanks for your help"),
    ("th15 15 50 c00l", "This is so cool"),

    # Mixed intensity
    ("idk wh4t 2 do tbh", "I don't know what to do to be honest"),
    ("c u l8r m8", "See you later mate"),
    ("thx 4 th3 h31p", "Thanks for the help"),
    ("gonna b l8 sry", "Going to be late sorry"),

    # Gaming/internet culture
    ("that was ez gg", "That was easy good game"),
    ("ur trash ngl", "You are trash not gonna lie"),
    ("diff gap", "Difference gap"),
    ("skill issue fr", "Skill issue for real"),

    # Gen Z slang
    ("thats bussin fr fr", "That's really good for real for real"),
    ("no cap this slaps", "No lie this is great"),
    ("lowkey fire ngl", "Kind of great not gonna lie"),
    ("its giving main character", "It's giving main character energy"),

    # Phonetic spellings
    ("prolly gonna rain 2day", "Probably going to rain today"),
    ("wut r u doin rn", "What are you doing right now"),
    ("i wanna go 2 da movies", "I want to go to the movies"),
    ("dat was kewl af", "That was cool as fuck"),

    # Complex multi-pattern
    ("1 d0nt kn0w wut 2 d0 4 r34l", "I don't know what to do for real"),
    ("ur prolly r1ght tbh n0 c4p", "You're probably right to be honest no cap"),
]


# ============================================================================
# SANITY CHECK CALLBACK
# ============================================================================

class SanityCheckCallback(TrainerCallback):
    """
    Enhanced callback that logs sample translations during training.
    Helps monitor model quality and catch regressions early.
    """

    def __init__(self, tokenizer, test_interval: int = 1000):
        self.tokenizer = tokenizer
        self.test_interval = test_interval
        self.last_step = -1

        # Comprehensive test cases covering all difficulty levels
        self.test_cases = [
            # === EASY (should nail these) ===
            ("hello there", "Hello there"),
            ("thx 4 the help", "Thanks for the help"),
            ("c u l8r", "See you later"),

            # === MEDIUM (common patterns) ===
            ("idk wh4t 2 d0 tbh", "I don't know what to do to be honest"),
            ("1 h4v3 2 c4t5", "I have 2 cats"),
            ("ur prolly right", "You're probably right"),

            # === HARD (edge cases) ===
            ("1t5 2 l8", "It is too late"),
            ("th4t5 2 much 4 m3", "That's too much for me"),
            ("1 n33d 2 g0 2 w0rk", "I need to go to work"),

            # === VERY HARD (complex patterns) ===
            ("wh4t t1m3 15 th3 m33t1ng @ 3 pm", "What time is the meeting at 3 PM"),
            ("1 g0t 100 p01nt5 0n th3 t35t", "I got 100 points on the test"),

            # === PRESERVATION (should NOT change) ===
            ("I have 2 cats", "I have 2 cats"),
            ("The meeting is at 3 PM", "The meeting is at 3 PM"),
        ]

    def on_step_end(self, args, state: TrainerState, control: TrainerControl,
                    model=None, **kwargs):
        """Run sanity check every N steps."""
        if state.global_step - self.last_step < self.test_interval:
            return

        self.last_step = state.global_step
        self._run_sanity_check(model, state.global_step)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl,
                     model=None, **kwargs):
        """Also run at end of each epoch."""
        self._run_sanity_check(model, state.global_step, is_epoch_end=True)

    def _run_sanity_check(self, model, step: int, is_epoch_end: bool = False):
        """Execute the sanity check."""
        if model is None:
            return

        header = "EPOCH END" if is_epoch_end else f"Step {step}"
        print(f"\n{'='*60}")
        print(f"SANITY CHECK @ {header}")
        print('='*60)

        model.eval()
        device = next(model.parameters()).device

        correct = 0
        total = len(self.test_cases)

        # Process in batches for efficiency
        inputs = [tc[0] for tc in self.test_cases]
        expected = [tc[1] for tc in self.test_cases]

        tokenized = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_length=256,
                num_beams=4,
                early_stopping=True,
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for inp, exp, out in zip(inputs, expected, decoded):
            # Check if output matches expected (case-insensitive, strip whitespace)
            match = out.strip().lower() == exp.strip().lower()
            if match:
                correct += 1

            status = "✓" if match else "✗"
            print(f"  {status} IN:  {inp}")
            print(f"    EXP: {exp}")
            print(f"    OUT: {out}")
            print()

        accuracy = correct / total * 100
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        print('='*60 + "\n")

        model.train()


# ============================================================================
# METRICS CALLBACK
# ============================================================================

class MetricsCallback(TrainerCallback):
    """Track and log detailed metrics during training."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs: Dict = None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((state.global_step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.learning_rates.append((state.global_step, logs["learning_rate"]))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(data_path: str, config: TrainingConfig) -> List[Dict]:
    """
    Load and validate training data from JSONL file(s).

    Handles multiple input formats:
    - Single JSONL file
    - Directory of JSONL files
    - Glob pattern

    Returns:
        List of dicts with 'input' and 'target' keys
    """
    data = []
    data_path = Path(data_path)

    # Handle different input types
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = list(data_path.glob("*.jsonl"))
    else:
        files = list(Path(".").glob(str(data_path)))

    print(f"[DATA] Loading from {len(files)} file(s)...")

    for file_path in tqdm(files, desc="Loading files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    # Handle various field names
                    inp = (item.get('input_text') or item.get('input') or
                           item.get('corrupted') or item.get('source', ''))
                    tgt = (item.get('target_text') or item.get('target') or
                           item.get('formal') or item.get('clean', ''))

                    if inp and tgt:
                        # Basic validation
                        if len(inp) > 10 and len(tgt) > 10:
                            data.append({'input': inp.strip(), 'target': tgt.strip()})

                except json.JSONDecodeError as e:
                    if line_num < 10:  # Only warn for first few
                        print(f"[WARN] JSON error at {file_path}:{line_num}: {e}")
                    continue

    print(f"[DATA] Loaded {len(data):,} valid examples")
    return data


def compute_difficulty_score(text: str) -> float:
    """
    Estimate difficulty of a leetspeak sample.
    Higher score = more difficult to decode.

    Factors:
    - Ratio of non-alpha characters
    - Presence of multi-char substitutions
    - Number/letter ambiguity
    """
    if not text:
        return 0.0

    score = 0.0
    text_lower = text.lower()

    # Count leetspeak indicators
    leet_chars = set('0123456789@#$%&|/<>{}[]')
    leet_count = sum(1 for c in text if c in leet_chars)
    score += leet_count / len(text) * 0.4

    # Multi-char patterns (harder to decode)
    hard_patterns = ['|\\|', '/\\/', '|-|', '|<', '|3', '|)', '()']
    for pattern in hard_patterns:
        if pattern in text:
            score += 0.05

    # Number ambiguity (2 = to/too/two)
    ambig_nums = ['2', '4', '8']
    for num in ambig_nums:
        if num in text:
            score += 0.03

    return min(score, 1.0)


def prepare_dataset(
    data: List[Dict],
    tokenizer,
    config: TrainingConfig,
    add_preservation: bool = True,
    add_edge_cases: bool = True,
    sort_by_difficulty: bool = False,
) -> DatasetDict:
    """
    Prepare HuggingFace datasets with train/val split.

    Args:
        data: List of {'input': str, 'target': str} dicts
        tokenizer: HuggingFace tokenizer
        config: Training configuration
        add_preservation: Add identity examples to prevent over-correction
        add_edge_cases: Add known difficult examples
        sort_by_difficulty: Sort by difficulty for curriculum learning

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    random.shuffle(data)

    # Split data
    split_idx = int(len(data) * config.train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Add preservation examples to training (not validation!)
    if add_preservation:
        preservation_data = []
        repeats = max(1, len(train_data) // 1000)  # Scale with dataset size

        for _ in range(repeats):
            for inp, tgt in PRESERVATION_EXAMPLES:
                preservation_data.append({'input': inp, 'target': tgt})

        train_data.extend(preservation_data)
        print(f"[DATA] Added {len(preservation_data)} preservation examples")

    # Add edge cases to training
    if add_edge_cases:
        edge_case_data = []
        repeats = max(1, len(train_data) // 500)  # Even more emphasis

        for _ in range(repeats):
            for inp, tgt in EDGE_CASE_EXAMPLES:
                edge_case_data.append({'input': inp, 'target': tgt})

        train_data.extend(edge_case_data)
        print(f"[DATA] Added {len(edge_case_data)} edge case examples")

    # Sort by difficulty for curriculum learning
    if sort_by_difficulty:
        train_data = sorted(train_data, key=lambda x: compute_difficulty_score(x['input']))
        print("[DATA] Sorted by difficulty (easy -> hard)")
    else:
        random.shuffle(train_data)

    print(f"[DATA] Train: {len(train_data):,}, Validation: {len(val_data):,}")

    # Tokenization function
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples['input'],
            max_length=config.max_input_length,
            truncation=True,
            padding=False,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'],
                max_length=config.max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Tokenize with multiprocessing
    num_proc = min(config.dataloader_num_workers, os.cpu_count() or 4)

    print("[DATA] Tokenizing train set...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'target'],
        num_proc=num_proc,
        desc="Tokenizing train",
    )

    print("[DATA] Tokenizing validation set...")
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'target'],
        num_proc=num_proc,
        desc="Tokenizing val",
    )

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })


# ============================================================================
# ADVERSARIAL FILTERING
# ============================================================================

def adversarial_filter(
    data: List[Dict],
    v2_model_path: str,
    threshold: float = 0.8,
    batch_size: int = 64,
) -> List[Dict]:
    """
    Filter training data using the v2 model.
    Keep only examples that v2 struggles with (hard negatives).

    Args:
        data: List of training examples
        v2_model_path: Path to v2 model for filtering
        threshold: Keep examples where v2 accuracy < threshold
        batch_size: Batch size for inference

    Returns:
        Filtered list of hard examples
    """
    print(f"\n[ADVERSARIAL] Loading v2 model for filtering...")
    print(f"[ADVERSARIAL] Will keep examples where v2 similarity < {threshold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(v2_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(v2_model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ADVERSARIAL] Failed to load v2 model: {e}")
        print("[ADVERSARIAL] Skipping adversarial filtering")
        return data

    hard_examples = []
    easy_examples = []

    print(f"[ADVERSARIAL] Processing {len(data):,} examples...")

    for i in tqdm(range(0, len(data), batch_size), desc="Adversarial filtering"):
        batch = data[i:i + batch_size]
        inputs = [item['input'] for item in batch]
        targets = [item['target'] for item in batch]

        # Generate v2 predictions
        tokenized = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **tokenized,
                max_length=256,
                num_beams=2,
                early_stopping=True,
            )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Compute similarity and filter
        for item, pred, target in zip(batch, predictions, targets):
            # Simple character-level similarity
            pred_lower = pred.strip().lower()
            target_lower = target.strip().lower()

            if pred_lower == target_lower:
                similarity = 1.0
            else:
                # Levenshtein-ish similarity
                common = sum(1 for a, b in zip(pred_lower, target_lower) if a == b)
                max_len = max(len(pred_lower), len(target_lower))
                similarity = common / max_len if max_len > 0 else 0

            if similarity < threshold:
                hard_examples.append(item)
            else:
                easy_examples.append(item)

    # Clean up
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Keep some easy examples for balance (10%)
    easy_sample = random.sample(easy_examples, min(len(easy_examples), len(hard_examples) // 10))

    final_data = hard_examples + easy_sample
    random.shuffle(final_data)

    print(f"\n[ADVERSARIAL] Results:")
    print(f"  Hard examples (v2 failed):  {len(hard_examples):,}")
    print(f"  Easy examples (v2 passed):  {len(easy_examples):,}")
    print(f"  Final dataset size:         {len(final_data):,}")

    return final_data


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class ComputeMetrics:
    """Compute evaluation metrics for the model."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")
        self.cer = evaluate.load("cer")

    def __call__(self, eval_pred):
        predictions, labels = eval_pred

        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 in labels (padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute BLEU
        bleu_result = self.bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )

        # Compute CER
        cer_result = self.cer.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        return {
            "bleu": bleu_result["score"],
            "cer": cer_result,
        }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train(
    data_path: str,
    config: TrainingConfig,
    use_adversarial: bool = False,
    resume_from: Optional[str] = None,
    use_curriculum: bool = True,
):
    """
    Main training function.

    Args:
        data_path: Path to training data (JSONL)
        config: Training configuration
        use_adversarial: Whether to filter with v2 model
        resume_from: Checkpoint path to resume from
        use_curriculum: Whether to use curriculum learning
    """
    print("=" * 70)
    print("ByT5 LEETSPEAK DECODER V3 - TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Base Model:     {config.base_model}")
    print(f"  Output Dir:     {config.output_dir}")
    print(f"  Batch Size:     {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} (per device)")
    print(f"  Learning Rate:  {config.learning_rate}")
    print(f"  Epochs:         {config.num_train_epochs}")
    print(f"  Mixed Precision: {'BF16' if config.bf16 else 'FP16' if config.fp16 else 'FP32'}")
    print(f"  Load to RAM:    {config.load_model_to_ram}")
    print()

    # Set seed
    set_seed(config.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"[DEVICE] Using: {device}")
    if torch.cuda.is_available():
        print(f"[DEVICE] GPUs available: {n_gpus}")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"[DEVICE]   GPU {i}: {name} ({vram:.1f} GB)")

        # Enable TF32 for faster matmul on Ampere+ / Blackwell
        if config.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("[DEVICE] TF32 enabled")

    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data = load_training_data(data_path, config)

    if not data:
        print("[ERROR] No training data found!")
        return

    # Adversarial filtering
    if use_adversarial and config.use_v2_for_adversarial:
        # Use RAM-cached model for filtering too
        v2_path = download_v2_model_to_ram() if config.load_model_to_ram else config.use_v2_for_adversarial
        data = adversarial_filter(
            data,
            v2_path,
            threshold=config.adversarial_threshold,
        )

    # Load tokenizer and model
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    model_path = get_model_path(config)
    print(f"[MODEL] Loading from: {model_path}")

    is_continuing_from_v2 = (config.base_model == V2_MODEL_HUB and config.continue_from is None)
    if is_continuing_from_v2:
        print(f"[MODEL] Continuing training from v2 model (fine-tuning mode)")
    else:
        print(f"[MODEL] Training from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[MODEL] Gradient checkpointing enabled")

    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable: {trainable_params:,}")

    # Prepare dataset
    print("\n" + "=" * 70)
    print("PREPARING DATASET")
    print("=" * 70)

    dataset = prepare_dataset(
        data,
        tokenizer,
        config,
        add_preservation=True,
        add_edge_cases=True,
        sort_by_difficulty=use_curriculum,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # Optimize for tensor cores
    )

    # Check for checkpoint to resume from
    last_checkpoint = None
    if resume_from:
        last_checkpoint = resume_from
    elif os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)

    if last_checkpoint:
        print(f"[CHECKPOINT] Resuming from: {last_checkpoint}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,

        # Batch settings
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,

        # Learning rate
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,

        # Duration
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,

        # Mixed precision
        bf16=config.bf16,
        fp16=config.fp16,

        # Evaluation & Saving
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,

        # Logging
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        logging_first_step=config.logging_first_step,
        report_to=config.report_to,

        # Performance
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,

        # Generation settings for evaluation
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=4,

        # Misc
        seed=config.seed,
        remove_unused_columns=True,
        label_names=["labels"],
    )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),
        SanityCheckCallback(tokenizer, test_interval=config.eval_steps),
        MetricsCallback(),
    ]

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer),
    )

    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"\n  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Val samples: {len(dataset['validation']):,}")
    print(f"  Total steps: ~{len(dataset['train']) // (config.per_device_train_batch_size * config.gradient_accumulation_steps) * config.num_train_epochs:,}")
    print()

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model saved to: {config.output_dir}")
    print(f"\n  Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Final eval loss:  {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Final BLEU:       {eval_metrics.get('eval_bleu', 'N/A'):.2f}")
    print()

    return trainer


# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def quick_test(model_path: str):
    """
    Quick test of a trained model.
    """
    print(f"\n[TEST] Loading model from: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    test_inputs = [
        "idk wh4t 2 d0 tbh",
        "1t5 2 l8",
        "c u l8r m8",
        "thx 4 th3 h31p",
        "ur prolly r1ght ngl",
        "I have 2 cats",
        "1 h4v3 2 c4t5",
        "th4t5 2 much 4 m3",
    ]

    print("\n" + "="*60)
    print("QUICK TEST RESULTS")
    print("="*60)

    for text in test_inputs:
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256, num_beams=4)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  IN:  {text}")
        print(f"  OUT: {decoded}")

    print("\n" + "="*60)


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ByT5 Leetspeak Decoder V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training (continues from v2 model - RECOMMENDED)
    python train_model_v3.py --data training_data_v3.jsonl

    # Train from scratch (google/byt5-base)
    python train_model_v3.py --data training_data_v3.jsonl --from-scratch

    # With adversarial filtering (keeps hard examples)
    python train_model_v3.py --data training_data_v3.jsonl --adversarial

    # Resume from checkpoint
    python train_model_v3.py --data training_data_v3.jsonl --resume checkpoint-5000

    # Custom output directory
    python train_model_v3.py --data training_data_v3.jsonl --output ./my_model

    # Multi-GPU with accelerate
    accelerate launch train_model_v3.py --data training_data_v3.jsonl

    # Quick test after training
    python train_model_v3.py --test ./byt5_leetspeak_v3
        """
    )

    # Required
    parser.add_argument("--data", "-d", type=str,
                        help="Path to training data (JSONL file or directory)")

    # Model options
    parser.add_argument("--from-scratch", action="store_true",
                        help="Train from google/byt5-base instead of continuing from v2")
    parser.add_argument("--continue-from", type=str, default=None,
                        help="Continue training from this checkpoint path")
    parser.add_argument("--output", "-o", type=str, default="./byt5_leetspeak_v3",
                        help="Output directory for model")
    parser.add_argument("--no-ram-cache", action="store_true",
                        help="Don't cache model in RAM (use if you have disk space)")

    # Training options
    parser.add_argument("--epochs", "-e", type=int, default=3,
                        help="Number of training epochs (default: 3 for fine-tuning)")
    parser.add_argument("--batch-size", "-b", type=int, default=24,
                        help="Per-device batch size (default: 24 for RTX 5090)")
    parser.add_argument("--learning-rate", "-lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5 for fine-tuning)")
    parser.add_argument("--accumulation", "-a", type=int, default=3,
                        help="Gradient accumulation steps (default: 3)")

    # Advanced options
    parser.add_argument("--adversarial", action="store_true",
                        help="Use v2 model to filter easy examples (keeps hard negatives)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning (sort by difficulty)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Evaluation/Testing
    parser.add_argument("--test", type=str, default=None,
                        help="Quick test a trained model (provide path)")

    # Hardware options
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision (default: True)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 instead of BF16")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Quick test mode
    if args.test:
        quick_test(args.test)
        return

    # Validate data path
    if not args.data:
        print("[ERROR] --data is required for training")
        print("Usage: python train_model_v3.py --data training_data_v3.jsonl")
        return

    # Determine base model
    if args.from_scratch:
        base_model = "google/byt5-base"
        learning_rate = 2e-4  # Higher LR for fresh training
        epochs = 5  # More epochs for fresh training
        print("[CONFIG] Training from scratch (google/byt5-base)")
    else:
        base_model = V2_MODEL_HUB
        learning_rate = args.learning_rate
        epochs = args.epochs
        print(f"[CONFIG] Fine-tuning from v2 ({V2_MODEL_HUB})")

    # Build config
    config = TrainingConfig(
        base_model=base_model,
        continue_from=args.continue_from,
        output_dir=args.output,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=learning_rate,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        load_model_to_ram=not args.no_ram_cache,
    )

    # Train
    try:
        train(
            data_path=args.data,
            config=config,
            use_adversarial=args.adversarial,
            resume_from=args.resume,
            use_curriculum=not args.no_curriculum,
        )
    finally:
        # Clean up RAM cache
        if config.load_model_to_ram:
            cleanup_ram_cache()


if __name__ == "__main__":
    main()
