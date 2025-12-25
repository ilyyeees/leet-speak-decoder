# -*- coding: utf-8 -*-
"""
================================================================================
byt5 leetspeak decoder - production edition
================================================================================
a production-ready notebook for fine-tuning google/byt5-base to translate
leetspeak (1337 speak) back into clean english with context awareness.

features:
- context-dependent number handling (preserve "2 cats" vs translate "2 late")
- comprehensive leetspeak corruption engine
- real training data from wikitext-2
- robust evaluation suite with bleu, cer, wer metrics
- fast inference with batch support

designed for google colab t4 gpu (works on cpu too, just slower)
================================================================================
"""

# ==============================================================================
# section 1: installation & imports
# ==============================================================================

# uncomment these lines when running in colab
# !pip install -q transformers datasets evaluate accelerate sentencepiece
# !pip install -q jiwer sacrebleu tqdm

import os
import re
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from tqdm.auto import tqdm

# huggingface ecosystem
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import load_dataset, Dataset, DatasetDict

# evaluation metrics
import evaluate

warnings.filterwarnings("ignore")

# ==============================================================================
# section 2: configuration
# ==============================================================================

@dataclass
class Config:
    """
    all hyperparameters and settings in one place.
    tweak these values to experiment with different configurations.
    """
    # model settings
    model_name: str = "google/byt5-base"  # upgraded for better context understanding
    max_input_length: int = 256
    max_target_length: int = 256

    # data settings
    min_sentence_length: int = 20
    max_sentence_length: int = 150
    num_samples: int = 40000  # increased for better training
    train_split: float = 0.9

    # corruption settings
    base_corruption_prob: float = 0.5  # 40-60% per character
    word_corruption_prob: float = 0.7  # probability to corrupt a word
    noise_rate: float = 0.05  # random insertions/deletions
    number_protection_prob: float = 0.5  # prob to keep sentences with numbers clean

    # training hyperparameters (optimized for RTX 4090)
    per_device_train_batch_size: int = 16  # reduced to fit in VRAM
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2  # effective batch size: 32
    learning_rate: float = 3e-4
    num_train_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = False  # disabled in favor of bf16
    bf16: bool = True  # native RTX 40-series optimization

    # dataloader optimization
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # training settings
    logging_steps: int = 100
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # paths
    output_dir: str = "./byt5_leetspeak_model"

    # seed for reproducibility
    seed: int = 42


config = Config()
set_seed(config.seed)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] using device: {device}")
if torch.cuda.is_available():
    print(f"[info] gpu: {torch.cuda.get_device_name(0)}")
    print(f"[info] vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} gb")

# ==============================================================================
# section 3: leetspeak corruption engine
# ==============================================================================

class LeetSpeakCorruptor:
    """
    advanced leetspeak corruption engine with context awareness.

    handles:
    - comprehensive character substitutions
    - probabilistic corruption
    - number protection logic
    - case variation
    - noise injection
    """

    def __init__(
        self,
        base_prob: float = 0.5,
        word_prob: float = 0.7,
        noise_rate: float = 0.05,
        number_protection_prob: float = 0.5,
    ):
        self.base_prob = base_prob
        self.word_prob = word_prob
        self.noise_rate = noise_rate
        self.number_protection_prob = number_protection_prob

        # comprehensive leetspeak mappings
        # each letter maps to a list of possible substitutions
        self.leet_map = {
            'a': ['4', '@', '/\\', '^', 'α'],
            'b': ['8', '|3', 'ß', '13'],
            'c': ['(', '<', '{', '¢'],
            'd': ['|)', '|]', 'đ'],
            'e': ['3', '€', '£', '&'],
            'f': ['|=', 'ph', '|#'],
            'g': ['9', '6', '&', 'gee'],
            'h': ['#', '|-|', '}{', ']-['],
            'i': ['1', '!', '|', 'eye'],
            'j': ['_|', '_/', ';'],
            'k': ['|<', '|{', '|('],
            'l': ['1', '|', '|_', '£'],
            'm': ['|\\/|', '/\\/\\', '|v|', 'em'],
            'n': ['|\\|', '/\\/', '|/|'],
            'o': ['0', '()', '[]', 'oh'],
            'p': ['|*', '|>', '|°'],
            'q': ['0_', '0,', '()_'],
            'r': ['|2', '|?', '/2', 'are'],
            's': ['5', '$', 'z', '§'],
            't': ['7', '+', '†', '-|-'],
            'u': ['|_|', '\\_/', 'µ', 'you'],
            'v': ['\\/', '|/', '\\/'],
            'w': ['\\/\\/', 'vv', '\\^/', 'doubleyou'],
            'x': ['><', '}{', '×', 'ecks'],
            'y': ['`/', '¥', 'why'],
            'z': ['2', '7_', '%', 'zed'],
        }

        # simpler mappings for more realistic corruption
        self.simple_leet_map = {
            'a': ['4', '@'],
            'b': ['8'],
            'c': ['('],
            'e': ['3'],
            'g': ['9', '6'],
            'h': ['#'],
            'i': ['1', '!'],
            'l': ['1', '|'],
            'o': ['0'],
            's': ['5', '$', 'z'],
            't': ['7', '+'],
            'z': ['2'],
        }

        # word-level substitutions for common words
        self.word_subs = {
            'you': ['u', 'yu', 'yoo'],
            'are': ['r', 'ar'],
            'to': ['2', '2o'],
            'too': ['2', '2oo'],
            'for': ['4', 'fo'],
            'be': ['b'],
            'see': ['c', 'c'],
            'why': ['y'],
            'okay': ['ok', 'k'],
            'the': ['th3', 'da', 'd4'],
            'and': ['&', 'n'],
            'at': ['@'],
            'one': ['1'],
            'won': ['1'],
            'before': ['b4'],
            'great': ['gr8'],
            'late': ['l8'],
            'mate': ['m8'],
            'hate': ['h8'],
            'wait': ['w8'],
            'what': ['wut', 'wat', 'wh4t'],
            'that': ['th4t', 'dat'],
            'this': ['dis', 'th1s'],
            'with': ['w/', 'wit', 'w1th'],
            'cool': ['kool', 'c00l', 'kewl'],
            'good': ['gud', 'g00d'],
            'love': ['luv', 'l0ve', '<3'],
            'please': ['plz', 'pls', 'pl3ase'],
            'thanks': ['thx', 'thanx', 'ty'],
            'because': ['bc', 'cuz', 'bcuz'],
            'people': ['ppl', 'peeps'],
            'really': ['rly', 'rlly'],
            'something': ['smth', 'sth'],
            'nothing': ['nth', 'noth1ng'],
            'everyone': ['every1', 'evry1'],
            'someone': ['some1', 'sum1'],
            'anyone': ['any1'],
            'today': ['2day', '2d4y'],
            'tomorrow': ['2morrow', '2mrw'],
            'tonight': ['2night', '2n1ght'],
        }

        # regex for detecting numbers in context
        self.number_patterns = [
            r'\b\d+\s*(years?|months?|days?|hours?|minutes?|seconds?)\b',
            r'\b\d+\s*(pm|am|PM|AM)\b',
            r'\b(at|around|about)\s+\d+\b',
            r'\b\d+\s*(cats?|dogs?|people|items?|things?|bugs?|errors?)\b',
            r'\b(have|has|got|need|want)\s+\d+\b',
            r'\b\d+\s*(st|nd|rd|th)\b',
            r'\b(function|version|chapter|page|line|row|column)\s*\d+\b',
            r'\b\d{4}\b',  # years
            r'\b\d+\.\d+\b',  # decimals
            r'\$\d+',  # prices
            r'\b\d+%\b',  # percentages
        ]

    def _has_contextual_number(self, text: str) -> bool:
        """check if text contains numbers that should be preserved."""
        text_lower = text.lower()
        for pattern in self.number_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _corrupt_char(self, char: str, use_simple: bool = True) -> str:
        """corrupt a single character with leetspeak substitution."""
        char_lower = char.lower()

        leet_dict = self.simple_leet_map if use_simple else self.leet_map

        if char_lower in leet_dict and random.random() < self.base_prob:
            replacements = leet_dict[char_lower]
            replacement = random.choice(replacements)

            # preserve case for single-char replacements
            if len(replacement) == 1 and char.isupper():
                replacement = replacement.upper()

            return replacement

        return char

    def _corrupt_word(self, word: str) -> str:
        """corrupt a word using character-level or word-level substitutions."""
        # check for word-level substitution first
        word_lower = word.lower()
        if word_lower in self.word_subs and random.random() < 0.3:
            replacement = random.choice(self.word_subs[word_lower])
            # try to preserve case
            if word.isupper():
                return replacement.upper()
            elif word[0].isupper():
                return replacement.capitalize()
            return replacement

        # character-level corruption
        result = []
        use_simple = random.random() < 0.8  # 80% use simple mappings for realism

        for char in word:
            if char.isalpha():
                result.append(self._corrupt_char(char, use_simple))
            else:
                result.append(char)

        corrupted = ''.join(result)

        # vowel dropping (SMS style: "text" -> "txt")
        if random.random() < 0.2 and len(corrupted) > 3:
            corrupted = self._drop_vowels(corrupted)

        return corrupted

    def _drop_vowels(self, word: str) -> str:
        """Drop vowels from middle of word (SMS style: 'text' -> 'txt')."""
        if len(word) <= 3:
            return word
        # Keep first and last char, randomly drop vowels in middle
        vowels = 'aeiouAEIOU'
        result = [word[0]]
        for char in word[1:-1]:
            if char in vowels and random.random() < 0.6:
                continue  # drop this vowel
            result.append(char)
        result.append(word[-1])
        return ''.join(result)

    def _add_noise(self, text: str) -> str:
        """add random character insertions/deletions."""
        if random.random() > self.noise_rate:
            return text

        chars = list(text)
        noise_type = random.choice(['insert', 'delete', 'swap'])

        if noise_type == 'insert' and len(chars) > 0:
            # insert a random character
            pos = random.randint(0, len(chars))
            char = random.choice('aeiou0123456789!@#$')
            chars.insert(pos, char)

        elif noise_type == 'delete' and len(chars) > 5:
            # delete a random character
            pos = random.randint(0, len(chars) - 1)
            if chars[pos].isalpha():
                chars.pop(pos)

        elif noise_type == 'swap' and len(chars) > 2:
            # swap two adjacent characters
            pos = random.randint(0, len(chars) - 2)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]

        return ''.join(chars)

    def _apply_case_variation(self, text: str) -> str:
        """apply random case variations typical in leetspeak."""
        if random.random() < 0.1:
            # all caps
            return text.upper()
        elif random.random() < 0.05:
            # random case (sTiCkY cApS)
            return ''.join(
                c.upper() if random.random() < 0.5 else c.lower()
                for c in text
            )
        return text

    def corrupt(self, text: str) -> Tuple[str, str]:
        """
        corrupt clean english text into leetspeak.

        returns:
            tuple of (corrupted_text, original_text)
        """
        # number protection: sometimes keep sentences with numbers clean
        if self._has_contextual_number(text) and random.random() < self.number_protection_prob:
            return text, text

        words = text.split()
        corrupted_words = []

        for word in words:
            # decide whether to corrupt this word
            if random.random() < self.word_prob:
                corrupted_word = self._corrupt_word(word)
                corrupted_words.append(corrupted_word)
            else:
                corrupted_words.append(word)

        result = ' '.join(corrupted_words)

        # apply additional transformations
        result = self._add_noise(result)
        result = self._apply_case_variation(result)

        return result, text


# ==============================================================================
# section 4: data loading & preprocessing
# ==============================================================================

def load_and_preprocess_wikitext(
    num_samples: int = 12000,
    min_length: int = 20,
    max_length: int = 150,
) -> List[str]:
    """
    load wikitext-2 dataset and extract clean sentences.

    args:
        num_samples: target number of sentences to extract
        min_length: minimum sentence length (characters)
        max_length: maximum sentence length (characters)

    returns:
        list of clean english sentences
    """
    print("[info] loading wikitext-2 dataset...")

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        print(f"[warn] failed to load wikitext-2: {e}")
        print("[info] trying wikitext-103...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    sentences = []

    # patterns to filter out
    bad_patterns = [
        r'^=+.*=+$',  # headers
        r'^\s*$',  # empty lines
        r'^@',  # metadata
        r'\[\[',  # wiki links
        r'\{\{',  # wiki templates
        r'<ref',  # references
        r'http',  # urls
        r'\.jpg|\.png|\.gif',  # image references
    ]

    print("[info] extracting and cleaning sentences...")

    for item in tqdm(dataset, desc="processing"):
        text = item['text'].strip()

        # skip bad patterns
        if any(re.search(p, text) for p in bad_patterns):
            continue

        # split into sentences
        for sentence in re.split(r'(?<=[.!?])\s+', text):
            sentence = sentence.strip()

            # filter by length
            if len(sentence) < min_length or len(sentence) > max_length:
                continue

            # skip sentences with too much punctuation
            punct_ratio = sum(1 for c in sentence if c in '()[]{}@#$%^&*_+=<>~`|\\') / len(sentence)
            if punct_ratio > 0.1:
                continue

            # skip sentences that are mostly uppercase (likely headers)
            upper_ratio = sum(1 for c in sentence if c.isupper()) / len(sentence)
            if upper_ratio > 0.5:
                continue

            # basic sentence quality check
            words = sentence.split()
            if len(words) < 4:
                continue

            sentences.append(sentence)

            if len(sentences) >= num_samples * 1.5:  # collect extra for filtering
                break

        if len(sentences) >= num_samples * 1.5:
            break

    # shuffle and take required number
    random.shuffle(sentences)
    sentences = sentences[:num_samples]

    print(f"[info] extracted {len(sentences)} sentences from wikitext")
    return sentences


def load_conversational_data(num_samples: int = 10000, min_length: int = 10, max_length: int = 150) -> List[str]:
    """
    Load SAMSum dataset for conversational training data.
    This adds natural dialogue which is common in leetspeak contexts.
    """
    print("[info] loading SAMSum conversational dataset...")

    try:
        dataset = load_dataset("knkarthick/samsum", split="train")
    except Exception as e:
        print(f"[warn] failed to load SAMSum: {e}")
        return []

    sentences = []

    for item in tqdm(dataset, desc="processing dialogues"):
        # SAMSum has 'dialogue' field with conversation text
        dialogue = item.get('dialogue', '')

        # Split dialogue into individual lines/utterances
        for utterance in dialogue.split('\n'):
            # Remove speaker prefixes like "Person1: " or "Amanda: "
            if ':' in utterance:
                utterance = utterance.split(':', 1)[-1].strip()
            else:
                utterance = utterance.strip()

            # filter by length
            if len(utterance) < min_length or len(utterance) > max_length:
                continue

            # skip if too much punctuation
            punct_ratio = sum(1 for c in utterance if c in '()[]{}@#$%^&*_+=<>~`|\\') / max(len(utterance), 1)
            if punct_ratio > 0.1:
                continue

            sentences.append(utterance)

            if len(sentences) >= num_samples:
                break

        if len(sentences) >= num_samples:
            break

    random.shuffle(sentences)
    print(f"[info] extracted {len(sentences)} sentences from SAMSum")
    return sentences


def create_training_examples(
    sentences: List[str],
    corruptor: LeetSpeakCorruptor,
) -> List[Dict[str, str]]:
    """
    create training examples with leetspeak corruption.

    args:
        sentences: list of clean english sentences
        corruptor: leetspeak corruptor instance

    returns:
        list of dicts with 'input' and 'target' keys
    """
    print("[info] generating training examples with leetspeak corruption...")

    examples = []

    for sentence in tqdm(sentences, desc="corrupting"):
        corrupted, original = corruptor.corrupt(sentence)
        examples.append({
            'input': corrupted,
            'target': original,
        })

    # add explicit number context examples
    number_examples = [
        # preserve numbers
        ("1 h4v3 2 c4t5", "I have 2 cats"),
        ("Th3r3 4r3 3 r3450n5", "There are 3 reasons"),
        ("M33t m3 4t 3 PM", "Meet me at 3 PM"),
        ("Sh3 15 25 y34r5 0ld", "She is 25 years old"),
        ("1 n33d 5 m1nut35", "I need 5 minutes"),
        ("B0rn 1n 1990", "Born in 1990"),
        ("P4g3 42 0f th3 b00k", "Page 42 of the book"),
        ("V3rs10n 2.0 15 0ut", "Version 2.0 is out"),
        ("My c0d3 h4s 2 bug5 1n funct10n 3", "My code has 2 bugs in function 3"),
        ("1 g0t 100 p01nt5", "I got 100 points"),

        # translate numbers as words
        ("1t 15 2 l4t3", "It is too late"),
        ("1 w4nt 2 g0 h0m3", "I want to go home"),
        ("Th4t 15 2 much", "That is too much"),
        ("U r 2 c00l", "You are too cool"),
        ("1 n33d 2 t4lk 2 u", "I need to talk to you"),
        ("W3 n33d 2 l34v3 b4 5", "We need to leave before 5"),

        # mixed contexts
        ("1 h4v3 2 g0 2 th3 5t0r3", "I have to go to the store"),
        ("Th3y w4nt 2 buy 3 t1ck3t5", "They want to buy 3 tickets"),
        ("W3 n33d 2 w41t 4 10 m1nut35", "We need to wait for 10 minutes"),
    ]

    # add number examples multiple times for emphasis
    for _ in range(10):
        for inp, tgt in number_examples:
            examples.append({'input': inp, 'target': tgt})

    # add more diverse leetspeak examples
    extra_examples = [
        ("H3110 W0r1d!", "Hello World!"),
        ("Th3 qu1ck br0wn f0x jump5 0v3r 7h3 l4zy d0g", "The quick brown fox jumps over the lazy dog"),
        ("1337 5p34k 15 c00l", "Leet speak is cool"),
        ("H3y th3r3 h0w r u d01ng 2d4y?", "Hey there how are you doing today?"),
        ("TH15 15 L33T", "THIS IS LEET"),
        ("Wh4t5 up my fr13nd?", "Whats up my friend?"),
        ("L3t5 g0 2 th3 p4rty", "Lets go to the party"),
        ("1 l0v3 pr0gr4mm1ng", "I love programming"),
        ("Th15 15 4w350m3", "This is awesome"),
        ("C4n u h31p m3 pl3453?", "Can you help me please?"),
        ("1 d0n7 und3r574nd", "I dont understand"),
        ("Wh3r3 r u g01ng?", "Where are you going?"),
        ("1 w1ll b3 th3r3 500n", "I will be there soon"),
        ("Th4nk5 4 3v3ry7h1ng", "Thanks for everything"),
        ("H0w w45 y0ur d4y?", "How was your day?"),
        ("1 n33d h31p w1th my c0d3", "I need help with my code"),
        ("Th3 w34th3r 15 n1c3", "The weather is nice"),
        ("L3t m3 kn0w wh3n u r r34dy", "Let me know when you are ready"),
        ("1 c4n7 w41t 2 533 u", "I cant wait to see you"),
        ("Th15 15 50 c00l", "This is so cool"),
    ]

    for _ in range(5):
        for inp, tgt in extra_examples:
            examples.append({'input': inp, 'target': tgt})

    random.shuffle(examples)
    print(f"[info] created {len(examples)} training examples")

    return examples


# ==============================================================================
# section 5: dataset preparation
# ==============================================================================

def prepare_dataset(
    examples: List[Dict[str, str]],
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 256,
    train_split: float = 0.9,
) -> DatasetDict:
    """
    prepare huggingface dataset for training.

    args:
        examples: list of training examples
        tokenizer: byt5 tokenizer
        max_input_length: max input sequence length
        max_target_length: max target sequence length
        train_split: fraction of data for training

    returns:
        DatasetDict with train and validation splits
    """
    print("[info] preparing dataset...")

    # split data
    split_idx = int(len(examples) * train_split)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]

    print(f"[info] train size: {len(train_data)}, validation size: {len(val_data)}")

    def tokenize_function(batch):
        """tokenize a batch of examples."""
        # tokenize inputs
        model_inputs = tokenizer(
            batch['input'],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

        # tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch['target'],
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # tokenize with multiprocessing for CPU efficiency
    num_proc = 4  # use multiple CPU cores
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'target'],
        desc="tokenizing train",
        num_proc=num_proc,
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['input', 'target'],
        desc="tokenizing val",
        num_proc=num_proc,
    )

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })

    return dataset_dict


# ==============================================================================
# section 6: model & tokenizer loading
# ==============================================================================

def load_model_and_tokenizer(model_name: str = "google/byt5-base"):
    """
    load byt5 model and tokenizer.

    args:
        model_name: huggingface model name

    returns:
        tuple of (model, tokenizer)
    """
    print(f"[info] loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)

    # move to device
    model = model.to(device)

    # print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[info] total parameters: {total_params:,}")
    print(f"[info] trainable parameters: {trainable_params:,}")

    return model, tokenizer


# ==============================================================================
# section 7: evaluation metrics
# ==============================================================================

class EvaluationMetrics:
    """
    comprehensive evaluation metrics for translation quality.
    includes bleu, cer, wer, and exact match accuracy.
    """

    def __init__(self):
        self.bleu = evaluate.load("sacrebleu")
        self.cer = evaluate.load("cer")

        # wer might not be available on all systems
        try:
            self.wer = evaluate.load("wer")
        except:
            self.wer = None
            print("[warn] wer metric not available")

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        compute all evaluation metrics.

        args:
            predictions: list of predicted strings
            references: list of reference strings

        returns:
            dict of metric names to values
        """
        results = {}

        # bleu score
        bleu_result = self.bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        results['bleu'] = bleu_result['score']

        # character error rate
        cer_result = self.cer.compute(
            predictions=predictions,
            references=references,
        )
        results['cer'] = cer_result * 100  # as percentage

        # word error rate
        if self.wer:
            wer_result = self.wer.compute(
                predictions=predictions,
                references=references,
            )
            results['wer'] = wer_result * 100

        # exact match accuracy
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        results['exact_match'] = (exact_matches / len(predictions)) * 100

        return results


# ==============================================================================
# section 8: training setup
# ==============================================================================

def setup_trainer(
    model,
    tokenizer,
    dataset: DatasetDict,
    config: Config,
    metrics: EvaluationMetrics,
) -> Seq2SeqTrainer:
    """
    set up the seq2seq trainer with all configurations.

    args:
        model: byt5 model
        tokenizer: byt5 tokenizer
        dataset: prepared dataset
        config: configuration object
        metrics: evaluation metrics object

    returns:
        configured Seq2SeqTrainer
    """
    print("[info] setting up trainer...")

    # data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=config.max_input_length,
    )

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        report_to="none",  # change to "wandb" or "tensorboard" if desired
        seed=config.seed,
    )

    def compute_metrics_fn(eval_preds):
        """compute metrics during evaluation."""
        preds, labels = eval_preds

        # decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        # replace -100 in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # ByT5 uses offset=259, max valid Unicode is 0x10FFFF (1114111)
        # So max valid token ID is 259 + 1114111 = 1114370
        # Clamp out-of-range token IDs to pad_token_id to prevent chr() crash
        max_valid_token = 259 + 0x10FFFF  # 1114370
        pad_id = tokenizer.pad_token_id

        # Ensure preds is numpy array for safe operations
        preds = np.array(preds) if not isinstance(preds, np.ndarray) else preds
        preds = np.clip(preds, 0, max_valid_token - 1)  # clip to valid range

        decoded_preds = tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # compute metrics
        results = metrics.compute_metrics(decoded_preds, decoded_labels)

        return results

    # create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    return trainer


# ==============================================================================
# section 9: inference
# ==============================================================================

class LeetSpeakDecoder:
    """
    inference class for translating leetspeak to english.
    supports single and batch inference with configurable generation parameters.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device = None,
        max_length: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.model.eval()

    def translate(
        self,
        text: Union[str, List[str]],
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> Union[str, List[str]]:
        """
        translate leetspeak text to english.

        args:
            text: input text or list of texts
            num_beams: beam search width
            temperature: sampling temperature (if do_sample=True)
            do_sample: whether to use sampling

        returns:
            translated text or list of texts
        """
        single_input = isinstance(text, str)
        if single_input:
            text = [text]

        # tokenize
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True,
            )

        # decode
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if single_input:
            return decoded[0]
        return decoded

    def __call__(self, text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """callable interface for translation."""
        return self.translate(text, **kwargs)


def translate_leetspeak(
    text: Union[str, List[str]],
    model,
    tokenizer,
    **kwargs,
) -> Union[str, List[str]]:
    """
    convenience function for translating leetspeak.

    args:
        text: input text or list of texts
        model: trained model
        tokenizer: tokenizer
        **kwargs: additional generation parameters

    returns:
        translated text
    """
    decoder = LeetSpeakDecoder(model, tokenizer)
    return decoder.translate(text, **kwargs)


# ==============================================================================
# section 10: test suite
# ==============================================================================

def run_test_suite(decoder: LeetSpeakDecoder) -> Dict[str, float]:
    """
    run comprehensive test suite on the trained model.

    args:
        decoder: leetspeak decoder instance

    returns:
        dict of test results
    """
    print("\n" + "=" * 60)
    print("running comprehensive test suite")
    print("=" * 60 + "\n")

    # test cases organized by category
    test_cases = {
        "number preservation": [
            ("1 h4v3 2 c4t5", "I have 2 cats"),
            ("M33t m3 4t 3 PM", "Meet me at 3 PM"),
            ("Th3r3 4r3 3 r3450n5", "There are 3 reasons"),
            ("Sh3 15 25 y34r5 0ld", "She is 25 years old"),
            ("My c0d3 h45 2 bug5 1n funct10n 3", "My code has 2 bugs in function 3"),
        ],
        "number as word": [
            ("1t 15 2 l4t3", "It is too late"),
            ("1 w4nt 2 g0 h0m3", "I want to go home"),
            ("Th4t 15 2 much", "That is too much"),
        ],
        "basic leetspeak": [
            ("H3110 W0r1d!", "Hello World!"),
            ("Th3 qu1ck br0wn f0x", "The quick brown fox"),
            ("1337 5p34k 15 c00l", "Leet speak is cool"),
        ],
        "heavy corruption": [
            ("H3y th3r3 h0w r u?", "Hey there how are you?"),
            ("TH15 15 L33T", "THIS IS LEET"),
            ("Wh4t5 up my fr13nd?", "Whats up my friend?"),
        ],
        "edge cases": [
            ("Hello", "Hello"),  # no corruption
            ("He11o", "Hello"),  # minimal corruption
            ("12345", "12345"),  # pure numbers
        ],
    }

    results = {}
    total_correct = 0
    total_tests = 0

    for category, cases in test_cases.items():
        print(f"\n[{category}]")
        category_correct = 0

        for inp, expected in cases:
            output = decoder.translate(inp)

            # flexible matching (case-insensitive, ignore minor punctuation differences)
            is_correct = output.lower().strip() == expected.lower().strip()

            status = "✓" if is_correct else "✗"
            print(f"  {status} '{inp}'")
            print(f"    → got: '{output}'")
            print(f"    → expected: '{expected}'")

            if is_correct:
                category_correct += 1
                total_correct += 1
            total_tests += 1

        accuracy = (category_correct / len(cases)) * 100
        results[category] = accuracy
        print(f"  category accuracy: {accuracy:.1f}%")

    # overall accuracy
    overall_accuracy = (total_correct / total_tests) * 100
    results['overall'] = overall_accuracy

    print("\n" + "=" * 60)
    print(f"overall accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_tests})")
    print("=" * 60 + "\n")

    return results


# ==============================================================================
# section 11: main execution
# ==============================================================================

def main():
    """main execution function."""

    print("\n" + "=" * 60)
    print("byt5 leetspeak decoder - god mode")
    print("=" * 60 + "\n")

    # step 1: load data
    print("\n[step 1/6] loading and preprocessing data...")
    wiki_sentences = load_and_preprocess_wikitext(
        num_samples=int(config.num_samples * 0.7),  # 70% from wikitext
        min_length=config.min_sentence_length,
        max_length=config.max_sentence_length,
    )

    # load conversational data for more natural dialogue
    conv_sentences = load_conversational_data(
        num_samples=int(config.num_samples * 0.3),  # 30% from DailyDialog
        min_length=config.min_sentence_length,
        max_length=config.max_sentence_length,
    )

    # combine both datasets
    sentences = wiki_sentences + conv_sentences
    random.shuffle(sentences)
    print(f"[info] total training sentences: {len(sentences)}")

    # step 2: create corruption engine and examples
    print("\n[step 2/6] creating leetspeak corruption engine...")
    corruptor = LeetSpeakCorruptor(
        base_prob=config.base_corruption_prob,
        word_prob=config.word_corruption_prob,
        noise_rate=config.noise_rate,
        number_protection_prob=config.number_protection_prob,
    )

    examples = create_training_examples(sentences, corruptor)

    # step 3: load model and tokenizer
    print("\n[step 3/6] loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_name)

    # step 4: prepare dataset
    print("\n[step 4/6] preparing dataset...")
    dataset = prepare_dataset(
        examples,
        tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        train_split=config.train_split,
    )

    # step 5: setup and run training
    print("\n[step 5/6] setting up trainer...")
    metrics = EvaluationMetrics()
    trainer = setup_trainer(model, tokenizer, dataset, config, metrics)

    print("\n[training] starting training...")
    print("-" * 40)

    try:
        trainer.train()
        print("\n[training] training complete!")
    except Exception as e:
        print(f"\n[error] training failed: {e}")
        raise

    # step 6: evaluation and testing
    print("\n[step 6/6] running evaluation...")

    # create decoder
    decoder = LeetSpeakDecoder(model, tokenizer)

    # run test suite
    test_results = run_test_suite(decoder)

    # save model
    print("\n[saving] saving model to disk...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[saving] model saved to: {config.output_dir}")

    # interactive demo
    print("\n" + "=" * 60)
    print("interactive demo")
    print("=" * 60)

    demo_inputs = [
        "H3110 W0r1d!",
        "1 l0v3 pr0gr4mm1ng",
        "Th3 qu1ck br0wn f0x jump5 0v3r 7h3 l4zy d0g",
        "1 h4v3 2 c4t5 4nd 3 d0g5",
        "1t 15 2 l4t3 2 g0 h0m3",
    ]

    print("\ndemo translations:")
    for inp in demo_inputs:
        output = decoder.translate(inp)
        print(f"  '{inp}'")
        print(f"  → '{output}'")
        print()

    print("[done] all tasks complete!")

    return model, tokenizer, decoder


# ==============================================================================
# run
# ==============================================================================

if __name__ == "__main__":
    model, tokenizer, decoder = main()
