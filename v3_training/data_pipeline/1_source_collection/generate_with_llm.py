#!/usr/bin/env python3
"""
LLM Sentence Generation
========================
Uses a local LLM (via Ollama) to generate diverse, clean English sentences.
This supplements WikiText/ELI5 with more varied and modern language.

Output: llm_sentences.jsonl
    {"text": "The sunset painted the sky in shades of orange and pink.", "source": "llm_generated"}

Requirements:
    - Ollama running locally with a model pulled (e.g., qwen2.5:32b)
    - pip install requests
"""

import json
import re
import argparse
import random
import requests
from pathlib import Path
from typing import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Default Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Diverse topic categories to ensure variety
TOPIC_CATEGORIES = [
    # Daily life
    "daily activities and routines",
    "cooking and food preparation",
    "shopping and errands",
    "home and housework",
    
    # Social
    "meeting friends",
    "family gatherings",
    "parties and celebrations",
    "conversations with strangers",
    
    # Work/School
    "office work situations",
    "school and studying",
    "job interviews",
    "teamwork and collaboration",
    
    # Hobbies
    "playing video games",
    "watching movies or TV shows",
    "sports and exercise",
    "reading and books",
    "music and concerts",
    
    # Technology
    "using smartphones",
    "computer problems",
    "social media",
    "online shopping",
    
    # Travel
    "planning a trip",
    "airport experiences",
    "hotel stays",
    "road trips",
    
    # Emotions
    "feeling excited about something",
    "being frustrated or annoyed",
    "feeling grateful",
    "being nervous about something",
    
    # Gaming/Internet culture (important for our use case!)
    "playing online multiplayer games",
    "watching gaming streams",
    "chatting in online communities",
    "competitive gaming moments",
    
    # Casual observations
    "weather and seasons",
    "news and current events",
    "funny things that happened",
    "random thoughts and opinions",
]

# Sentence structure prompts
STRUCTURE_PROMPTS = [
    "a simple statement about",
    "someone explaining",
    "a person describing",
    "an observation about",
    "a casual comment about",
    "someone's opinion on",
    "a fact about",
    "a question about",  # We'll filter these out, but they add variety
    "someone's plan for",
    "a memory of",
]


def generate_prompt_batch(batch_size: int = 10) -> str:
    """Generate a prompt asking for multiple diverse sentences."""
    # Pick random topics
    topics = random.sample(TOPIC_CATEGORIES, min(5, len(TOPIC_CATEGORIES)))
    
    prompt = f"""Generate {batch_size} short, natural English sentences. Each sentence should be:
- 5 to 25 words long
- Grammatically correct standard English
- Sound like something a real person would say or write
- Cover different topics

Topics to include: {', '.join(topics)}

IMPORTANT RULES:
- Use ONLY standard English letters and punctuation
- Do NOT use numbers for letters (no "l33t" or "h3llo")
- Do NOT use internet slang or abbreviations yet (no "lol", "brb", "u" for "you")
- Each sentence on a new line
- No numbering or bullets

Example output format:
The weather is perfect for a walk in the park today.
I really enjoyed watching that movie last night.
My computer keeps crashing whenever I open too many tabs.

Now generate {batch_size} diverse sentences:"""
    
    return prompt


def query_ollama(prompt: str, model: str = "qwen2.5:32b") -> str:
    """Send a prompt to Ollama and get the response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.9,  # High for diversity
                    "top_p": 0.95,
                    "num_predict": 1024,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"       [ERROR] Ollama request failed: {e}")
        return ""


def extract_sentences(response: str) -> list:
    """Extract individual sentences from LLM response."""
    sentences = []
    
    for line in response.strip().split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Remove common prefixes (numbering, bullets)
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•*]\s*', '', line)
        line = re.sub(r'^[A-Z][.)]\s*', '', line)
        
        # Clean up
        line = line.strip()
        
        if line:
            sentences.append(line)
    
    return sentences


def is_valid_generated(text: str) -> bool:
    """Check if generated sentence is valid."""
    words = text.split()
    
    # Length check
    if len(words) < 5 or len(words) > 30:
        return False
    
    # Must have letters
    if not any(c.isalpha() for c in text):
        return False
    
    # Skip questions (we want statements)
    if text.endswith('?'):
        return False
    
    # Must end with proper punctuation (or we add it)
    if not text[-1] in '.!':
        if text[-1].isalpha():
            text = text + '.'
        else:
            return False
    
    # Skip leetspeak patterns
    if re.search(r'[a-zA-Z]\d[a-zA-Z]', text):
        return False
    
    # Skip if contains common slang we want to add later
    slang_patterns = ['lol', 'brb', 'idk', 'omg', 'wtf', 'tbh', ' u ', ' ur ', ' r ']
    for pattern in slang_patterns:
        if pattern in text.lower():
            return False
    
    return True


def generate_sentences(
    output_path: str,
    model: str = "qwen2.5:32b",
    target_count: int = 15000,
    batch_size: int = 20,
    ollama_url: str = OLLAMA_URL,
    workers: int = 4
) -> int:
    """
    Generate diverse clean English sentences using LLM.
    """
    global OLLAMA_URL
    OLLAMA_URL = ollama_url
    
    print("=" * 60)
    print("LLM SENTENCE GENERATION")
    print("=" * 60)
    print(f"\n  Model: {model}")
    print(f"  Target: {target_count} sentences")
    print(f"  Workers: {workers}")
    print(f"  Ollama: {ollama_url}")
    
    # Test connection
    print(f"\n[1/3] Testing Ollama connection...")
    test_response = query_ollama("Say 'hello' in one word.", model)
    if not test_response:
        print("[ERROR] Cannot connect to Ollama.")
        return 0
    print("       Connection OK!")
    
    print(f"\n[2/3] Generating sentences...")
    
    valid_sentences = set()
    pbar = tqdm(total=target_count)
    
    def worker_job():
        local_valid = []
        prompt = generate_prompt_batch(batch_size)
        response = query_ollama(prompt, model)
        if response:
            sentences = extract_sentences(response)
            for sent in sentences:
                if is_valid_generated(sent):
                    local_valid.append(sent)
        return local_valid

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(valid_sentences) < target_count:
            # Submit a batch of jobs
            futures = [executor.submit(worker_job) for _ in range(workers * 2)]
            
            for future in as_completed(futures):
                results = future.result()
                for res in results:
                    if len(valid_sentences) < target_count:
                        if res not in valid_sentences:
                            valid_sentences.add(res)
                            pbar.update(1)
                
                if len(valid_sentences) >= target_count:
                    break
    
    pbar.close()
    
    print(f"\n[3/3] Saving {len(valid_sentences)} sentences to {output_path}...")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    sentences_list = list(valid_sentences)[:target_count]
    
    with open(output_file, 'w') as f:
        for sent in sentences_list:
            data = {"text": sent, "source": "llm_generated"}
            f.write(json.dumps(data) + '\n')
    
    print(f"\n✓ Done! Saved {len(sentences_list)} sentences")
    print(f"  Output: {output_path}")
    
    return len(sentences_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean sentences with LLM")
    parser.add_argument("--output", "-o", default="llm_sentences.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--count", "-n", type=int, default=15000,
                        help="Target number of sentences (default: 15000)")
    parser.add_argument("--model", "-m", default="qwen2.5:32b",
                        help="Ollama model name (default: qwen2.5:32b)")
    parser.add_argument("--batch", "-b", type=int, default=20,
                        help="Sentences per LLM call (default: 20)")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate",
                        help="Ollama API URL")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    generate_with_llm(args.output, args.model, args.count, args.batch, args.ollama_url, args.workers)
