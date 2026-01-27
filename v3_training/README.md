# v3 training pipeline

## overview

v3 uses a hybrid approach for training data generation.
the key insight: llm handles semantics, script handles visuals.

```
source collection     llm corruption        script corruption
(wikitext + eli5)  -> (slang, no leet)  ->  (10x multiplex)
50k clean             50k slang pairs       500k training pairs
```

## why this works

| component | job                    | example                          |
|-----------|------------------------|----------------------------------|
| llm       | semantic/phonetic      | "i dont know" -> "idk"           |
| script    | visual char swaps      | "idk wut to do" -> "1dk wut 2 d0"|

the llm understands vibes and culture. the script is consistent and tunable.
together they create diverse, realistic training data.

## data pipeline

```
data_pipeline/
├── 1_source_collection/
│   ├── collect_wikitext.py      # wikitext-103
│   ├── collect_eli5.py          # eli5 subreddit
│   ├── generate_with_llm.py     # llm-generated sentences
│   └── merge_sources.py         # combine and dedupe
│
├── 2_llm_corruption/
│   ├── slangify_with_llm.py     # clean -> slang (no leetspeak)
│   └── prompts.py               # 5 personas for variety
│
└── 3_script_corruption/
    └── corrupt_v3.py            # slang -> leetspeak (10x multiplex)
```

## quick start

```bash
# 1. collect clean english
cd data_pipeline/1_source_collection
python collect_wikitext.py
python collect_eli5.py
python generate_with_llm.py
python merge_sources.py

# 2. llm slangify (needs ollama with qwen 32b)
cd ../2_llm_corruption
python slangify_with_llm.py

# 3. script corruption (cpu is fine)
cd ../3_script_corruption
python corrupt_v3.py

# output: training_data_v3.jsonl
```

## hardware

- data generation: rtx 5090 with ollama (qwen 2.5 32b)
- training: a100 (tensordock/lambda)

## key improvements over v2

- llm-based semantic corruption (not just char swaps)
- 10x multiplexing (50k llm calls -> 500k training pairs)
- digit protection (wont destroy "2nite" from llm)
- no protected words (we corrupt slang too: thx -> 7hx)
- suffix z handling (skills -> skillz)
