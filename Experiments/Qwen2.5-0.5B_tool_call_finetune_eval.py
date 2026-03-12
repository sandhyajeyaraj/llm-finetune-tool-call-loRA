# ============================================================
#  Tool Call Evaluation Script for Qwen 2.5 LoRA
#  Run this AFTER training to measure model accuracy
# ============================================================

# !pip install -q transformers peft accelerate bitsandbytes

# ── HuggingFace login (avoids rate limits) ───────────────────
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")   # replace with your token

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ────────────────────────────────────────────────────────────
# 1. CONFIG
# ────────────────────────────────────────────────────────────
BASE_MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
USE_4BIT       = True

# Auto-detect adapter path
import os, subprocess
_result = subprocess.run(["find", "/content", "-name", "adapter_config.json"], capture_output=True, text=True)
_paths  = [p.replace("/adapter_config.json", "") for p in _result.stdout.strip().splitlines() if p]
if not _paths:
    raise FileNotFoundError(
        "No adapter_config.json found under /content.\n"
        "Please re-run the training script first, making sure OUTPUT_DIR is set to an absolute path like /content/qwen25-tool-lora-final"
    )
ADAPTER_PATH = sorted(_paths)[-1]   # pick the latest if multiple
print(f"Auto-detected adapter at: {ADAPTER_PATH}")

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When you need to use a tool, respond ONLY with a JSON object like:\n"
    '{"tool": "<n>", "arguments": {<args>}}\n'
    "After receiving the tool result, give a natural language answer."
)

# ────────────────────────────────────────────────────────────
# 2. EVALUATION DATASET
#    ground_truth = what the model SHOULD output
# ────────────────────────────────────────────────────────────
EVAL_EXAMPLES = [
    # ── Seen during training (in-distribution) ───────────────
    {
        "split": "train-dist",
        "user": "What's the weather like in Tokyo right now?",
        "ground_truth": {"tool": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}},
    },
    {
        "split": "train-dist",
        "user": "What is 15% of 840?",
        "ground_truth": {"tool": "calculator", "arguments": {"expression": "0.15 * 840"}},
    },
    {
        "split": "train-dist",
        "user": "What's the current stock price of Apple?",
        "ground_truth": {"tool": "get_stock_price", "arguments": {"ticker": "AAPL"}},
    },
    {
        "split": "train-dist",
        "user": "Find me some Python tutorials for beginners.",
        "ground_truth": {"tool": "search_web", "arguments": {"query": "Python tutorials for beginners", "max_results": 3}},
    },
    {
        "split": "train-dist",
        "user": "What is Microsoft's stock price?",
        "ground_truth": {"tool": "get_stock_price", "arguments": {"ticker": "MSFT"}},
    },
    # ── Unseen during training (out-of-distribution) ─────────
    {
        "split": "out-of-dist",
        "user": "What is the weather in Sydney?",
        "ground_truth": {"tool": "get_weather", "arguments": {"city": "Sydney", "units": "celsius"}},
    },
    {
        "split": "out-of-dist",
        "user": "What is the square root of 144?",
        "ground_truth": {"tool": "calculator", "arguments": {"expression": "144 ** 0.5"}},
    },
    {
        "split": "out-of-dist",
        "user": "Search for news about electric vehicles.",
        "ground_truth": {"tool": "search_web", "arguments": {"query": "news about electric vehicles", "max_results": 3}},
    },
    {
        "split": "out-of-dist",
        "user": "Get me the stock price of Amazon.",
        "ground_truth": {"tool": "get_stock_price", "arguments": {"ticker": "AMZN"}},
    },
    {
        "split": "out-of-dist",
        "user": "How hot is it in Singapore right now?",
        "ground_truth": {"tool": "get_weather", "arguments": {"city": "Singapore", "units": "celsius"}},
    },
]

# ────────────────────────────────────────────────────────────
# 3. LOAD MODEL + ADAPTER
# ────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
) if USE_4BIT else None

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

print("Loading LoRA adapter ...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model ready.\n")

# ────────────────────────────────────────────────────────────
# 4. INFERENCE HELPER
# ────────────────────────────────────────────────────────────
def get_model_output(user_query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

# ────────────────────────────────────────────────────────────
# 5. SCORING FUNCTIONS
# ────────────────────────────────────────────────────────────
def parse_json_output(text: str):
    """Try to extract JSON from model output. Returns dict or None."""
    text = text.strip()
    # Find the first { ... } block
    try:
        start = text.index("{")
        end   = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None

def score_prediction(predicted: dict | None, ground_truth: dict) -> dict:
    """Score a single prediction against ground truth."""
    if predicted is None:
        return {
            "json_valid":    False,
            "tool_correct":  False,
            "args_correct":  False,
            "partial_score": 0.0,
            "full_score":    False,
        }

    tool_correct = predicted.get("tool") == ground_truth.get("tool")

    # Check args: all required keys present with correct values
    gt_args   = ground_truth.get("arguments", {})
    pred_args = predicted.get("arguments", {})
    args_correct = all(
        pred_args.get(k) == v for k, v in gt_args.items()
    )

    partial = (0.4 * tool_correct) + (0.6 * args_correct)

    return {
        "json_valid":    True,
        "tool_correct":  tool_correct,
        "args_correct":  args_correct,
        "partial_score": round(partial, 2),
        "full_score":    tool_correct and args_correct,
    }

# ────────────────────────────────────────────────────────────
# 6. RUN EVALUATION
# ────────────────────────────────────────────────────────────
print("=" * 65)
print(f"{'TOOL CALL EVALUATION RESULTS':^65}")
print("=" * 65)

results     = []
seen_pass   = 0
unseen_pass = 0
seen_total  = 0
unseen_total= 0

for i, ex in enumerate(EVAL_EXAMPLES):
    raw_output = get_model_output(ex["user"])
    parsed     = parse_json_output(raw_output)
    scores     = score_prediction(parsed, ex["ground_truth"])

    results.append({**ex, "raw_output": raw_output, "parsed": parsed, **scores})

    tag = "[TRAIN]" if ex["split"] == "train-dist" else "[UNSEEN]"
    ok  = "PASS" if scores["full_score"] else "FAIL"
    print(f"\n{tag} Q: {ex['user']}")
    print(f"  Expected : {json.dumps(ex['ground_truth'])}")
    print(f"  Got      : {raw_output[:120]}")
    print(f"  JSON OK  : {scores['json_valid']} | Tool OK: {scores['tool_correct']} | Args OK: {scores['args_correct']} | [{ok}]")

    if ex["split"] == "train-dist":
        seen_total += 1
        seen_pass  += scores["full_score"]
    else:
        unseen_total += 1
        unseen_pass  += scores["full_score"]

# ────────────────────────────────────────────────────────────
# 7. SUMMARY
# ────────────────────────────────────────────────────────────
total      = len(results)
total_pass = sum(r["full_score"] for r in results)
json_pass  = sum(r["json_valid"] for r in results)
tool_pass  = sum(r["tool_correct"] for r in results)
args_pass  = sum(r["args_correct"] for r in results)

print("\n" + "=" * 65)
print(f"{'SUMMARY':^65}")
print("=" * 65)
print(f"  JSON Valid Outputs  : {json_pass}/{total}  ({100*json_pass/total:.1f}%)")
print(f"  Correct Tool Name   : {tool_pass}/{total}  ({100*tool_pass/total:.1f}%)")
print(f"  Correct Arguments   : {args_pass}/{total}  ({100*args_pass/total:.1f}%)")
print(f"  Full Match (Strict) : {total_pass}/{total}  ({100*total_pass/total:.1f}%)")
print(f"\n  In-distribution     : {seen_pass}/{seen_total}  ({100*seen_pass/seen_total:.1f}%)")
print(f"  Out-of-distribution : {unseen_pass}/{unseen_total}  ({100*unseen_pass/unseen_total:.1f}%)")
print("=" * 65)
print("\nInterpretation:")
print(f"  JSON Valid  → Can the model output parseable tool calls?")
print(f"  Tool Name   → Does it know WHICH tool to use?")
print(f"  Arguments   → Does it know HOW to call the tool?")
print(f"  Full Match  → Both tool + args exactly right")
print(f"  Out-of-dist → Generalization beyond training examples")
