# ============================================================
#  Fine-Tuning Qwen 2.5 with QLoRA for Tool / Function Calls
#  Based on: huggingface.co/arvindcr4/tool-call-lora-qwen0.5b
#  Tested working on Google Colab (T4 GPU)
# ============================================================

# ── STEP 0: Run this first in Colab ─────────────────────────
# !pip install -q transformers datasets peft trl accelerate bitsandbytes

# ────────────────────────────────────────────────────────────
# 1. IMPORTS
# ────────────────────────────────────────────────────────────
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# Optional: set HuggingFace token to avoid rate limits
# from huggingface_hub import login
# from google.colab import userdata
# login(token=userdata.get("HF_TOKEN"))

# ────────────────────────────────────────────────────────────
# 2. CONFIG
# ────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-0.5B-Instruct"   # 0.5B = most stable on free T4
                                                # swap to 1.5B or 3B for better quality
OUTPUT_DIR   = "./qwen25-tool-lora"
MAX_SEQ_LEN  = 1024
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TRAIN_EPOCHS = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
LR           = 2e-4
USE_4BIT     = True    # QLoRA — keeps VRAM low on free T4

# ────────────────────────────────────────────────────────────
# 3. SYNTHETIC TOOL-CALL DATASET
# ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When you need to use a tool, respond ONLY with a JSON object like:\n"
    '{"tool": "<name>", "arguments": {<args>}}\n'
    "After receiving the tool result, give a natural language answer."
)

RAW_EXAMPLES = [
    # Weather
    {
        "user": "What's the weather like in Tokyo right now?",
        "tool_call": {"tool": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}},
        "tool_result": {"temperature": 22, "condition": "Partly cloudy", "humidity": "60%"},
        "assistant": "The current weather in Tokyo is 22C and partly cloudy with 60% humidity.",
    },
    {
        "user": "Is it going to be hot in Dubai today?",
        "tool_call": {"tool": "get_weather", "arguments": {"city": "Dubai", "units": "celsius"}},
        "tool_result": {"temperature": 41, "condition": "Sunny", "humidity": "20%"},
        "assistant": "Yes, very hot in Dubai today at 41C and sunny with low humidity (20%).",
    },
    {
        "user": "What's the weather in New York in Fahrenheit?",
        "tool_call": {"tool": "get_weather", "arguments": {"city": "New York", "units": "fahrenheit"}},
        "tool_result": {"temperature": 75, "condition": "Clear", "humidity": "45%"},
        "assistant": "New York is currently 75F with clear skies and 45% humidity.",
    },
    {
        "user": "How cold is it in London today?",
        "tool_call": {"tool": "get_weather", "arguments": {"city": "London", "units": "celsius"}},
        "tool_result": {"temperature": 9, "condition": "Rainy", "humidity": "85%"},
        "assistant": "It's quite cold in London today at 9C with rain and high humidity at 85%.",
    },
    # Web search
    {
        "user": "Search for the latest news on quantum computing.",
        "tool_call": {"tool": "search_web", "arguments": {"query": "latest news quantum computing", "max_results": 3}},
        "tool_result": {"results": ["IBM unveils 1000-qubit processor", "Google claims quantum supremacy milestone", "Quantum computing startups raise record funding"]},
        "assistant": "Latest quantum computing headlines:\n1. IBM unveiled a 1000-qubit processor.\n2. Google announced a new quantum supremacy milestone.\n3. Quantum computing startups are raising record funding.",
    },
    {
        "user": "Find me some Python tutorials for beginners.",
        "tool_call": {"tool": "search_web", "arguments": {"query": "Python tutorials for beginners", "max_results": 3}},
        "tool_result": {"results": ["Official Python docs tutorial", "Real Python beginner guide", "freeCodeCamp Python course"]},
        "assistant": "Great resources for learning Python:\n1. Official Python docs tutorial\n2. Real Python beginner guide\n3. freeCodeCamp Python course",
    },
    {
        "user": "Search for recent AI research papers.",
        "tool_call": {"tool": "search_web", "arguments": {"query": "recent AI research papers 2024", "max_results": 3}},
        "tool_result": {"results": ["Attention follow-up paper", "LLM scaling laws revisited", "Multimodal AI advances"]},
        "assistant": "Recent AI research highlights:\n1. Follow-up work on transformer attention.\n2. New findings on LLM scaling laws.\n3. Advances in multimodal AI.",
    },
    # Calculator
    {
        "user": "What is 15% of 840?",
        "tool_call": {"tool": "calculator", "arguments": {"expression": "0.15 * 840"}},
        "tool_result": {"result": 126.0},
        "assistant": "15% of 840 is 126.",
    },
    {
        "user": "Calculate the area of a circle with radius 7.",
        "tool_call": {"tool": "calculator", "arguments": {"expression": "3.14159 * 7 ** 2"}},
        "tool_result": {"result": 153.938},
        "assistant": "The area of a circle with radius 7 is approximately 153.94 square units.",
    },
    {
        "user": "How much is (128 / 4) + 56 * 3?",
        "tool_call": {"tool": "calculator", "arguments": {"expression": "(128 / 4) + 56 * 3"}},
        "tool_result": {"result": 200.0},
        "assistant": "The result of (128 / 4) + 56 x 3 is 200.",
    },
    # Stock price
    {
        "user": "What's the current stock price of Apple?",
        "tool_call": {"tool": "get_stock_price", "arguments": {"ticker": "AAPL"}},
        "tool_result": {"ticker": "AAPL", "price": 189.45, "currency": "USD"},
        "assistant": "Apple (AAPL) is currently trading at $189.45 USD.",
    },
    {
        "user": "How is Tesla stock doing today?",
        "tool_call": {"tool": "get_stock_price", "arguments": {"ticker": "TSLA"}},
        "tool_result": {"ticker": "TSLA", "price": 245.10, "currency": "USD"},
        "assistant": "Tesla (TSLA) is currently priced at $245.10 USD.",
    },
    {
        "user": "Give me the latest price for NVIDIA shares.",
        "tool_call": {"tool": "get_stock_price", "arguments": {"ticker": "NVDA"}},
        "tool_result": {"ticker": "NVDA", "price": 875.30, "currency": "USD"},
        "assistant": "NVIDIA (NVDA) is currently trading at $875.30 USD.",
    },
    {
        "user": "What is Microsoft's stock price?",
        "tool_call": {"tool": "get_stock_price", "arguments": {"ticker": "MSFT"}},
        "tool_result": {"ticker": "MSFT", "price": 415.20, "currency": "USD"},
        "assistant": "Microsoft (MSFT) is currently trading at $415.20 USD.",
    },
]


def build_chat_text(example, tokenizer):
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": example["user"]},
        {"role": "assistant", "content": json.dumps(example["tool_call"])},
        {"role": "tool",      "content": json.dumps(example["tool_result"])},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ────────────────────────────────────────────────────────────
# 4. TOKENIZER
# ────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

texts   = [build_chat_text(ex, tokenizer) for ex in RAW_EXAMPLES]
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.15, seed=42)
print(f"Train samples: {len(dataset['train'])}  |  Eval samples: {len(dataset['test'])}")

# ────────────────────────────────────────────────────────────
# 5. MODEL
# ────────────────────────────────────────────────────────────
print("Loading model ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
) if USE_4BIT else None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.config.use_cache = False

# ────────────────────────────────────────────────────────────
# 6. LoRA CONFIG  (matches arvindcr4/tool-call-lora-qwen0.5b)
# ────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ────────────────────────────────────────────────────────────
# 7. SFTConfig + SFTTrainer  (trl >= 0.16)
# ────────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
    seed=42,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    args=sft_config,
)

# ────────────────────────────────────────────────────────────
# 8. TRAIN
# ────────────────────────────────────────────────────────────
print("\nStarting training ...\n")
trainer.train()

# ────────────────────────────────────────────────────────────
# 9. SAVE ADAPTER
# ────────────────────────────────────────────────────────────
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapter saved to {OUTPUT_DIR}")

# ────────────────────────────────────────────────────────────
# 10. QUICK INFERENCE TEST
# ────────────────────────────────────────────────────────────
def run_inference(user_query):
    model.eval()
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
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

print("\n-- Inference test --")
q = "What is the stock price of Microsoft?"
print(f"User : {q}")
print(f"Model: {run_inference(q)}")
