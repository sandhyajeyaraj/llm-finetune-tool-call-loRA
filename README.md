# 🛠️ Qwen 2.5-0.5B Tool Call Fine-Tuning (Phase 1 — Synthetic SFT)

Fine-tuning **Qwen 2.5-0.5B** on a small synthetic dataset to make structured JSON tool calls using QLoRA.

> This is **Phase 1** of a 3-phase experiment. See the full experiment repo for SFT + GRPO on real data.

---

## 📌 What This Does

Trains a 0.5B model to respond to natural language with structured tool calls:

```json
{"tool": "get_stock_price", "arguments": {"ticker": "MSFT"}}
```

---

## 🤖 Setup

| | |
|---|---|
| **Model** | Qwen/Qwen2.5-0.5B-Instruct |
| **Method** | QLoRA (4-bit, LoRA rank 16) |
| **Dataset** | 14 hand-written synthetic examples |
| **Tools** | get_weather, search_web, calculator, get_stock_price |
| **Platform** | Google Colab (T4 GPU) |

---

## 📊 Results

Evaluated on 10 queries (5 seen during training, 5 unseen):

| Metric | Score |
|--------|-------|
| JSON Valid | 30% |
| Correct Tool Name | 20% |
| Correct Arguments | 10% |
| Full Match (strict) | 0% |
| In-distribution | 0/5 |
| Out-of-distribution | 0/5 |

**Key observations:**
- Model learned the concept of JSON tool calls but hallucinated tool names (`"weather"` instead of `"get_weather"`)
- Wrong argument keys (`"ticker_symbol"` instead of `"ticker"`)
- Calculator answered directly in plain text instead of calling the tool
- 14 examples is too small — model memorises poorly and doesn't generalise


---

## 📁 Files

| File | Description |
|------|-------------|
| `qwen25_tool_call_finetune.py` | Training script — SFT with synthetic data |
| `Qwen2.5-0.5B_tool_call_finetune_eval.py` | Evaluation script — scores model on 10 queries |

---

## 💡 Lessons Learned

- **14 examples is not enough** — model fails to generalise even to similar queries
- **Exact argument matching is strict** — `"ticker_symbol"` vs `"ticker"` counts as failure
- **Bigger dataset + RL needed** → See Phase 2 & 3 for improvement

---

## 🔧 Stack

`Qwen2.5-0.5B` • `QLoRA` • `SFT` • `trl` • `peft` • `Google Colab T4`