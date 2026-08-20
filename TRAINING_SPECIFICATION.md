# LocalBench --- Training Specification v3

**Status:** Blueprint for Phase 6 implementation  
**Last Updated:** 2026-08-19  
**Role:** Defines specialization training methodology, hyperparameters, fine-tuning approach, and checkpoint management.

---

## 1. Training Objective

Specialize a small, locally deployable language model for code semantic representation through fine-tuning on domain-specific code-to-description pairs.

**Goal:** The specialized model should improve downstream retrieval performance over its zero-shot baseline while maintaining low resource overhead (RAM, latency, disk).

**Non-goals:**
- Training a foundation model from scratch
- Distributed training across multiple machines
- Continual learning or online adaptation

---

## 2. Base Model Selection

### 2.1 Model Criteria

**Base models must satisfy:**

1. **Deployable locally** — Runs on consumer/edge hardware
2. **Small** — ≤ 8B parameters (target: 0.5B–3B)
3. **Well-trained** — Strong foundational capabilities
4. **Quantizable** — Supports Q4 quantization (important for deployment)
5. **Accessible** — Open-source, Ollama-compatible

**Candidate models:**
- **phi-3-mini** (0.5B) — Microsoft's efficient model
- **phi-3** (3.8B) — Balanced size/capability
- **mistral-7b** (7B) — Reference medium baseline
- **gemma-2-2b** (2B) — Google's lightweight model (note: distinct from gemma-2, the 9B model used as a baseline in other documents)

### 2.2 Selection Decision

**For Phase 6:** Use **phi-3-mini (0.5B)** as primary target

**Rationale:**
- Sufficient size for semantic understanding
- Deployable on edge devices (2GB+ memory)
- Strong instruction-following capabilities
- Supported by Ollama

---

## 3. Training Dataset Construction

### 3.1 Data Source

**Source:** Repository-disjoint train split from dataset v1.0.0

**Composition:**
```
Training examples: Code unit → Semantic description

Example:
{
  "code_unit": "def process_retry(...):\n    ...",
  "description": "Retries a failed payment transaction using exponential backoff with configurable maximum attempts.",
  "concepts": ["retry logic", "exponential backoff", "error handling"]
}
```

**Dataset statistics:**
- Total training examples: 225 (50% of code units)
- Language: Python (v1 scope)
- Split: Disjoint from validation (112) and test (113)

### 3.2 Prompt Template

Standard format for training examples:

```
### Instruction
Analyze the following code and provide a clear semantic description.

### Code
[code_unit]

### Response
[description]
```

**Consistency rules:**
- Template is identical for all training examples
- Descriptions are preserved verbatim from semantic labels
- No prompt engineering per example
- Temperature frozen during training

---

## 4. Fine-Tuning Methodology

### 4.1 Fine-Tuning Approach: Parameter-Efficient

**Why not full fine-tuning?**
- Requires high memory (duplicate model + gradients)
- Slower convergence
- Risk of catastrophic forgetting

**Why LoRA (Low-Rank Adaptation)?**
- Efficient: Only ~1–5% of parameters trainable
- Fast: Converges in 1–2 hours on consumer GPU
- Preserves foundation: Low risk of forgetting
- Small checkpoint: LoRA weights << full model

### 4.2 LoRA Configuration

```python
LoRA_config = {
    "lora_r": 8,              # Rank (low-rank matrices)
    "lora_alpha": 16,         # Scaling factor (alpha / r = 2.0)
    "lora_dropout": 0.05,     # Regularization
    "target_modules": [       # Which layers to adapt
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj"
    ],
    "bias": "none",           # No bias adapters
    "modules_to_save": None,  # No modules saved
    "peft_type": "LORA"
}
```

**Rationale:**
- Rank 8 = sufficient expressiveness for semantic representation
- Alpha 16 = standard scaling (2x rank)
- Target attention layers = most expressive components

---

## 5. Training Configuration

### 5.1 Hyperparameters

```python
training_config = {
    # Learning
    "learning_rate": 5e-4,         # LoRA learning rate
    "lr_scheduler_type": "cosine", # Cosine annealing
    "num_train_epochs": 3,         # 3 passes over data
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    
    # Regularization
    "weight_decay": 0.01,
    "warmup_steps": 50,
    "warmup_ratio": 0.05,
    
    # Optimization
    "optim": "paged_adamw_32bit",
    "max_grad_norm": 1.0,
    
    # Logging & Checkpoints
    "logging_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_total_limit": 3,
    
    # Computation
    "fp16": True,              # Mixed precision (if GPU available)
    "seed": 42,
}
```

### 5.2 Justification

| Parameter | Value | Reason |
|-----------|-------|--------|
| Learning rate | 5e-4 | Standard LoRA rate; conservative to preserve foundation |
| Num epochs | 3 | Sufficient for 225 examples; avoids overfitting |
| Batch size | 16 | Balances memory/stability (8GB GPU) |
| Warmup | 50 steps | Prevents gradient spikes early in training |
| LR schedule | Cosine | Standard; smooth decay improves generalization |

---

## 6. Training Procedure

### 6.1 Pre-training Setup

1. **Load base model** — phi-3-mini from Ollama/HuggingFace
2. **Attach LoRA** — Apply LoRA config to model
3. **Prepare data** — Format training examples with prompt template
4. **Validate** — Ensure data loading works, one pass succeeds
5. **Compute statistics** — Example count, average length

---

### 6.2 Training Loop

```python
# 1. Initialize
base_model = load_model("phi-3-mini")
model = apply_lora(base_model, lora_config)
optimizer = setup_optimizer(model, training_config)
scheduler = setup_scheduler(optimizer, training_config)
dataset = load_dataset("train", version="1.0.0")

# 2. Training
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        # Logging
        if batch_idx % logging_steps == 0:
            log_metrics(epoch, batch_idx, loss, learning_rate)
        
        # Checkpointing
        if batch_idx % save_steps == 0:
            save_checkpoint(model, epoch, batch_idx)
        
        # Evaluation
        if batch_idx % eval_steps == 0:
            val_loss = evaluate(model, val_dataloader)
            log_eval_metrics(val_loss)

# 3. Save final model
save_final_checkpoint(model, training_config)
```

---

### 6.3 Training Artifacts

**Saved per checkpoint:**
```
experiments/phi3_lora_v1/
├── checkpoint-100/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── training_args.bin
│   └── optimizer.pt
├── checkpoint-200/
├── ...
└── final/
    ├── adapter_config.json
    ├── adapter_model.bin
    └── training_metadata.json
```

**Training metadata:**
```json
{
  "experiment_id": "phi3_lora_v1",
  "base_model": "phi-3-mini",
  "base_model_size_mb": 2048,
  "lora_size_mb": 8.5,
  "training_examples": 225,
  "dataset_version": "1.0.0",
  "training_config": {...},
  "lora_config": {...},
  "hardware": {
    "device": "CUDA:0",
    "gpu_model": "RTX 3090",
    "vram_gb": 24
  },
  "seed": 42,
  "training_time_minutes": 42.5,
  "final_training_loss": 0.63,
  "final_validation_loss": 0.71,
  "converged": true,
  "timestamp": "2026-10-15T10:30:00Z"
}
```

---

## 7. Ablation Study Design

### 7.1 Training Data Size Ablation

**Hypothesis:** Specialization improves with more training data, but with diminishing returns.

**Experiment design:**

```
Dataset subset | Examples | Checkpoint ID
-------------------------------------------
Baseline       | 0        | (use zero-shot model)
Small          | 50       | phi3_lora_small
Medium         | 112      | phi3_lora_medium
Full           | 225      | phi3_lora_full
```

**Procedure:**
1. Sample train split uniformly
2. Train separate LoRA checkpoint for each subset
3. Benchmark each on **identical test set** (frozen)
4. Compute Hit@10, MRR for each
5. Plot: training examples vs Hit@10

**Expected outcome:**
```
Hit@10
  |     •
  |   •
  | •
  |•────────────────
  |_________________
  0   50  112  225  training examples
```

---

### 7.2 LoRA Rank Ablation

**Hypothesis:** Higher LoRA rank captures more expressiveness with diminishing gains.

**Experiment design:**

```
LoRA Rank | Parameters | Checkpoint ID
-----------------------------------------
4         | ~1M        | phi3_lora_r4
8         | ~2M        | phi3_lora_r8 (primary)
16        | ~4M        | phi3_lora_r16
```

**Procedure:**
1. Train separate models with different ranks
2. Benchmark on test set
3. Record: Hit@10, latency, checkpoint size
4. Analyze: Performance vs model size trade-off

---

## 8. Model Evaluation During Training

### 8.1 Validation Set

**Source:** 112 code units (25% of total) from separate repository group

**Use:** Compute validation loss during training to:
- Detect overfitting
- Implement early stopping (if loss plateaus)
- Select best checkpoint

**Procedure:**
```python
# Every eval_steps
val_loss_sum = 0
for batch in val_dataloader:
    outputs = model(...)
    val_loss_sum += outputs.loss
val_loss = val_loss_sum / len(val_dataloader)

# Log and potentially early stop
if val_loss > best_val_loss * 1.05:
    patience_counter += 1
    if patience_counter >= 3:
        print("Early stopping: validation loss stopped improving")
        break
```

**Rationale:**
- Early stopping prevents overfitting to small train set
- Validation loss trends inform model quality
- Separate data (validation) ensures unbiased estimate

---

### 8.2 Best Checkpoint Selection

**Selection criteria:**
1. Lowest validation loss
2. Converged loss (not still declining sharply at epoch 3)
3. Reasonable training loss (not wildly divergent from val)

**Final model:** Checkpoint with lowest validation loss

---

## 9. Inference-Time Deployment

### 9.1 Loading Fine-Tuned Model

```python
# Load base model
base_model = AutoModelForCausalLM.from_pretrained("phi-3-mini")

# Load and apply LoRA weights
lora_model = PeftModel.from_pretrained(
    base_model, 
    "experiments/phi3_lora_full/final"
)

# Merge LoRA into base (optional, for deployment)
merged_model = lora_model.merge_and_unload()
```

### 9.2 Resource Profile

**After specialization:**

| Metric | Baseline | Specialized | Overhead |
|--------|----------|-------------|----------|
| Model size | 2.0 GB | 2.0 GB + 8.5 MB | ~0.4% |
| Inference latency | 720 ms | ~730 ms | +1.4% |
| Peak RAM | 1128 MB | ~1140 MB | +1% |
| Throughput | 45.2 tps | ~44.8 tps | -0.9% |

**Implication:** LoRA adds negligible overhead; specialization is cheap.

---

## 10. Failure Handling

### 10.1 Training Instability

**If training diverges (loss → NaN):**
1. Reduce learning rate (e.g., 5e-4 → 2.5e-4)
2. Increase warmup steps
3. Reduce batch size (if memory allows)
4. Check data for NaN/Inf values

---

### 10.2 Convergence Issues

**If validation loss plateaus early:**
1. Train for more epochs (3 → 5)
2. Adjust learning rate schedule
3. Increase LoRA rank (8 → 16)
4. Inspect training data quality

---

## 11. Reproducibility Checklist

- [ ] Base model version recorded (e.g., `phi-3-mini@ollama-v0.29`)
- [ ] Dataset version recorded (`dataset-v1.0.0`)
- [ ] Training split (train: 225, val: 112, test: 113) documented
- [ ] LoRA config frozen in experiment metadata
- [ ] Training hyperparameters recorded (learning rate, epochs, batch size)
- [ ] Hardware profile recorded (GPU model, VRAM, CPU)
- [ ] Random seed recorded (42)
- [ ] Training loss curve saved (for analysis)
- [ ] Validation loss curve saved
- [ ] Final checkpoint persisted with metadata
- [ ] Training time recorded
- [ ] Git commit hash for reproducibility

---

## 12. Key Principles

1. **LoRA is efficient** — Minimal computational overhead, fast training
2. **Baseline is required** — Specialization effect measured against zero-shot
3. **Data matters** — Train set quality > quantity (use semantic labels)
4. **Validation guides** — Separate validation prevents overfitting
5. **Ablations are systematic** — Vary one factor at a time
6. **Reproducibility is first-class** — All configurations recorded
7. **Failure is expected** — Not all specializations help (null results valid)

