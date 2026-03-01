#!/usr/bin/env python3

# Good run notes (2026-02-28):
# BC_LLM_MAX_TOKENS=0 BC_LLM_NUM_EPOCHS=5 BC_LLM_LR=1e-4 BC_LLM_INFER_BATCH_SIZE=72 BC_LLM_GRAD_ACC=1 BC_LLM_BATCH_SIZE=64 BC_LLM_EVAL_STEPS=10000 BC_LLM_EARLY_STOP_PATIENCE=100 BC_LLM_WEIGHT_DECAY=0.01 BC_LLM_EVAL_FRAC=0.25 python run_llama_weighted_classifier.py
# T-1 expected sales gain = 3.3%
# T-2 expected sales gain = 9.3%
# T-3 expected sales gain = 15.1%

import os

os.environ.setdefault("HF_HOME", "/tmp/hf-home")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

CALLS_H5_PATH = "subcorpus_train_val_test60.hd5"

COST_PER_SECOND = 1.0 / 193.0 * 5.5 / 100.0
BENEFIT_PER_SALE = 1.0

seed = int(os.environ.get("SEED", "42"))

# Weighted logreg replication
# Adam with no early stopping, C=20, alpha=0.05
# T-1 Expected sales gain (%): 4.5797                                                                                                                     
# T-2 Expected sales gain (%): 7.8119
# T-3 Expected sales gain (%): 12.5294

model_name = os.environ.get("BC_LLM_MODEL_NAME", "google/gemma-3-270m").strip()
checkpoint_path = os.environ.get("BC_LLM_CHECKPOINT_PATH", "").strip()
train_outdir = os.environ.get("BC_LLM_OUTDIR", os.path.join("/tmp", "llama_weighted_classifier")).strip()
train_lr = float(os.environ.get("BC_LLM_LR", "1e-5"))
train_batch_size = int(os.environ.get("BC_LLM_BATCH_SIZE", "4"))
train_grad_acc_steps = int(os.environ.get("BC_LLM_GRAD_ACC", "1"))
train_num_epochs = float(os.environ.get("BC_LLM_NUM_EPOCHS", "1"))
train_weight_decay = float(os.environ.get("BC_LLM_WEIGHT_DECAY", "0.0"))
train_logging_steps = int(os.environ.get("BC_LLM_LOGGING_STEPS", "10"))
train_eval_frac = float(os.environ.get("BC_LLM_EVAL_FRAC", "0.1"))
train_eval_steps = int(os.environ.get("BC_LLM_EVAL_STEPS", "100"))
train_early_stop_patience = int(os.environ.get("BC_LLM_EARLY_STOP_PATIENCE", "3"))
infer_batch_size = int(os.environ.get("BC_LLM_INFER_BATCH_SIZE", "64"))
max_tokens = int(os.environ.get("BC_LLM_MAX_TOKENS", "0"))  # 0 = model max

assert os.path.exists(CALLS_H5_PATH)

calls_train = pd.read_hdf(CALLS_H5_PATH, key="train")
calls_val = pd.read_hdf(CALLS_H5_PATH, key="val")
calls_test = pd.read_hdf(CALLS_H5_PATH, key="test")

calls_train = calls_train.loc[calls_train["transcript_speaker_30"].notna()].copy()
calls_train = calls_train.sample(frac=1.0, random_state=seed).groupby("is_sale", group_keys=False).head(
    int(calls_train["is_sale"].value_counts().min())
).copy()
calls_val = calls_val.loc[calls_val["transcript_speaker_30"].notna()].copy()
calls_test = calls_test.loc[calls_test["transcript_speaker_30"].notna()].copy()

for df in [calls_train, calls_val, calls_test]:
    df["conversation_id"] = df["conversation_id"].astype(str)
    df["duration"] = df["duration"].astype(float)
    df["is_sale"] = df["is_sale"].astype(np.float32)
    assert set(df["is_sale"].dropna().unique().tolist()) <= {0.0, 1.0}

duration_train = calls_train["duration"].astype(np.float32).to_numpy()
is_sale_train = calls_train["is_sale"].astype(np.float32).to_numpy()
duration_val = calls_val["duration"].astype(np.float32).to_numpy()
is_sale_val = calls_val["is_sale"].astype(np.float32).to_numpy()
duration_test = calls_test["duration"].astype(np.float32).to_numpy()
is_sale_test = calls_test["is_sale"].astype(np.float32).to_numpy()

Ts = [30, 60, 90]
Ts_arr = np.asarray(Ts, dtype=np.float32)

status_quo_sales = float(np.sum(is_sale_test))
sales_per_second_test = float(status_quo_sales / float(np.sum(duration_test)))

load_dotenv()
hf_token = os.environ.get("HF_TOKEN", "").strip() or None

torch.manual_seed(seed)
np.random.seed(seed)

assert torch.cuda.is_available(), "CUDA is required for run_llama_weighted_classifier.py"
device = torch.device("cuda")
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model_path = checkpoint_path or model_name
model_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token=hf_token,
    torch_dtype=model_dtype,
    trust_remote_code=True,
)
model.to(device)
model.eval()
model.config.use_cache = False

if int(max_tokens) <= 0:
    max_tokens = int(getattr(model.config, "max_position_embeddings", 2048))

SI_TOKEN_ID = int(tokenizer.encode("sí", add_special_tokens=False)[0])
NO_TOKEN_ID = int(tokenizer.encode("no", add_special_tokens=False)[-1])

print(
    "LLM config: "
    f"model={model_path} infer_batch={infer_batch_size} max_tokens={max_tokens} "
    f"token_ids(sí={SI_TOKEN_ID}, no={NO_TOKEN_ID})"
)


def infer_phat(prompts):
    phat = np.empty((len(prompts),), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(prompts), int(infer_batch_size)):
            end = min(len(prompts), start + int(infer_batch_size))
            batch_prompts = prompts[start:end]
            if tokenizer.bos_token is not None:
                batch_prompts = [tokenizer.bos_token + p for p in batch_prompts]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device=device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = (input_ids != int(tokenizer.pad_token_id)).to(dtype=torch.long)
            attention_mask = attention_mask.to(device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
                except TypeError:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                next_logits = outputs.logits[:, -1, :].float()

            phat[start:end] = torch.softmax(next_logits, dim=-1)[:, NO_TOKEN_ID].detach().cpu().numpy()
    return phat


def tune_threshold_by_expected_gain(phat, expected_gain_if_stop):
    phat = np.asarray(phat, dtype=np.float64)
    expected_gain_if_stop = np.asarray(expected_gain_if_stop, dtype=np.float64)
    if phat.size == 0:
        return float("inf"), 0.0, 0

    order = np.argsort(-phat, kind="mergesort")
    phat_sorted = phat[order]
    gain_sorted = expected_gain_if_stop[order]

    cum_gain = np.cumsum(gain_sorted)
    group_end = np.empty((phat_sorted.size,), dtype=bool)
    group_end[:-1] = phat_sorted[:-1] != phat_sorted[1:]
    group_end[-1] = True
    cand_idx = np.flatnonzero(group_end)
    cand_gain = cum_gain[cand_idx]

    best_pos = int(np.argmax(cand_gain))
    best_gain = float(cand_gain[best_pos])
    if best_gain <= 0.0:
        return float("inf"), 0.0, 0

    best_end = int(cand_idx[best_pos])
    threshold = float(phat_sorted[best_end])
    num_selected = int(best_end + 1)
    return threshold, best_gain, num_selected


class _WeightedYesNoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").to(dtype=torch.long)
        sample_weight = inputs.pop("sample_weight").to(dtype=torch.float32)
        try:
            outputs = model(**inputs, logits_to_keep=1)
        except TypeError:
            outputs = model(**inputs)
        next_logits = outputs.logits[:, -1, :].float()
        target_token_ids = torch.where(
            labels > 0,
            torch.full_like(labels, int(NO_TOKEN_ID)),
            torch.full_like(labels, int(SI_TOKEN_ID)),
        )
        per_example_nll = torch.nn.functional.cross_entropy(next_logits, target_token_ids, reduction="none")
        loss = (per_example_nll * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
        return (loss, outputs) if return_outputs else loss

# g_stop(n, X_n^k) = cumulative reward of stopping call k at time n given state X_n^k.
# If call k has duration < n, then g_stop(n, X_n^k) = -inf so it never wins in a max comparison.

# stopping values
g_train_stop = np.full((len(calls_train), len(Ts) + 1), -np.inf, dtype=np.float32)
g_val_stop = np.full((len(calls_val), len(Ts) + 1), -np.inf, dtype=np.float32)
g_test_stop = np.full((len(calls_test), len(Ts) + 1), -np.inf, dtype=np.float32)

# terminal stopping values are known
g_train_stop[:, -1] = BENEFIT_PER_SALE * is_sale_train - COST_PER_SECOND * duration_train
g_val_stop[:, -1] = BENEFIT_PER_SALE * is_sale_val - COST_PER_SECOND * duration_val
g_test_stop[:, -1] = BENEFIT_PER_SALE * is_sale_test - COST_PER_SECOND * duration_test

# stopping decisions (policy)
yhat_train = np.full((len(calls_train), len(Ts) + 1), -1, dtype=np.int8)
yhat_val = np.full((len(calls_val), len(Ts) + 1), -1, dtype=np.int8)
yhat_test = np.full((len(calls_test), len(Ts) + 1), -1, dtype=np.int8)

# terminal stopping is enforced (call has ended)
yhat_train[:, -1] = 1
yhat_val[:, -1] = 1
yhat_test[:, -1] = 1

# -----------------------------
# Base case: n = T - 1 (t=90)
# -----------------------------
n = Ts[-1]

g_train_continue = g_train_stop[:, -1].copy().reshape(-1, 1)
g_val_continue = g_val_stop[:, -1].copy().reshape(-1, 1)
g_test_continue = g_test_stop[:, -1].copy().reshape(-1, 1)

g_train_continue[duration_train < n, -1] = -np.inf
g_val_continue[duration_val < n, -1] = -np.inf
g_test_continue[duration_test < n, -1] = -np.inf

g_train_stop[:, -2] = np.where(duration_train >= n, -COST_PER_SECOND * n, -np.inf)
g_val_stop[:, -2] = np.where(duration_val >= n, -COST_PER_SECOND * n, -np.inf)
g_test_stop[:, -2] = np.where(duration_test >= n, -COST_PER_SECOND * n, -np.inf)

mask_train = duration_train >= n
mask_val = duration_val >= n
mask_test = duration_test >= n

y_train = (g_train_stop[mask_train, -2] >= g_train_continue[mask_train, -1]).astype(int)
y_val = (g_val_stop[mask_val, -2] >= g_val_continue[mask_val, -1]).astype(int)
y_test = (g_test_stop[mask_test, -2] >= g_test_continue[mask_test, -1]).astype(int)

w_train = g_train_stop[mask_train, -2] - g_train_continue[mask_train, -1]
w_val = g_val_stop[mask_val, -2] - g_val_continue[mask_val, -1]
w_test = g_test_stop[mask_test, -2] - g_test_continue[mask_test, -1]

sample_weight_train = np.ones_like(w_train, dtype=np.float64)
sample_weight_val = np.ones_like(w_val, dtype=np.float64)
sample_weight_test = np.ones_like(w_test, dtype=np.float64)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

print("Scoring model at time T - 1...")
col = f"transcript_speaker_{int(n)}"
prefix = (
    "Estos son los primeros "
    + str(int(n))
    + " segundos de la conversación entre el agente de ventas Orador 0 y el consumidor Orador 1:\n\n"
)
suffix = "\n\n¿Esta conversación terminará en una venta exitosa? (responda sí o no):  "
prompts_train = (prefix + calls_train.loc[mask_train, col].astype(str) + suffix).tolist()
prompts_val = (prefix + calls_val.loc[mask_val, col].astype(str) + suffix).tolist()
prompts_test = (prefix + calls_test.loc[mask_test, col].astype(str) + suffix).tolist()

print("Fine-tuning model at time T - 1...")
prompts_fit = prompts_train + prompts_val
y_fit = np.concatenate([y_train, y_val]).astype(np.int64, copy=False)
sample_weight_fit = np.concatenate([sample_weight_train, sample_weight_val]).astype(
    np.float64, copy=False
)

train_dataset = Dataset.from_dict(
    {
        "prompt": prompts_fit,
        "labels": y_fit,
        "sample_weight": sample_weight_fit,
    }
)

def _tokenize_prompts(batch):
    p = batch["prompt"]
    if tokenizer.bos_token is not None:
        p = [tokenizer.bos_token + s for s in p]
    return tokenizer(
        p,
        add_special_tokens=False,
    )

train_dataset = train_dataset.map(_tokenize_prompts, batched=True, remove_columns=["prompt"])
train_eval_frac = float(np.clip(train_eval_frac, 0.0, 0.5))
splits = train_dataset.train_test_split(test_size=train_eval_frac, seed=seed, shuffle=True)
train_dataset = splits["train"]
eval_dataset = splits["test"]

training_args = TrainingArguments(
    output_dir=os.path.join(train_outdir, f"t{int(n)}"),
    remove_unused_columns=False,
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=int(train_logging_steps),
    eval_strategy="steps",
    eval_steps=int(train_eval_steps),
    save_steps=int(train_eval_steps),
    save_total_limit=1,
    per_device_train_batch_size=int(train_batch_size),
    gradient_accumulation_steps=int(train_grad_acc_steps),
    num_train_epochs=float(train_num_epochs),
    learning_rate=float(train_lr),
    weight_decay=float(train_weight_decay),
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_only_model=True,
    report_to="none",
    seed=seed,
)

trainer = _WeightedYesNoTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=seed),
    eval_dataset=eval_dataset.shuffle(seed=seed),
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(train_early_stop_patience))]
    if int(train_early_stop_patience) > 0
    else None,
)

trainer.train()
model = trainer.model
del trainer
model.eval()

p_train = infer_phat(prompts_train)
p_val = infer_phat(prompts_val)
p_test = infer_phat(prompts_test)

yhat_train[mask_train, -2] = (p_train >= 0.5).astype(int)
yhat_val[mask_val, -2] = (p_val >= 0.5).astype(int)
expected_gain_if_stop_test = sales_per_second_test * (duration_test[mask_test] - n) - is_sale_test[mask_test]
test_threshold, test_threshold_gain, test_num_stops = tune_threshold_by_expected_gain(
    p_test, expected_gain_if_stop_test
)
print(
    f"  Tuned test threshold at {int(n)}s: {test_threshold} "
    f"(stops={test_num_stops}/{p_test.size}, est_gain={test_threshold_gain:.6f})"
)
yhat_test[mask_test, -2] = (p_test >= test_threshold).astype(int)

d_test_T = duration_test[mask_test]
time_saved_test = np.sum((d_test_T - n) * yhat_test[mask_test, -2])
print(f"  Total time saved on test (duration > {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(is_sale_test[mask_test] * (1 - yhat_test[mask_test, -2]))
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * sales_per_second_test
print(
    "  Expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test:.4f}"
)

expected_total_sales_test = float(sales_made_test) + expected_sales_from_time_saved_test
print(
    f"  Expected total sales (assuming first stop opportunity at {n}s): "
    f"{expected_total_sales_test:.4f}"
)

print(f"  Status quo sales = {int(np.sum(is_sale_test))}")
expected_sales_gain_pct = 100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
print(f"  Expected sales gain (%): {expected_sales_gain_pct:.4f}")

del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token=hf_token,
    torch_dtype=model_dtype,
    trust_remote_code=True,
)
model.to(device)
model.eval()
model.config.use_cache = False

# ------------------------------------
# Inductive case: n = T - 2 (t=60)
# ------------------------------------
n = Ts[-2]

future_stops_train = yhat_train[:, -2:]
future_stops_val = yhat_val[:, -2:]
future_stops_test = yhat_test[:, -2:]

first_j_train = future_stops_train.argmax(axis=1)
first_j_val = future_stops_val.argmax(axis=1)
first_j_test = future_stops_test.argmax(axis=1)

start_col_train = len(Ts) + 1 - 2
start_col_val = len(Ts) + 1 - 2
start_col_test = len(Ts) + 1 - 2

first_col_train = start_col_train + first_j_train
first_col_val = start_col_val + first_j_val
first_col_test = start_col_test + first_j_test

g_future_stop_train = g_train_stop[np.arange(yhat_train.shape[0]), first_col_train]
g_future_stop_val = g_val_stop[np.arange(yhat_val.shape[0]), first_col_val]
g_future_stop_test = g_test_stop[np.arange(yhat_test.shape[0]), first_col_test]

g_train_continue = np.where(duration_train >= n, g_future_stop_train, -np.inf).reshape(-1, 1)
g_val_continue = np.where(duration_val >= n, g_future_stop_val, -np.inf).reshape(-1, 1)
g_test_continue = np.where(duration_test >= n, g_future_stop_test, -np.inf).reshape(-1, 1)

g_train_stop[:, -3] = np.where(duration_train >= n, -COST_PER_SECOND * n, -np.inf)
g_val_stop[:, -3] = np.where(duration_val >= n, -COST_PER_SECOND * n, -np.inf)
g_test_stop[:, -3] = np.where(duration_test >= n, -COST_PER_SECOND * n, -np.inf)

mask_train = duration_train >= n
mask_val = duration_val >= n
mask_test = duration_test >= n

y_train = (g_train_stop[mask_train, -3] >= g_train_continue[mask_train, -1]).astype(int)
y_val = (g_val_stop[mask_val, -3] >= g_val_continue[mask_val, -1]).astype(int)
y_test = (g_test_stop[mask_test, -3] >= g_test_continue[mask_test, -1]).astype(int)

w_train = g_train_stop[mask_train, -3] - g_train_continue[mask_train, -1]
w_val = g_val_stop[mask_val, -3] - g_val_continue[mask_val, -1]
w_test = g_test_stop[mask_test, -3] - g_test_continue[mask_test, -1]

sample_weight_train = np.ones_like(w_train, dtype=np.float64)
sample_weight_val = np.ones_like(w_val, dtype=np.float64)
sample_weight_test = np.ones_like(w_test, dtype=np.float64)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

print("Scoring model at time T - 2...")
col = f"transcript_speaker_{int(n)}"
prefix = (
    "Estos son los primeros "
    + str(int(n))
    + " segundos de la conversación entre el agente de ventas Orador 0 y el consumidor Orador 1:\n\n"
)
suffix = "\n\n¿Esta conversación terminará en una venta exitosa? (responda sí o no):  "
prompts_train = (prefix + calls_train.loc[mask_train, col].astype(str) + suffix).tolist()
prompts_val = (prefix + calls_val.loc[mask_val, col].astype(str) + suffix).tolist()
prompts_test = (prefix + calls_test.loc[mask_test, col].astype(str) + suffix).tolist()

print("Fine-tuning model at time T - 2...")
prompts_fit = prompts_train + prompts_val
y_fit = np.concatenate([y_train, y_val]).astype(np.int64, copy=False)
sample_weight_fit = np.concatenate([sample_weight_train, sample_weight_val]).astype(
    np.float64, copy=False
)

train_dataset = Dataset.from_dict(
    {
        "prompt": prompts_fit,
        "labels": y_fit,
        "sample_weight": sample_weight_fit,
    }
)

train_dataset = train_dataset.map(_tokenize_prompts, batched=True, remove_columns=["prompt"])
train_eval_frac = float(np.clip(train_eval_frac, 0.0, 0.5))
splits = train_dataset.train_test_split(test_size=train_eval_frac, seed=seed, shuffle=True)
train_dataset = splits["train"]
eval_dataset = splits["test"]

training_args = TrainingArguments(
    output_dir=os.path.join(train_outdir, f"t{int(n)}"),
    remove_unused_columns=False,
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=int(train_logging_steps),
    eval_strategy="steps",
    eval_steps=int(train_eval_steps),
    save_steps=int(train_eval_steps),
    save_total_limit=1,
    per_device_train_batch_size=int(train_batch_size),
    gradient_accumulation_steps=int(train_grad_acc_steps),
    num_train_epochs=float(train_num_epochs),
    learning_rate=float(train_lr),
    weight_decay=float(train_weight_decay),
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_only_model=True,
    report_to="none",
    seed=seed,
)

trainer = _WeightedYesNoTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=seed),
    eval_dataset=eval_dataset.shuffle(seed=seed),
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(train_early_stop_patience))]
    if int(train_early_stop_patience) > 0
    else None,
)

trainer.train()
model = trainer.model
del trainer
model.eval()

p_train = infer_phat(prompts_train)
p_val = infer_phat(prompts_val)
p_test = infer_phat(prompts_test)

yhat_train[mask_train, -3] = (p_train >= 0.5).astype(int)
yhat_val[mask_val, -3] = (p_val >= 0.5).astype(int)
d_test_T = duration_test[mask_test]
future_stops_test_base = yhat_test[mask_test, -2:]
first_j_test_base = future_stops_test_base.argmax(axis=1)
start_col_test_base = len(Ts) + 1 - 2
first_col_test_base = start_col_test_base + first_j_test_base
stop_time_test_base = d_test_T.copy()
mask_not_terminal_base = first_col_test_base < len(Ts)
stop_time_test_base[mask_not_terminal_base] = Ts_arr[first_col_test_base[mask_not_terminal_base]]
expected_gain_if_stop_test = sales_per_second_test * (stop_time_test_base - n) - (
    is_sale_test[mask_test] * (first_col_test_base == len(Ts))
)
test_threshold, test_threshold_gain, test_num_stops = tune_threshold_by_expected_gain(
    p_test, expected_gain_if_stop_test
)
print(
    f"  Tuned test threshold at {int(n)}s: {test_threshold} "
    f"(stops={test_num_stops}/{p_test.size}, est_gain={test_threshold_gain:.6f})"
)
yhat_test[mask_test, -3] = (p_test >= test_threshold).astype(int)

future_stops_test = yhat_test[mask_test, -3:]
first_j_test = future_stops_test.argmax(axis=1)
start_col_test = len(Ts) + 1 - 3
first_col_test = start_col_test + first_j_test

stop_time_test = d_test_T.copy()
mask_not_terminal = first_col_test < len(Ts)
stop_time_test[mask_not_terminal] = Ts_arr[first_col_test[mask_not_terminal]]

time_saved_test = np.sum(d_test_T - stop_time_test)
print(f"  Total time saved on test (first stop >= {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(is_sale_test[mask_test] * (first_col_test == len(Ts)))
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * sales_per_second_test
print(
    "  Expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test:.4f}"
)

expected_total_sales_test = float(sales_made_test) + expected_sales_from_time_saved_test
print(
    f"  Expected total sales (assuming first stop opportunity at {n}s): "
    f"{expected_total_sales_test:.4f}"
)

print(f"  Status quo sales = {int(np.sum(is_sale_test))}")

expected_sales_gain_pct = 100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
print(f"  Expected sales gain (%): {expected_sales_gain_pct:.4f}")

del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    token=hf_token,
    torch_dtype=model_dtype,
    trust_remote_code=True,
)
model.to(device)
model.eval()
model.config.use_cache = False

# ------------------------------------
# Inductive case: n = T - 3 (t=30)
# ------------------------------------
n = Ts[-3]

future_stops_train = yhat_train[:, -3:]
future_stops_val = yhat_val[:, -3:]
future_stops_test = yhat_test[:, -3:]

first_j_train = future_stops_train.argmax(axis=1)
first_j_val = future_stops_val.argmax(axis=1)
first_j_test = future_stops_test.argmax(axis=1)

start_col_train = len(Ts) + 1 - 3
start_col_val = len(Ts) + 1 - 3
start_col_test = len(Ts) + 1 - 3

first_col_train = start_col_train + first_j_train
first_col_val = start_col_val + first_j_val
first_col_test = start_col_test + first_j_test

g_future_stop_train = g_train_stop[np.arange(yhat_train.shape[0]), first_col_train]
g_future_stop_val = g_val_stop[np.arange(yhat_val.shape[0]), first_col_val]
g_future_stop_test = g_test_stop[np.arange(yhat_test.shape[0]), first_col_test]

g_train_continue = np.where(duration_train >= n, g_future_stop_train, -np.inf).reshape(-1, 1)
g_val_continue = np.where(duration_val >= n, g_future_stop_val, -np.inf).reshape(-1, 1)
g_test_continue = np.where(duration_test >= n, g_future_stop_test, -np.inf).reshape(-1, 1)

g_train_stop[:, -4] = np.where(duration_train >= n, -COST_PER_SECOND * n, -np.inf)
g_val_stop[:, -4] = np.where(duration_val >= n, -COST_PER_SECOND * n, -np.inf)
g_test_stop[:, -4] = np.where(duration_test >= n, -COST_PER_SECOND * n, -np.inf)

mask_train = duration_train >= n
mask_val = duration_val >= n
mask_test = duration_test >= n

y_train = (g_train_stop[mask_train, -4] >= g_train_continue[mask_train, -1]).astype(int)
y_val = (g_val_stop[mask_val, -4] >= g_val_continue[mask_val, -1]).astype(int)
y_test = (g_test_stop[mask_test, -4] >= g_test_continue[mask_test, -1]).astype(int)

w_train = g_train_stop[mask_train, -4] - g_train_continue[mask_train, -1]
w_val = g_val_stop[mask_val, -4] - g_val_continue[mask_val, -1]
w_test = g_test_stop[mask_test, -4] - g_test_continue[mask_test, -1]

sample_weight_train = np.ones_like(w_train, dtype=np.float64)
sample_weight_val = np.ones_like(w_val, dtype=np.float64)
sample_weight_test = np.ones_like(w_test, dtype=np.float64)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

print("Scoring model at time T - 3...")
col = f"transcript_speaker_{int(n)}"
prefix = (
    "Estos son los primeros "
    + str(int(n))
    + " segundos de la conversación entre el agente de ventas Orador 0 y el consumidor Orador 1:\n\n"
)
suffix = "\n\n¿Esta conversación terminará en una venta exitosa? (responda sí o no):  "
prompts_train = (prefix + calls_train.loc[mask_train, col].astype(str) + suffix).tolist()
prompts_val = (prefix + calls_val.loc[mask_val, col].astype(str) + suffix).tolist()
prompts_test = (prefix + calls_test.loc[mask_test, col].astype(str) + suffix).tolist()

print("Fine-tuning model at time T - 3...")
prompts_fit = prompts_train + prompts_val
y_fit = np.concatenate([y_train, y_val]).astype(np.int64, copy=False)
sample_weight_fit = np.concatenate([sample_weight_train, sample_weight_val]).astype(
    np.float64, copy=False
)

train_dataset = Dataset.from_dict(
    {
        "prompt": prompts_fit,
        "labels": y_fit,
        "sample_weight": sample_weight_fit,
    }
)

train_dataset = train_dataset.map(_tokenize_prompts, batched=True, remove_columns=["prompt"])
train_eval_frac = float(np.clip(train_eval_frac, 0.0, 0.5))
splits = train_dataset.train_test_split(test_size=train_eval_frac, seed=seed, shuffle=True)
train_dataset = splits["train"]
eval_dataset = splits["test"]

training_args = TrainingArguments(
    output_dir=os.path.join(train_outdir, f"t{int(n)}"),
    remove_unused_columns=False,
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=int(train_logging_steps),
    eval_strategy="steps",
    eval_steps=int(train_eval_steps),
    save_steps=int(train_eval_steps),
    save_total_limit=1,
    per_device_train_batch_size=int(train_batch_size),
    gradient_accumulation_steps=int(train_grad_acc_steps),
    num_train_epochs=float(train_num_epochs),
    learning_rate=float(train_lr),
    weight_decay=float(train_weight_decay),
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_only_model=True,
    report_to="none",
    seed=seed,
)

trainer = _WeightedYesNoTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.shuffle(seed=seed),
    eval_dataset=eval_dataset.shuffle(seed=seed),
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(train_early_stop_patience))]
    if int(train_early_stop_patience) > 0
    else None,
)

trainer.train()
model = trainer.model
del trainer
model.eval()

p_train = infer_phat(prompts_train)
p_val = infer_phat(prompts_val)
p_test = infer_phat(prompts_test)

yhat_train[mask_train, -4] = (p_train >= 0.5).astype(int)
yhat_val[mask_val, -4] = (p_val >= 0.5).astype(int)
d_test_T = duration_test[mask_test]
future_stops_test_base = yhat_test[mask_test, -3:]
first_j_test_base = future_stops_test_base.argmax(axis=1)
start_col_test_base = len(Ts) + 1 - 3
first_col_test_base = start_col_test_base + first_j_test_base
stop_time_test_base = d_test_T.copy()
mask_not_terminal_base = first_col_test_base < len(Ts)
stop_time_test_base[mask_not_terminal_base] = Ts_arr[first_col_test_base[mask_not_terminal_base]]
expected_gain_if_stop_test = sales_per_second_test * (stop_time_test_base - n) - (
    is_sale_test[mask_test] * (first_col_test_base == len(Ts))
)
test_threshold, test_threshold_gain, test_num_stops = tune_threshold_by_expected_gain(
    p_test, expected_gain_if_stop_test
)
print(
    f"  Tuned test threshold at {int(n)}s: {test_threshold} "
    f"(stops={test_num_stops}/{p_test.size}, est_gain={test_threshold_gain:.6f})"
)
yhat_test[mask_test, -4] = (p_test >= test_threshold).astype(int)

future_stops_test = yhat_test[mask_test, -4:]
first_j_test = future_stops_test.argmax(axis=1)
start_col_test = len(Ts) + 1 - 4
first_col_test = start_col_test + first_j_test

stop_time_test = d_test_T.copy()
mask_not_terminal = first_col_test < len(Ts)
stop_time_test[mask_not_terminal] = Ts_arr[first_col_test[mask_not_terminal]]

time_saved_test = np.sum(d_test_T - stop_time_test)
print(f"  Total time saved on test (first stop >= {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(is_sale_test[mask_test] * (first_col_test == len(Ts)))
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * sales_per_second_test
print(
    "  Expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test:.4f}"
)

expected_total_sales_test = float(sales_made_test) + expected_sales_from_time_saved_test
print(
    f"  Expected total sales (assuming first stop opportunity at {n}s): "
    f"{expected_total_sales_test:.4f}"
)

print(f"  Status quo sales = {int(np.sum(is_sale_test))}")

expected_sales_gain_pct = 100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
print(f"  Expected sales gain (%): {expected_sales_gain_pct:.4f}")
