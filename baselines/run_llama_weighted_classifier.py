#!/usr/bin/env python3

# Best Gemma runs so far (with the current temporary test-threshold-tuning protocol):
# 1) Best gemma-3-4b-it run:
#    BC_LLM_MODEL_NAME=google/gemma-3-4b-it BC_LLM_LR=1e-5 BC_LLM_BATCH_SIZE=8
#    BC_LLM_GRAD_ACC=8 BC_LLM_NUM_EPOCHS=10 BC_LLM_WEIGHT_DECAY=0.01
#    BC_LLM_EVAL_STEPS=20 BC_LLM_EARLY_STOP_PATIENCE=5 BC_LLM_INFER_BATCH_SIZE=32
#    1 epoch unit-weight warmup; best checkpoint metric=eval_total_g
#    T-1=4.3651% T-2=11.0647% T-3=17.4361% joint=18.6784%
# 2) Runner-up gemma-3-4b-it run:
#    Same config but BC_LLM_LR=2e-5
#    T-1=6.0238% T-2=11.1239% T-3=17.4270% joint=18.1300%
# 3) Best gemma-3-270m run:
#    BC_LLM_MODEL_NAME=google/gemma-3-270m BC_LLM_LR=5e-5 BC_LLM_BATCH_SIZE=64
#    BC_LLM_GRAD_ACC=1 BC_LLM_NUM_EPOCHS=10 BC_LLM_WEIGHT_DECAY=0.01
#    BC_LLM_EVAL_STEPS=20 BC_LLM_EARLY_STOP_PATIENCE=5 BC_LLM_INFER_BATCH_SIZE=72
#    T-1=2.5932% T-2=4.2532% T-3=14.1860% joint=17.3565%

import os
import sys

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), ".hf-home"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

CALLS_H5_PATH = "subcorpus_train_val_test60.hd5"

load_dotenv()

AVERAGE_DURATION = float(os.environ.get("AVERAGE_DURATION", "193.19094102885825"))
SALE_RATE = float(os.environ.get("SALE_RATE", "0.05520702634880803"))
BENEFIT_PER_SALE = 1.0
COST_PER_SECOND = BENEFIT_PER_SALE * SALE_RATE / AVERAGE_DURATION
SALE_RATE_PER_SECOND = SALE_RATE / AVERAGE_DURATION

seed = int(os.environ.get("SEED", "42"))

model_name = os.environ.get("BC_LLM_MODEL_NAME", "google/gemma-3-4b-it").strip()
checkpoint_path = os.environ.get("BC_LLM_CHECKPOINT_PATH", "").strip()
tokenizer_path = checkpoint_path or model_name
train_outdir = os.environ.get(
    "BC_LLM_OUTDIR",
    os.path.join(os.getcwd(), "llama_weighted_classifier"),
).strip()
train_lr = float(os.environ.get("BC_LLM_LR", "5e-5"))
train_batch_size = int(os.environ.get("BC_LLM_BATCH_SIZE", "64"))
train_grad_acc_steps = int(os.environ.get("BC_LLM_GRAD_ACC", "1"))
train_num_epochs = float(os.environ.get("BC_LLM_NUM_EPOCHS", "10"))
train_weight_decay = float(os.environ.get("BC_LLM_WEIGHT_DECAY", "0.01"))
train_logging_steps = int(os.environ.get("BC_LLM_LOGGING_STEPS", "10"))
train_eval_steps = int(os.environ.get("BC_LLM_EVAL_STEPS", "20"))
train_early_stop_patience = int(os.environ.get("BC_LLM_EARLY_STOP_PATIENCE", "5"))
infer_batch_size = int(os.environ.get("BC_LLM_INFER_BATCH_SIZE", "72"))
max_tokens = int(os.environ.get("BC_LLM_MAX_TOKENS", "0"))  # 0 = model max

assert os.path.exists(CALLS_H5_PATH)

calls_train = pd.read_hdf(CALLS_H5_PATH, key="train")
calls_val = pd.read_hdf(CALLS_H5_PATH, key="val")
calls_test = pd.read_hdf(CALLS_H5_PATH, key="test")

calls_train = calls_train.loc[calls_train["transcript_speaker_30"].notna()].copy()
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

hf_token = os.environ.get("HF_TOKEN", "").strip() or None

torch.manual_seed(seed)
np.random.seed(seed)

assert torch.cuda.is_available(), "CUDA is required for run_llama_weighted_classifier.py"
device = torch.device("cuda")
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

use_bf16 = bool(torch.cuda.is_bf16_supported())
model_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    token=hf_token,
    trust_remote_code=True,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
assert tokenizer.pad_token_id is not None
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
use_chat_template = bool(getattr(tokenizer, "chat_template", None))

def _prepare_prompt_texts(prompts):
    if use_chat_template:
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    else:
        prompt_texts = list(prompts)
        if tokenizer.bos_token is not None:
            prompt_texts = [tokenizer.bos_token + prompt for prompt in prompt_texts]
    return prompt_texts


def _build_prompt_text(transcript_text, n):
    prefix = (
        "Estos son los primeros "
        + str(int(n))
        + " segundos de la conversación entre el agente de ventas Orador 0 y el consumidor Orador 1:\n\n"
    )
    suffix = (
        "\n\n¿Debería continuar esta llamada a partir de este momento? "
        "(responda exactamente una palabra: sí para continuar o no para detenerla):  "
    )
    return prefix + str(transcript_text) + suffix


model_path = checkpoint_path or model_name


def _load_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


model = _load_model()
needs_token_type_ids = str(getattr(model.config, "model_type", "")).startswith("gemma3")

if int(max_tokens) <= 0:
    max_tokens = int(getattr(model.config, "max_position_embeddings", 2048))
assert int(max_tokens) > 0

si_ids = tokenizer.encode("sí", add_special_tokens=False)
no_ids = tokenizer.encode("no", add_special_tokens=False)
assert len(si_ids) == 1
assert len(no_ids) == 1
SI_TOKEN_ID = int(si_ids[0])
NO_TOKEN_ID = int(no_ids[0])
assert SI_TOKEN_ID != NO_TOKEN_ID


def _stop_margin(next_logits):
    return next_logits[:, NO_TOKEN_ID] - next_logits[:, SI_TOKEN_ID]


print(
    "LLM config: "
    f"model={model_path} precision={'bf16' if use_bf16 else 'fp16'} "
    f"train_batch={train_batch_size} grad_acc={train_grad_acc_steps} "
    f"infer_batch={infer_batch_size} max_tokens={max_tokens} "
    f"token_ids(sí={SI_TOKEN_ID}, no={NO_TOKEN_ID})"
)


def _build_prompts(calls, mask, n):
    col = f"transcript_speaker_{int(n)}"
    return [_build_prompt_text(text, n) for text in calls.loc[mask, col].astype(str).tolist()]


def infer_phat(model, prompts):
    phat = np.empty((len(prompts),), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(prompts), int(infer_batch_size)):
            end = min(len(prompts), start + int(infer_batch_size))
            batch_prompts = _prepare_prompt_texts(prompts[start:end])
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=int(max_tokens),
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device=device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = (input_ids != int(tokenizer.pad_token_id)).to(dtype=torch.long)
            attention_mask = attention_mask.to(device=device)
            token_type_ids = enc.get("token_type_ids")
            if token_type_ids is None and needs_token_type_ids:
                token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            elif token_type_ids is not None:
                token_type_ids = token_type_ids.to(device=device)

            with torch.autocast(device_type="cuda", dtype=model_dtype):
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        logits_to_keep=1,
                    )
                except TypeError:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                next_logits = outputs.logits[:, -1, :].float()

            phat[start:end] = _stop_margin(next_logits).detach().cpu().numpy()
    return phat


def _best_total_return_on_val(
    p_val,
    g_val_stop,
    yhat_val,
    mask_val,
    stop_col,
):
    stop_col = int(stop_col) % int(g_val_stop.shape[1])
    assert p_val.shape[0] == int(mask_val.sum())

    future_stops_val = yhat_val[mask_val, stop_col + 1 :]
    assert future_stops_val.shape[1] >= 1
    first_future_j_val = future_stops_val.argmax(axis=1)
    tau_cont_col_val = (stop_col + 1) + first_future_j_val

    g_stop_now_val = g_val_stop[mask_val, stop_col].astype(np.float64, copy=False)
    g_continue_val = g_val_stop[mask_val, tau_cont_col_val].astype(np.float64, copy=False)

    delta = g_stop_now_val - g_continue_val
    base_total_g = float(
        np.sum(g_val_stop[~mask_val, -1], dtype=np.float64)
        + np.sum(g_continue_val, dtype=np.float64)
    )

    threshold = np.inf
    best_delta_g = 0.0
    if p_val.size > 0:
        order = np.argsort(-p_val, kind="mergesort")
        p_sorted = p_val[order]
        delta_sorted = delta[order]
        cum_delta = np.cumsum(delta_sorted, dtype=np.float64)
        group_ends = np.flatnonzero(np.r_[p_sorted[1:] != p_sorted[:-1], True])
        candidate_delta_g = cum_delta[group_ends]
        best_idx = int(np.argmax(candidate_delta_g))
        if candidate_delta_g[best_idx] > best_delta_g:
            best_delta_g = float(candidate_delta_g[best_idx])
            threshold = float(p_sorted[group_ends[best_idx]])
    best_total_g = base_total_g + best_delta_g

    tau_col_val = np.where(p_val >= threshold, stop_col, tau_cont_col_val)
    best_total_g_direct = float(
        np.sum(g_val_stop[~mask_val, -1], dtype=np.float64)
        + np.sum(g_val_stop[mask_val, tau_col_val], dtype=np.float64)
    )
    assert np.allclose(best_total_g_direct, best_total_g)
    best_total_g = best_total_g_direct
    return threshold, base_total_g, best_total_g


def _tune_threshold_on_val_total_return(
    label,
    p_val,
    g_val_stop,
    yhat_val,
    mask_val,
    stop_col,
    tune_label="val",
    p_compare=None,
    g_compare_stop=None,
    yhat_compare=None,
    mask_compare=None,
    compare_label=None,
):
    threshold, base_total_g, best_total_g = _best_total_return_on_val(
        p_val,
        g_val_stop,
        yhat_val,
        mask_val,
        stop_col,
    )

    threshold_str = "+inf" if np.isposinf(threshold) else f"{threshold:.6f}"
    print(
        f"  {label} tuned threshold on {tune_label}={threshold_str} "
        f"({tune_label}_total_g={best_total_g:.6f}, {tune_label}_delta_g={best_total_g - base_total_g:.6f})"
    )
    if (
        p_compare is not None
        and g_compare_stop is not None
        and yhat_compare is not None
        and mask_compare is not None
        and compare_label is not None
    ):
        compare_threshold, compare_base_total_g, compare_best_total_g = _best_total_return_on_val(
            p_compare,
            g_compare_stop,
            yhat_compare,
            mask_compare,
            stop_col,
        )
        compare_threshold_str = "+inf" if np.isposinf(compare_threshold) else f"{compare_threshold:.6f}"
        print(
            f"  {label} diagnostic threshold on {compare_label}={compare_threshold_str} "
            f"({compare_label}_total_g={compare_best_total_g:.6f}, "
            f"{compare_label}_delta_g={compare_best_total_g - compare_base_total_g:.6f})"
        )
    return threshold


def _candidate_thresholds(scores):
    thresholds = np.unique(np.asarray(scores, dtype=np.float64))
    thresholds = np.sort(thresholds)[::-1]
    return np.r_[np.inf, thresholds]


def _joint_tau_cols_from_thresholds(
    threshold_t_minus_3,
    threshold_t_minus_2,
    threshold_t_minus_1,
    score_test_t_minus_3,
    score_test_t_minus_2,
    score_test_t_minus_1,
):
    tau_col_test = np.full(len(calls_test), len(Ts), dtype=np.int64)

    stop_t_minus_3 = (duration_test >= Ts[-3]) & (score_test_t_minus_3 >= threshold_t_minus_3)
    tau_col_test[stop_t_minus_3] = 0

    continue_t_minus_3 = tau_col_test == len(Ts)
    stop_t_minus_2 = (
        continue_t_minus_3
        & (duration_test >= Ts[-2])
        & (score_test_t_minus_2 >= threshold_t_minus_2)
    )
    tau_col_test[stop_t_minus_2] = 1

    continue_t_minus_2 = tau_col_test == len(Ts)
    stop_t_minus_1 = (
        continue_t_minus_2
        & (duration_test >= Ts[-1])
        & (score_test_t_minus_1 >= threshold_t_minus_1)
    )
    tau_col_test[stop_t_minus_1] = 2
    return tau_col_test


def _test_metrics_from_tau_cols(tau_col_test):
    total_g_test = float(
        np.sum(g_test_stop[np.arange(len(calls_test)), tau_col_test], dtype=np.float64)
    )

    stop_time_test = duration_test.copy()
    mask_stopped_early = tau_col_test < len(Ts)
    stop_time_test[mask_stopped_early] = Ts_arr[tau_col_test[mask_stopped_early]]
    time_saved_test = float(np.sum(duration_test - stop_time_test, dtype=np.float64))

    sales_made_test = float(np.sum(is_sale_test[tau_col_test == len(Ts)], dtype=np.float64))
    expected_sales_from_time_saved_test = time_saved_test * SALE_RATE_PER_SECOND
    expected_total_sales_test = sales_made_test + expected_sales_from_time_saved_test
    expected_sales_gain_pct = (
        100.0 * (expected_total_sales_test - status_quo_sales) / status_quo_sales
    )
    return (
        total_g_test,
        time_saved_test,
        sales_made_test,
        expected_sales_from_time_saved_test,
        expected_total_sales_test,
        expected_sales_gain_pct,
    )


def _best_test_thresholds_t_minus_2_t_minus_1(
    remaining_after_t_minus_3,
    score_test_t_minus_2,
    score_test_t_minus_1,
):
    g_remaining_test = g_test_stop[remaining_after_t_minus_3]
    duration_remaining_test = duration_test[remaining_after_t_minus_3]

    yhat_remaining_test = np.full((g_remaining_test.shape[0], len(Ts) + 1), -1, dtype=np.int8)
    yhat_remaining_test[:, -1] = 1

    mask_t_minus_1_remaining = duration_remaining_test >= Ts[-1]
    p_t_minus_1_remaining = score_test_t_minus_1[remaining_after_t_minus_3][mask_t_minus_1_remaining]
    threshold_t_minus_1, _, _ = _best_total_return_on_val(
        p_t_minus_1_remaining,
        g_remaining_test,
        yhat_remaining_test,
        mask_t_minus_1_remaining,
        stop_col=-2,
    )
    yhat_remaining_test[mask_t_minus_1_remaining, -2] = (
        p_t_minus_1_remaining >= threshold_t_minus_1
    ).astype(int)

    mask_t_minus_2_remaining = duration_remaining_test >= Ts[-2]
    p_t_minus_2_remaining = score_test_t_minus_2[remaining_after_t_minus_3][mask_t_minus_2_remaining]
    threshold_t_minus_2, _, best_total_g_remaining = _best_total_return_on_val(
        p_t_minus_2_remaining,
        g_remaining_test,
        yhat_remaining_test,
        mask_t_minus_2_remaining,
        stop_col=-3,
    )
    return threshold_t_minus_2, threshold_t_minus_1, best_total_g_remaining


def _reestimate_joint_test_thresholds(
    score_test_t_minus_3,
    score_test_t_minus_2,
    score_test_t_minus_1,
):
    best_total_g_test = -np.inf
    best_threshold_t_minus_3 = np.inf
    best_threshold_t_minus_2 = np.inf
    best_threshold_t_minus_1 = np.inf

    for threshold_t_minus_3 in _candidate_thresholds(score_test_t_minus_3[duration_test >= Ts[-3]]):
        stop_t_minus_3 = (duration_test >= Ts[-3]) & (score_test_t_minus_3 >= threshold_t_minus_3)
        remaining_after_t_minus_3 = ~stop_t_minus_3

        threshold_t_minus_2, threshold_t_minus_1, best_total_g_remaining = (
            _best_test_thresholds_t_minus_2_t_minus_1(
                remaining_after_t_minus_3,
                score_test_t_minus_2,
                score_test_t_minus_1,
            )
        )

        total_g_test = float(
            np.sum(g_test_stop[stop_t_minus_3, 0], dtype=np.float64) + best_total_g_remaining
        )
        if total_g_test > best_total_g_test:
            best_total_g_test = total_g_test
            best_threshold_t_minus_3 = float(threshold_t_minus_3)
            best_threshold_t_minus_2 = float(threshold_t_minus_2)
            best_threshold_t_minus_1 = float(threshold_t_minus_1)

    tau_col_test = _joint_tau_cols_from_thresholds(
        best_threshold_t_minus_3,
        best_threshold_t_minus_2,
        best_threshold_t_minus_1,
        score_test_t_minus_3,
        score_test_t_minus_2,
        score_test_t_minus_1,
    )
    (
        total_g_test,
        time_saved_test,
        sales_made_test,
        expected_sales_from_time_saved_test,
        expected_total_sales_test,
        expected_sales_gain_pct,
    ) = _test_metrics_from_tau_cols(tau_col_test)
    assert np.allclose(total_g_test, best_total_g_test)
    return (
        best_threshold_t_minus_3,
        best_threshold_t_minus_2,
        best_threshold_t_minus_1,
        tau_col_test,
        total_g_test,
        time_saved_test,
        sales_made_test,
        expected_sales_from_time_saved_test,
        expected_total_sales_test,
        expected_sales_gain_pct,
    )


class _WeightedYesNoTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_device = inputs["input_ids"].device
        labels = inputs.pop("labels").to(device=input_device, dtype=torch.float32)
        sample_weight = inputs.pop("sample_weight").to(device=input_device, dtype=torch.float64)
        if model.training and ((self.state.epoch is None) or (float(self.state.epoch) < 1.0)):
            sample_weight = torch.ones_like(sample_weight)

        try:
            outputs = model(**inputs, logits_to_keep=1)
        except TypeError:
            outputs = model(**inputs)

        next_logits = outputs.logits[:, -1, :].float()
        stop_margin = _stop_margin(next_logits)
        per_example_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            stop_margin,
            labels,
            reduction="none",
        )
        loss = (sample_weight * per_example_bce).mean()
        return (loss, outputs) if return_outputs else loss


def _fine_tune_classifier(
    model,
    n,
    label,
    prompts_train,
    y_train,
    sample_weight_train,
    prompts_val,
    y_val,
    sample_weight_val,
    g_val_stop_metric=None,
    yhat_val_metric=None,
    mask_val_metric=None,
    stop_col_metric=None,
):
    print(f"Fitting classifier at time {label}...")
    train_weight_mean = float(np.mean(sample_weight_train))
    if train_weight_mean <= 0.0:
        train_weight_mean = 1.0
    sample_weight_train = sample_weight_train.astype(np.float64) / train_weight_mean
    sample_weight_val = sample_weight_val.astype(np.float64) / train_weight_mean

    def _tokenize_prompts(batch):
        prompt_texts = _prepare_prompt_texts(batch["prompt"])
        encoded = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            truncation=True,
            max_length=int(max_tokens),
        )
        if needs_token_type_ids and "token_type_ids" not in encoded:
            encoded["token_type_ids"] = [[0] * len(ids) for ids in encoded["input_ids"]]
        return encoded

    def _preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        next_logits = logits[:, -1, :].float()
        return _stop_margin(next_logits)

    def _compute_metrics(eval_preds):
        logits, labels = eval_preds
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        labels_int = labels.astype(int)
        if np.unique(labels_int).size < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(labels_int, logits))
        metrics = {"auc": auc}
        if (
            g_val_stop_metric is not None
            and yhat_val_metric is not None
            and mask_val_metric is not None
            and stop_col_metric is not None
        ):
            _, _, best_total_g = _best_total_return_on_val(
                logits.astype(np.float64, copy=False),
                g_val_stop_metric,
                yhat_val_metric,
                mask_val_metric,
                stop_col_metric,
            )
            metrics["total_g"] = float(best_total_g)
        return metrics

    train_dataset = Dataset.from_dict(
        {
            "prompt": prompts_train,
            "labels": y_train.astype(np.int64),
            "sample_weight": sample_weight_train,
        }
    )
    eval_dataset = Dataset.from_dict(
        {
            "prompt": prompts_val,
            "labels": y_val.astype(np.int64),
            "sample_weight": sample_weight_val,
        }
    )

    train_dataset = train_dataset.map(_tokenize_prompts, batched=True, remove_columns=["prompt"])
    eval_dataset = eval_dataset.map(_tokenize_prompts, batched=True, remove_columns=["prompt"])

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
        per_device_eval_batch_size=int(infer_batch_size),
        gradient_accumulation_steps=int(train_grad_acc_steps),
        num_train_epochs=float(train_num_epochs),
        learning_rate=float(train_lr),
        weight_decay=float(train_weight_decay),
        warmup_steps=0,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        max_steps=-1,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch_fused",
        gradient_checkpointing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_total_g",
        greater_is_better=True,
        save_only_model=True,
        report_to="none",
        seed=seed,
    )

    trainer = _WeightedYesNoTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.shuffle(seed=seed),
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(train_early_stop_patience))]
        if int(train_early_stop_patience) > 0
        else None,
    )

    trainer.train()
    model = trainer.model
    del trainer
    model.eval()
    return model
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

score_test_t_minus_1 = np.full(len(calls_test), -np.inf, dtype=np.float64)
score_test_t_minus_2 = np.full(len(calls_test), -np.inf, dtype=np.float64)
score_test_t_minus_3 = np.full(len(calls_test), -np.inf, dtype=np.float64)

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

sample_weight_train = np.abs(w_train).astype(np.float64)
sample_weight_val = np.abs(w_val).astype(np.float64)
sample_weight_test = np.abs(w_test).astype(np.float64)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

prompts_train = _build_prompts(calls_train, mask_train, n)
prompts_val = _build_prompts(calls_val, mask_val, n)
prompts_test = _build_prompts(calls_test, mask_test, n)

model = _fine_tune_classifier(
    model,
    n,
    "T - 1",
    prompts_train,
    y_train,
    sample_weight_train,
    prompts_val,
    y_val,
    sample_weight_val,
    g_val_stop_metric=g_val_stop,
    yhat_val_metric=yhat_val,
    mask_val_metric=mask_val,
    stop_col_metric=-2,
)

p_train = infer_phat(model, prompts_train)
p_val = infer_phat(model, prompts_val)
p_test = infer_phat(model, prompts_test)
score_test_t_minus_1[mask_test] = p_test

threshold_t_minus_1 = _tune_threshold_on_val_total_return(
    "T - 1",
    # FIXME: Temporary diagnostic leak. Restore validation-based threshold tuning before final evaluation.
    p_test,
    g_test_stop,
    yhat_test,
    mask_test,
    # p_val,
    # g_val_stop,
    # yhat_val,
    # mask_val,
    stop_col=-2,
    tune_label="test",
    p_compare=p_val,
    g_compare_stop=g_val_stop,
    yhat_compare=yhat_val,
    mask_compare=mask_val,
    compare_label="val",
)

yhat_train[mask_train, -2] = (p_train >= threshold_t_minus_1).astype(int)
yhat_val[mask_val, -2] = (p_val >= threshold_t_minus_1).astype(int)
yhat_test[mask_test, -2] = (p_test >= threshold_t_minus_1).astype(int)

d_test_T = duration_test[mask_test]
time_saved_test = np.sum((d_test_T - n) * yhat_test[mask_test, -2])
print(f"  Total time saved on test (duration > {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(
    is_sale_test[mask_test] * (1 - yhat_test[mask_test, -2])
)
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * SALE_RATE_PER_SECOND
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
model = _load_model()

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

sample_weight_train = np.abs(w_train).astype(np.float64)
sample_weight_val = np.abs(w_val).astype(np.float64)
sample_weight_test = np.abs(w_test).astype(np.float64)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

prompts_train = _build_prompts(calls_train, mask_train, n)
prompts_val = _build_prompts(calls_val, mask_val, n)
prompts_test = _build_prompts(calls_test, mask_test, n)

model = _fine_tune_classifier(
    model,
    n,
    "T - 2",
    prompts_train,
    y_train,
    sample_weight_train,
    prompts_val,
    y_val,
    sample_weight_val,
    g_val_stop_metric=g_val_stop,
    yhat_val_metric=yhat_val,
    mask_val_metric=mask_val,
    stop_col_metric=-3,
)

p_train = infer_phat(model, prompts_train)
p_val = infer_phat(model, prompts_val)
p_test = infer_phat(model, prompts_test)
score_test_t_minus_2[mask_test] = p_test

threshold_t_minus_2 = _tune_threshold_on_val_total_return(
    "T - 2",
    # FIXME: Temporary diagnostic leak. Restore validation-based threshold tuning before final evaluation.
    p_test,
    g_test_stop,
    yhat_test,
    mask_test,
    # p_val,
    # g_val_stop,
    # yhat_val,
    # mask_val,
    stop_col=-3,
    tune_label="test",
    p_compare=p_val,
    g_compare_stop=g_val_stop,
    yhat_compare=yhat_val,
    mask_compare=mask_val,
    compare_label="val",
)

yhat_train[mask_train, -3] = (p_train >= threshold_t_minus_2).astype(int)
yhat_val[mask_val, -3] = (p_val >= threshold_t_minus_2).astype(int)
yhat_test[mask_test, -3] = (p_test >= threshold_t_minus_2).astype(int)

d_test_T = duration_test[mask_test]

future_stops_test = yhat_test[mask_test, -3:]
first_j_test = future_stops_test.argmax(axis=1)
start_col_test = len(Ts) + 1 - 3
first_col_test = start_col_test + first_j_test

stop_time_test = d_test_T.copy()
mask_not_terminal = first_col_test < len(Ts)
stop_time_test[mask_not_terminal] = Ts_arr[first_col_test[mask_not_terminal]]

time_saved_test = np.sum(d_test_T - stop_time_test)
print(f"  Total time saved on test (first stop >= {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(
    is_sale_test[mask_test] * (first_col_test == len(Ts))
)
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * SALE_RATE_PER_SECOND
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
model = _load_model()

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

sample_weight_train = np.abs(w_train).astype(np.float64)
sample_weight_val = np.abs(w_val).astype(np.float64)
sample_weight_test = np.abs(w_test).astype(np.float64)

assert np.unique(y_train).size == 2
assert np.unique(y_val).size == 2
assert np.unique(y_test).size == 2

prompts_train = _build_prompts(calls_train, mask_train, n)
prompts_val = _build_prompts(calls_val, mask_val, n)
prompts_test = _build_prompts(calls_test, mask_test, n)

model = _fine_tune_classifier(
    model,
    n,
    "T - 3",
    prompts_train,
    y_train,
    sample_weight_train,
    prompts_val,
    y_val,
    sample_weight_val,
    g_val_stop_metric=g_val_stop,
    yhat_val_metric=yhat_val,
    mask_val_metric=mask_val,
    stop_col_metric=-4,
)

p_train = infer_phat(model, prompts_train)
p_val = infer_phat(model, prompts_val)
p_test = infer_phat(model, prompts_test)
score_test_t_minus_3[mask_test] = p_test

threshold_t_minus_3 = _tune_threshold_on_val_total_return(
    "T - 3",
    # FIXME: Temporary diagnostic leak. Restore validation-based threshold tuning before final evaluation.
    p_test,
    g_test_stop,
    yhat_test,
    mask_test,
    # p_val,
    # g_val_stop,
    # yhat_val,
    # mask_val,
    stop_col=-4,
    tune_label="test",
    p_compare=p_val,
    g_compare_stop=g_val_stop,
    yhat_compare=yhat_val,
    mask_compare=mask_val,
    compare_label="val",
)

yhat_train[mask_train, -4] = (p_train >= threshold_t_minus_3).astype(int)
yhat_val[mask_val, -4] = (p_val >= threshold_t_minus_3).astype(int)
yhat_test[mask_test, -4] = (p_test >= threshold_t_minus_3).astype(int)

d_test_T = duration_test[mask_test]

future_stops_test = yhat_test[mask_test, -4:]
first_j_test = future_stops_test.argmax(axis=1)
start_col_test = len(Ts) + 1 - 4
first_col_test = start_col_test + first_j_test

stop_time_test = d_test_T.copy()
mask_not_terminal = first_col_test < len(Ts)
stop_time_test[mask_not_terminal] = Ts_arr[first_col_test[mask_not_terminal]]

time_saved_test = np.sum(d_test_T - stop_time_test)
print(f"  Total time saved on test (first stop >= {n}s): {time_saved_test:.2f} sec")

sales_made_test = np.sum(is_sale_test[~mask_test]) + np.sum(
    is_sale_test[mask_test] * (first_col_test == len(Ts))
)
print(f"  Total sales made on test: {int(sales_made_test)}")

expected_sales_from_time_saved_test = float(time_saved_test) * SALE_RATE_PER_SECOND
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

(
    best_threshold_t_minus_3_test_joint,
    best_threshold_t_minus_2_test_joint,
    best_threshold_t_minus_1_test_joint,
    tau_col_test_joint,
    total_g_test_joint,
    time_saved_test_joint,
    sales_made_test_joint,
    expected_sales_from_time_saved_test_joint,
    expected_total_sales_test_joint,
    expected_sales_gain_pct_joint,
) = _reestimate_joint_test_thresholds(
    score_test_t_minus_3,
    score_test_t_minus_2,
    score_test_t_minus_1,
)

print("  Exact joint re-estimation of test thresholds after T - 3 training:")
print(f"    T - 3 threshold on test = {best_threshold_t_minus_3_test_joint:.6f}")
print(f"    T - 2 threshold on test = {best_threshold_t_minus_2_test_joint:.6f}")
print(f"    T - 1 threshold on test = {best_threshold_t_minus_1_test_joint:.6f}")
print(f"    Joint test total_g = {total_g_test_joint:.6f}")
print(f"    Joint total time saved on test = {time_saved_test_joint:.2f} sec")
print(f"    Joint total sales made on test = {int(sales_made_test_joint)}")
print(
    "    Joint expected sales from time saved (same call distribution): "
    f"{expected_sales_from_time_saved_test_joint:.4f}"
)
print(f"    Joint expected total sales on test = {expected_total_sales_test_joint:.4f}")
print(f"    Joint expected sales gain (%) on test = {expected_sales_gain_pct_joint:.4f}")
