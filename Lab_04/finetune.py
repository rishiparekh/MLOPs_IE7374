"""Training loop and the three ways of setting up the model for it."""
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import lora
import quant
from model_utils import DEVICE, label_logits, load_model
from tickets import LABELS, sample_per_class, zero_shot_prompt

ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_full_ft():
    """Full fine-tuning: every transformer weight is trainable.
    Embeddings stay frozen (they're also the output head here, and it keeps
    Adam's memory in check on a laptop)."""
    model, tokenizer = load_model()
    for name, p in model.named_parameters():
        p.requires_grad = "embed_tokens" not in name
    return model, tokenizer


def build_lora(r=8, alpha=16, dropout=0.05):
    model, tokenizer = load_model()
    for p in model.parameters():
        p.requires_grad = False
    lora.apply_lora(model, ATTN_TARGETS, r=r, alpha=alpha, dropout=dropout)
    lora.freeze_all_but_lora(model)
    return model, tokenizer


def build_qlora(r=8, alpha=16, dropout=0.05, bits=4, block_size=64):
    """Quantize every linear layer of the decoder blocks, then put LoRA on the
    attention projections."""
    model, tokenizer = load_model()
    for p in model.parameters():
        p.requires_grad = False
    quant.quantize_linears(model.model.layers, bits=bits, block_size=block_size)
    lora.apply_lora(model, ATTN_TARGETS, r=r, alpha=alpha, dropout=dropout)
    lora.freeze_all_but_lora(model)
    return model, tokenizer


def train(model, tokenizer, prompts, labels, label_ids, epochs=5, lr=1e-3,
          batch_size=8, seed=0, log=True):
    """Cross-entropy over the 5 category scores. Returns the loss per epoch."""
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    y = torch.tensor([LABELS.index(l) for l in labels], device=DEVICE)

    history = []
    start = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(prompts)).tolist()
        epoch_loss = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            logits = label_logits(model, tokenizer, [prompts[j] for j in idx], label_ids,
                                  batch_size=len(idx))
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        history.append(epoch_loss / len(order))
        if log:
            print(f"  epoch {epoch + 1}/{epochs}  loss {history[-1]:.4f}  ({time.time() - start:.0f}s)")
    model.eval()
    return history


# ---------------------------------------------------------------------------
# one full experiment: build -> (evaluate) -> train -> evaluate -> measure
# ---------------------------------------------------------------------------
DEFAULT_LR = {"full": 1e-5, "lora": 1e-3, "qlora": 1e-3}
ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def run_experiment(method, train_df, test_df, general_sentences, n_per_class, seed,
                   r=8, epochs=5, lr=None, bits=4):
    """method is 'full', 'lora' or 'qlora'. Returns one flat dict of results."""
    from model_utils import (count_params, evaluate, general_perplexity,
                             label_token_ids)

    lr = lr or DEFAULT_LR[method]
    set_seed(seed)
    if method == "full":
        model, tokenizer = build_full_ft()
    elif method == "lora":
        model, tokenizer = build_lora(r=r, alpha=2 * r)
    else:
        model, tokenizer = build_qlora(r=r, alpha=2 * r, bits=bits)
    label_ids = label_token_ids(tokenizer)

    shots = sample_per_class(train_df, n_per_class, seed)
    train_prompts = [zero_shot_prompt(t) for t in shots["text"]]
    test_prompts = [zero_shot_prompt(t) for t in test_df["text"]]
    test_labels = test_df["label"].tolist()

    result = {"method": method, "n_per_class": n_per_class, "seed": seed, "r": r,
              "lr": lr, "epochs": epochs}
    if method == "qlora":
        result["bits"] = bits
        # the quantized model before any training: shows what quantization costs
        before = evaluate(model, tokenizer, test_prompts, test_labels, label_ids)
        result["acc_after_quant_before_train"] = before["accuracy"]

    trainable, total = count_params(model)
    start = time.time()
    history = train(model, tokenizer, train_prompts, shots["label"].tolist(), label_ids,
                    epochs=epochs, lr=lr, seed=seed, log=False)
    result["train_sec"] = time.time() - start
    result["loss_history"] = history

    metrics = evaluate(model, tokenizer, test_prompts, test_labels, label_ids)
    result.update(accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                  sec_per_ticket=metrics["sec_per_ticket"], preds=metrics["preds"])
    result["general_ppl"] = general_perplexity(model, tokenizer, general_sentences)
    result["trainable_params"] = trainable
    result["total_params"] = total

    # what you would actually have to store to ship this fine-tune
    if method == "full":
        result["checkpoint_bytes"] = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)
    else:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        path = ARTIFACT_DIR / f"{method}_r{r}_n{n_per_class}_s{seed}.pt"
        torch.save(lora.lora_state_dict(model), path)
        result["checkpoint_bytes"] = path.stat().st_size
    result["base_weight_bytes"] = quant.tensor_bytes(model.model.layers)
    del model
    return result
