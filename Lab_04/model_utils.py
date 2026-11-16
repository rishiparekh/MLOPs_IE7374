"""Loading the model, scoring tickets, and measuring things.

Every method in the lab (zero-shot, few-shot, full fine-tuning, LoRA, QLoRA)
reads out the answer the same way: take the hidden state at the last prompt
token and compare the next-token scores of the 5 category names. Only the
way the model gets adapted differs, so the numbers are comparable.
"""
import math
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from tickets import LABELS

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

torch.set_num_threads(8)

# use the GPU if torch can see one, otherwise fall back to the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer


def label_token_ids(tokenizer):
    """First token of ' <label>' for each category. They must all differ,
    otherwise we couldn't tell the categories apart from one next-token score."""
    ids = [tokenizer.encode(" " + lab)[0] for lab in LABELS]
    assert len(set(ids)) == len(ids), "label names share a first token"
    return ids


def label_logits(model, tokenizer, prompts, label_ids, batch_size=8):
    """Returns a (len(prompts), 5) tensor of scores. Keeps the graph, so it can
    be used for training as well as evaluation."""
    out = []
    # the output embedding rows of our 5 labels (tied with the input embeddings)
    head = model.lm_head.weight[label_ids]  # (5, hidden)
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, padding_side="right").to(DEVICE)
        hidden = model.model(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"]).last_hidden_state
        last = enc["attention_mask"].sum(dim=1) - 1  # index of last real token
        h = hidden[torch.arange(len(batch), device=DEVICE), last]
        out.append(h @ head.T)
    return torch.cat(out)


@torch.no_grad()
def predict(model, tokenizer, prompts, label_ids, batch_size=8):
    model.eval()
    return label_logits(model, tokenizer, prompts, label_ids, batch_size).argmax(dim=1).tolist()


def evaluate(model, tokenizer, prompts, true_labels, label_ids, batch_size=8):
    """true_labels are category strings. Returns a dict of metrics."""
    y_true = [LABELS.index(l) for l in true_labels]
    start = time.time()
    y_pred = predict(model, tokenizer, prompts, label_ids, batch_size)
    secs = time.time() - start
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "sec_per_ticket": secs / len(prompts),
        "preds": y_pred,
    }


@torch.no_grad()
def general_perplexity(model, tokenizer, sentences):
    """Perplexity on plain English. If it shoots up after fine-tuning, the model
    is forgetting how to do ordinary language modelling."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        loss = F.cross_entropy(logits[0, :-1], enc["input_ids"][0, 1:], reduction="sum")
        total_loss += loss.item()
        total_tokens += enc["input_ids"].shape[1] - 1
    return math.exp(total_loss / total_tokens)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def prompt_tokens(tokenizer, prompt):
    return len(tokenizer.encode(prompt))
