"""Data loading and prompt building for the helpdesk ticket task."""
import random
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

LABELS = ["account_access", "network", "hardware", "software", "email"]

INSTRUCTION = (
    "You are an IT helpdesk assistant. Read the support ticket and answer with "
    "exactly one category.\n"
    "Categories: " + ", ".join(LABELS) + "\n\n"
)


def load_tickets():
    df = pd.read_csv(DATA_DIR / "tickets.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test


def load_general_sentences():
    """Plain English sentences, used only to watch the model's general ability."""
    text = (DATA_DIR / "general_sentences.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def sample_per_class(train, n_per_class, seed):
    """Pick n tickets from each category (the same pick is used by few-shot
    prompting and by fine-tuning, so the comparison is fair)."""
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        idx = list(train.index[train["label"] == label])
        rows += rng.sample(idx, n_per_class)
    rng.shuffle(rows)  # mix the classes so the prompt isn't grouped by label
    return train.loc[rows].reset_index(drop=True)


def zero_shot_prompt(text):
    return INSTRUCTION + f"Ticket: {text}\nCategory:"


def few_shot_prompt(text, examples):
    """examples is a dataframe with 'text' and 'label' columns."""
    shots = ""
    for _, row in examples.iterrows():
        shots += f"Ticket: {row['text']}\nCategory: {row['label']}\n\n"
    return INSTRUCTION + shots + f"Ticket: {text}\nCategory:"
