# Lab 04 — Adapting a Pretrained LLM: Prompting, LoRA and QLoRA (IT Helpdesk Triage)

A port of the Fine Tuning lab from the course repo
([`Labs/Model_Development/Fine Tuning`](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Model_Development/Fine%20Tuning)).
That lab has one notebook each for zero-shot prompting, few-shot prompting and
LoRA/Q-LoRA, plus a graded assignment built on few-shot prompting. The upstream
applications are restaurant reviews and customer-service replies, so I used a
different one: **routing IT helpdesk tickets** to one of five queues.

## Objectives

The upstream lab is really one question: *given a pretrained model and a task, what are
the ways to adapt it, and what does each cost?* This lab answers it on one task, with
the same model and the same scoring for every method:

1. **Zero-shot** — describe the task in the prompt, no examples
2. **Few-shot** — paste `k` labelled tickets per category into the prompt
3. **Full fine-tuning** — update every transformer weight
4. **LoRA** — freeze the model, train a low-rank update per layer (written from scratch)
5. **QLoRA** — LoRA on top of a 4-bit NF4-quantized base (quantizer written from scratch)

and compares them on accuracy, trainable parameters, memory, size of what you ship,
and how much the model forgets about ordinary language.

`tests/test_methods.py` plays the role of upstream's `tests/test_methods`: it checks the
from-scratch pieces.

## Task and data

Tickets like *"My Outlook inbox isn't receiving new messages since this morning"* →
`email`. Categories: `account_access`, `network`, `hardware`, `software`, `email`.

200 tickets, **hand-written** (no download): 40 per category, with some deliberately
ambiguous ones (a VPN *login* problem vs. a VPN *timeout*). For each category the first 20
are the train pool, the last 20 the test set, so the test set is 100 balanced tickets that
are never used for examples or training. `data/make_tickets.py` regenerates
`data/tickets.csv`.

`data/general_sentences.txt` holds 20 plain English sentences. They're only used to
measure perplexity before and after fine-tuning, as a cheap check for forgetting.

## Model and read-out

`HuggingFaceTB/SmolLM2-135M-Instruct` (135M parameters), small enough to fine-tune on a laptop GPU.

Every prompt ends with `Category:`. The prediction is whichever of the five category names has
the highest next-token score at that point. Because all methods are scored this way, the
numbers are comparable; fine-tuning just trains exactly that read-out
(cross-entropy over the five scores).

## Layout

| file | what it is |
|---|---|
| `01_zero_and_few_shot.ipynb` | zero-shot and few-shot (k = 1, 2, 4 × 3 example picks) |
| `02_lora_and_qlora.ipynb` | quantization cost, then full FT / LoRA / QLoRA, shots sweep, rank sweep, adapter checks |
| `03_comparison.ipynb` | reads `results/*.json`, one table + plots + takeaways (trains nothing) |
| `lora.py` | `LoRALinear`, apply / merge / save / load adapters |
| `quant.py` | NF4 + int8 block-wise quantization, `QuantLinear` (dequantizes in `forward`) |
| `tickets.py` | data loading, prompts, balanced sampling |
| `model_utils.py` | model loading, scoring, metrics, perplexity |
| `finetune.py` | training loop, the three model setups, `run_experiment` |
| `tests/test_methods.py` | 16 unit tests on tiny random models (no weights needed) |
| `results/` | JSON results and the figures the notebooks write |

## LoRA and QLoRA in short

LoRA keeps a weight `W` frozen and learns `W' = W + (alpha/r)·B·A`, with `A: r×in`,
`B: out×r`. `B` starts at zero so training starts from the pretrained model. Here it is
applied to the `q/k/v/o` projections of all 30 layers: 921,600 trainable weights, 0.68% of
the model.

QLoRA stores the frozen weights in 4 bits: blocks of 64 values are scaled by their absmax and
snapped to the nearest of the 16 NF4 levels (two codes packed per byte). `forward` expands
them back to fp32 on every call, so memory is saved and compute is not. Only the LoRA
matrices are trained. Notebook 2 reports accuracy right after quantizing and before training.

## Running it

```bash
pip install -r requirements.txt
```

```bash
python -m pytest tests -v
```

The notebooks need the model weights (270 MB, downloaded from the Hugging Face hub on first
run) and are meant to run in order, 01 → 02 → 03. The code uses a CUDA GPU if torch sees one
and otherwise falls back to the CPU. The default Windows `torch` wheel is CPU-only; for an
NVIDIA GPU install the CUDA build as described at the top of `requirements.txt`.

Rough run times: on an RTX 4060 laptop GPU a 20-per-class LoRA run takes ~15 s (full
fine-tuning ~15 s, QLoRA ~20 s). On a 16-thread CPU a 3-epoch LoRA run at 10 per class took
~33 s, about 6× slower than the GPU, so on CPU expect the whole notebook 2 to take a good while.

**Nothing needs to be run to review the lab.** The notebooks are committed with their
outputs. Those outputs come from running the notebooks' cells in a plain Python process,
because the Jupyter kernel could not start in my environment (zmq failed to create its
socket context). The cell code is unchanged either way.

Notebook 2 appends every run to `results/finetune.json`, so if it is interrupted a re-run
picks up where it stopped. Saved adapters go to `artifacts/` (not committed).

## Results

Test accuracy on the 100 held-out tickets. Mean ± std over seeds (3 for most, 2 for the rank
sweep). Trained methods use 20 labelled tickets per category; few-shot uses 4 per category.

| method | accuracy | trainable weights | shipped | general-text perplexity |
|---|---|---|---|---|
| zero-shot | 0.440 | 0 | – | 32.3 |
| few-shot (k=4) | 0.453 ± 0.192 | 0 | – | 32.3 |
| full fine-tuning | 0.903 ± 0.006 | 106.2M | 425 MB | 32.6 |
| LoRA (r=8) | 0.900 ± 0.026 | 0.92M | 3.8 MB | 33.2 |
| QLoRA (NF4, r=8) | 0.937 ± 0.015 | 0.92M | 3.8 MB + 4-bit base | 38.9 |

What stood out:

- **Zero-shot is weak.** It answers `account_access` for 73 of the 100 tickets and never says
  `email`.
- **Few-shot helps unreliably.** Means of 46% / 63% / 45% for k = 1 / 2 / 4, with std of 0.10-0.19
  across which tickets are picked. Best and worst single runs are 78% and 28%. More examples did not
  help this small model, and k=4 is ~5× slower per ticket because the prompt is ~510 tokens.
- **Training on the examples beats pasting them in.** Even one ticket per category (5 total) with
  LoRA reaches 83%, higher than 20 tickets in the prompt.
- **LoRA ties full fine-tuning** at ~1/115 of the trainable weights and ~1/110 of the shipped size.
- **QLoRA recovered from a bad start.** 4-bit quantization alone drops accuracy from 44% to 23%
  (8-bit leaves it at 45%). Training the adapters brought it to 94%, the highest of the three.
  That gap over LoRA is 3-4 points on a 100-ticket test set with 3 seeds, so I would not
  read much into it.
- **The perplexity rise for QLoRA comes from quantization, not training:** the 4-bit model has the
  same 38.9 before any training. LoRA and full fine-tuning moved it by under a point, so there was no
  real forgetting at this scale.
- **Rank barely mattered** (r = 1…32 all 84-89% at 10 tickets per category).
- **The adapter is separable:** reloading the saved file reproduces the run, turning it off
  restores the zero-shot predictions exactly, and merging it into the weights changes no predictions.

Estimated training memory for the decoder blocks (weights + gradients + Adam state, no
activations): full fine-tuning ~1.7 GB, LoRA ~0.44 GB, QLoRA ~0.07 GB.

## Caveats

- One 135M model, one small hand-written dataset, 100 test tickets (one ticket = one point), 2-3 seeds.
  Differences of a few points are within noise.
- The learning rates (1e-5 full, 1e-3 LoRA/QLoRA) are common defaults. I did not tune them per
  method, though I did run a single 10-per-class sanity check on the test tickets before fixing them.
- All five methods are scored with the next-token read-out above. It suits fine-tuning, which trains
  that exact read-out, and probably makes few-shot look worse than it would with free-text generation.
- 4-bit quantization hurts much more on a 135M model than it would on a multi-billion-parameter one,
  so the 23% should not be read as typical QLoRA behaviour.
- The memory numbers are estimates from parameter counts, not measured peaks.
