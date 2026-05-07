# Medical Visual Question Answering — BLIP-2 Zero-Shot on VQA-RAD

> **Research question:** How accurately can a multimodal model trained on everyday images (COCO, LAION) answer medical questions without any fine-tuning?

[![Python 3.12](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Dataset](https://img.shields.io/badge/dataset-VQA--RAD-green)](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
[![Model](https://img.shields.io/badge/model-BLIP--2%20OPT--2.7B-orange)](https://huggingface.co/Salesforce/blip2-opt-2.7b)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Metrics](#metrics)
6. [Key Findings](#key-findings)
7. [Installation](#installation)
8. [Running the Demo App](#running-the-demo-app)
9. [Running the Notebook](#running-the-notebook)
10. [Model Strategy & VRAM Guide](#model-strategy--vram-guide)
11. [Output Files](#output-files)
12. [Limitations & Future Work](#limitations--future-work)

---

## Overview

This project evaluates **BLIP-2 OPT-2.7B** in a **pure zero-shot setting** on the **VQA-RAD** medical imaging benchmark — no fine-tuning, no adapters, no domain-specific training data. The goal is to quantify the model's practical ceiling as a clinical screening tool and characterise three fundamental failure modes:

| Phenomenon | Description |
|---|---|
| **Visual Shortcut** | Model answers Yes/No correctly by matching pixel patterns to question keywords — not by understanding anatomy |
| **Hallucination** | Model generates plausible-sounding text based on language probability, independent of the actual image content |
| **Semantic Misalignment** | Model uses everyday language ("a large white patch") instead of medical terminology ("lower-lobe consolidation"), losing clinical value even when semantically correct |

### Core Hypothesis
BLIP-2 zero-shot achieves **higher ROUGE-L on closed-ended (Yes/No) questions** than on open-ended questions — not because it understands radiology, but because of Visual Shortcut pattern matching.

### Result Summary

| Question Type | ROUGE-L | Interpretation |
|---|---|---|
| Closed-ended (Yes / No) | **0.4750** | Inflated by Visual Shortcut |
| Open-ended (free-form) | **0.1603** | Depressed by Hallucination |
| Gap | **+0.3147** | Confirms the hypothesis |

---

## Project Structure

```
.
├── task2-medical-vqa.ipynb   # Full experiment notebook (Kaggle / local)
├── app.py                    # Gradio web demo (localhost)
├── requirements.txt          # Python dependencies
└── README.md                 # This file

outputs/                      # Auto-created at runtime
├── predictions_zeroshot.csv  # Per-sample: question, ground truth, prediction
├── metrics_zeroshot.json     # Aggregated ROUGE-L / BERTScore / LLM-Judge
├── eda.png                   # Dataset distribution plots
├── visual_shortcut_vs_hallucination.png
├── qualitative_results.png   # Image grid annotated with model outputs
├── architecture_blip2_zeroshot.png
└── working_files.zip         # Full output archive (Kaggle download)
```

---

## Architecture

```
Medical Image (X-ray / CT / MRI / Ultrasound)
              │
              ▼
┌──────────────────────────────┐
│   ViT-L/14  Image Encoder   │  ← frozen, pretrained on LAION
│   (patch embeddings)        │
└─────────────┬────────────────┘
              │  visual tokens
              ▼
┌──────────────────────────────┐
│   Q-Former  Bridge          │  ← 32 learnable query tokens
│   (cross-attention)         │    bridges vision → language
└─────────────┬────────────────┘
              │  compressed visual representation
              ▼
┌──────────────────────────────┐
│   OPT-2.7B  LLM Decoder     │  ← frozen (fp16 or 8-bit on GPU)
│   + Question prompt         │
└─────────────┬────────────────┘
              │
              ▼
        Answer: "yes" / "no" / free-form phrase
```

**Prompt template** (used in both the notebook and `app.py`):
```
Question: {question} Answer:
```

**Generation parameters:**
```python
max_new_tokens=80
num_beams=1       # greedy decoding
do_sample=False
```

> ⚠️ OPT-2.7B is **not** instruction-tuned. Adding a system prompt (e.g. "You are a radiologist…") changes the output distribution and degrades performance. The bare prompt above is intentional.

---

## Dataset

**VQA-RAD** — the standard benchmark for medical visual question answering.

| Property | Value |
|---|---|
| Source | HuggingFace `flaviagiammarino/vqa-rad` |
| Images | 315 radiology images (X-ray, CT, MRI) |
| QA pairs | 3,515 question-answer pairs |
| Language | English |
| Test samples used | 200 (random seed 42) |

**Question types:**

| Type | Description | Example |
|---|---|---|
| **Closed-ended** | Yes / No answer | "Is there pleural effusion?" |
| **Open-ended** | Free-form clinical answer | "Where is the abnormality located?" |

**Preprocessing:**
- Images converted to RGB (handles grayscale DICOM-sourced images)
- Answers lowercased and whitespace-normalised
- Samples with a missing image or empty answer are dropped
- 90/10 train/val split applied if no official test split exists

---

## Metrics

Three complementary metrics are used, listed in increasing clinical relevance:

### 1. ROUGE-L
Measures the longest common subsequence overlap between prediction and ground truth. Fast and interpretable but penalises valid synonyms ("opacity" vs "consolidation").

### 2. BERTScore F1
Embeds both strings with a pretrained RoBERTa-large model and computes cosine similarity. More tolerant of paraphrase — better suited to the medical domain where multiple phrasings are clinically equivalent.

### 3. LLM-as-a-Judge (Claude API)
Scores each prediction 0–10 for **clinical correctness**, not just lexical overlap. Requires an `ANTHROPIC_API_KEY`.

```python
# Scores are normalised to 0.0–1.0 in the code
# Evaluated on the first 30 test samples to control API cost
```

> **Priority order for evaluation:** LLM-Judge ≈ BERTScore > ROUGE-L. ROUGE-L is reported first for compatibility with prior literature.

---

## Key Findings

### Visual Shortcut Effect
BLIP-2's higher closed-ended accuracy comes from matching low-level pixel patterns to question keywords rather than genuine anatomical understanding.

### Hallucination Profile
On open-ended questions, the OPT decoder generates responses based on language probability — independent of the image. Predictions may be fluent and superficially plausible while being clinically incorrect.

### Error Taxonomy (200-sample test set)

| Error Type | Condition | Impact |
|---|---|---|
| **Visual Shortcut** | Closed, Yes/No correct | Inflated metrics — not true understanding |
| **Semantic Misalignment** | Open, ROUGE-L < 0.30 | Correct intent, wrong terminology |
| **Partial Match** | Open, 0.30 ≤ ROUGE-L < 0.60 | Partially correct |
| **Total Collapse** | Closed, Yes/No wrong | Fully incorrect |

### Clinical Implication
> BLIP-2 zero-shot is acceptable for **binary screening triage** (flagging obvious abnormalities) but **must not be used** for differential diagnosis, anatomical localisation, or any output requiring clinical specificity.

---

## Installation

### Requirements
- **Python 3.12.3**
- CUDA-capable GPU recommended (see VRAM guide); CPU works but is slow (~60 s/query)
- Linux recommended for 8-bit quantisation via `bitsandbytes`; Windows users should use `--small` or `--cpu`

### Step 1 — Install PyTorch

**GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch
```

### Step 2 — Install all other dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Set Anthropic API key for LLM-as-a-Judge
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Linux / macOS
set ANTHROPIC_API_KEY=sk-ant-...      # Windows CMD
```
On Kaggle: **Settings → Secrets → Add secret** → name: `ANTHROPIC_API_KEY`

---

## Running the Demo App

`app.py` provides a Gradio web interface that is **exactly synchronised** with the notebook's B1 zero-shot pipeline (same prompt, same generation parameters).

### Quick start
```bash
python app.py
# → open http://localhost:7860
```

### All launch flags

| Flag | Description |
|---|---|
| `--port 7861` | Change port (default: 7860) |
| `--share` | Generate a public Gradio link (72 h, useful for Colab / remote) |
| `--cpu` | Force CPU inference (no GPU required, ~60 s/query) |
| `--small` | Force BLIP-vqa-base (~1 GB VRAM, faster, lower accuracy) |

```bash
python app.py --port 7861 --share   # public link
python app.py --cpu                 # CPU-only machine
python app.py --small               # GPU with < 6 GB VRAM
```

### How to use the app

1. Upload an X-ray, CT, MRI, or ultrasound image
2. Type a clinical question in English
3. *(Optional)* Enter the ground truth answer → live ROUGE-L score appears
4. Click **Run Inference** or press **Shift+Enter**

The **Details** panel shows: predicted answer · inference time · active model · ROUGE-L (if ground truth provided).

---

## Running the Notebook

`task2-medical-vqa.ipynb` is designed for **Kaggle** (GPU T4 / P100) but also runs locally.

### On Kaggle
1. Upload `task2-medical-vqa.ipynb` to Kaggle Notebooks
2. Set accelerator: **Settings → Accelerator → GPU T4 x2** (or P100)
3. Enable internet: **Settings → Internet → On**
4. Add API key: **Settings → Secrets → Add secret** → name: `ANTHROPIC_API_KEY`
5. **Run All**

### Locally
```bash
pip install jupyter
jupyter notebook task2-medical-vqa.ipynb
```

### Notebook Section Map

| Section | Title | What it does |
|---|---|---|
| 0 | Install Dependencies | `pip install` all packages |
| 1 | Imports & Configuration | Imports, reproducibility seed (42), device detection, paths |
| 2 | Load VQA-RAD Dataset | Download via HuggingFace `datasets` |
| 3 | Preprocessing & Data Splits | Normalise samples, 90/10 train/val, filter nulls |
| 4 | Exploratory Data Analysis | Distribution plots: question types, answer categories, lengths |
| 5 | Evaluation Metrics | Define ROUGE-L, BERTScore, LLM-Judge functions |
| 6 | BLIP-2 Zero-Shot Inference | Load model (`AutoProcessor` + `Blip2ForConditionalGeneration`) |
| 7 | Visual Shortcut vs Hallucination | Run inference on 200 samples, compute & compare metrics |
| 8 | Qualitative Analysis | Print and visualise 8 representative predictions |
| 9 | Error Analysis | Classify all errors into 4 categories |
| 10 | Architecture Diagram | Matplotlib diagram of the BLIP-2 pipeline |
| 11 | Conclusions | Result table + explanation of each failure mode + roadmap |
| 12 | Save Results | Export `predictions_zeroshot.csv`, `metrics_zeroshot.json`, zip |
| 13 | Gradio Demo | Interactive web UI inside the notebook (public link via `share=True`) |

---

## Model Strategy & VRAM Guide

Both `app.py` and the notebook auto-select the best strategy based on available VRAM:

| VRAM | Strategy | Model | Notebook-compatible |
|---|---|---|---|
| ≥ 10 GB | `blip2-fp16` | BLIP-2 OPT-2.7B fp16 | ✅ Identical outputs |
| 6–10 GB | `blip2-8bit` | BLIP-2 OPT-2.7B 8-bit | ⚠️ Slight logit difference |
| 1–6 GB | `small` | BLIP-vqa-base fp16 | ❌ Different model |
| No GPU | `blip2-cpu` | BLIP-2 OPT-2.7B fp32 | ✅ Same outputs, slow |

**Approximate inference speed:**

| Mode | Time per query |
|---|---|
| BLIP-2 fp16 GPU (A100 40 GB) | ~1–2 s |
| BLIP-2 8-bit GPU (T4 16 GB) | ~2–5 s |
| BLIP-vqa-base fp16 GPU | < 1 s |
| BLIP-2 CPU fp32 | ~60 s |

> **Windows users:** `bitsandbytes` 8-bit quantisation is Linux-only. Use `--small` or `--cpu` on Windows.

---

## Output Files

| File | Description |
|---|---|
| `predictions_zeroshot.csv` | Per-sample: question, ground truth, predicted answer, predicted explanation |
| `metrics_zeroshot.json` | Aggregated ROUGE-L, BERTScore F1, LLM-Judge score |
| `eda.png` | 4-panel EDA: question types, closed/open split, answer categories, answer lengths |
| `visual_shortcut_vs_hallucination.png` | Bar chart comparing closed vs open ROUGE-L |
| `qualitative_results.png` | 2×3 grid of images annotated with predictions vs ground truth |
| `architecture_blip2_zeroshot.png` | BLIP-2 pipeline diagram |
| `working_files.zip` | Archive of all outputs (Kaggle download) |

---

## Limitations & Future Work

### Known Limitations
- **No fine-tuning:** Zero-shot results reflect general vision-language capability, not radiology expertise
- **English only:** VQA-RAD is English; multilingual performance is untested
- **200-sample cap:** Inference is limited to 200 test samples due to compute constraints

### Recommended Improvements (priority order)

1. **QLoRA fine-tuning on VQA-RAD** — Update only ~1–2% of parameters via LoRA adapters; reduces VRAM usage by ~65% and directly improves both question types

2. **Medical-specific few-shot prompting** — Add 3–5 radiology QA examples to the prompt to reduce Hallucination without retraining

3. **Specialised medical models** — Replace BLIP-2 with [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) or [BioViL-T](https://huggingface.co/microsoft/BioViL-T), pretrained on clinical imaging data (PubMed / MIMIC-CXR)

4. **RLHF with clinical reward** — Use BERTScore or physician ratings as the reward signal so the model optimises for correctness rather than fluency

5. **Structured output parsing** — Post-process free-form outputs through a medical NER model to extract canonical clinical terms, improving all lexical metrics

---

## Citation

```bibtex
@dataset{vqarad,
  title  = {VQA-RAD: A Dataset for Visually Aligned Question Answering in Radiology},
  author = {Lau, Jason J. and others},
  year   = {2018}
}

@article{blip2,
  title   = {BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models},
  author  = {Li, Junnan and others},
  journal = {ICML},
  year    = {2023}
}
```