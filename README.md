# Medical Visual Question Answering — BLIP-2 Zero-shot on VQA-RAD

> **Research question:** How accurately can a multimodal model trained on everyday images (COCO, LAION) answer medical questions without any fine-tuning?

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
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

This project evaluates **BLIP-2 OPT-2.7B** in a **pure zero-shot setting** on the **VQA-RAD** medical imaging benchmark — no fine-tuning, no adapters, no domain-specific training data. The goal is to quantify the model's practical ceiling as a clinical screening tool and characterise two failure modes that are fundamental to vision-language models applied in medicine:

| Phenomenon | Description |
|---|---|
| **Visual Shortcut** | Model answers Yes/No correctly by matching pixel patterns to question keywords — not by understanding anatomy |
| **Hallucination** | Model generates plausible-sounding text based on language probability, independent of the actual image content |
| **Semantic Misalignment** | Model uses everyday language ("a large white patch") instead of medical terminology ("lower-lobe consolidation"), losing clinical value even when semantically correct |

### Core Hypothesis
BLIP-2 zero-shot achieves **higher ROUGE-L on closed-ended (Yes/No) questions** than on open-ended (free-form) questions — not because it understands radiology, but because of Visual Shortcut pattern matching.

**Result summary:**

| Question type | ROUGE-L | Interpretation |
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
├── predictions_zeroshot.csv  # Per-sample predictions & ground truth
├── metrics_zeroshot.json     # Aggregated ROUGE-L / BERTScore / LLM-Judge
└── qualitative_results.png   # Grid of sample images with model outputs
```

---

## Architecture

```
Medical Image (X-ray / CT / MRI / Ultrasound)
        │
        ▼
┌─────────────────────────────┐
│  ViT-L/14  Image Encoder    │  ← frozen, pretrained on LAION
│  (patch embeddings)         │
└─────────────┬───────────────┘
              │  visual tokens
              ▼
┌─────────────────────────────┐
│  Q-Former  Bridge           │  ← 32 learnable query tokens
│  (cross-attention bottleneck│    bridge between vision & language
└─────────────┬───────────────┘
              │  compressed visual representation
              ▼
┌─────────────────────────────┐
│  OPT-2.7B  LLM Decoder      │  ← frozen, 8-bit quantised on GPU
│  + Clinical Question prompt │
└─────────────┬───────────────┘
              │
              ▼
        Answer: "yes" / "no" / free-form phrase
```

**Prompt template used in inference:**
```
You are a radiology expert.
Answer the question with one or two words only.
Question: {question}
Answer:
```

**Generation parameters:**
```python
max_new_tokens=5, num_beams=5, do_sample=False,
repetition_penalty=1.3, length_penalty=0.5, early_stopping=True
```

---

## Dataset

**VQA-RAD** — the standard benchmark for medical visual question answering.

| Property | Value |
|---|---|
| Source | HuggingFace `flaviagiammarino/vqa-rad` |
| Images | 315 radiology images (X-ray, CT, MRI) |
| QA pairs | 3,515 question-answer pairs |
| Languages | English |
| Split used | 200 test samples (random seed 42) |

**Question type breakdown:**

| Type | Description | Example |
|---|---|---|
| **Closed-ended** | Yes / No answer | "Is there pleural effusion?" |
| **Open-ended** | Free-form clinical answer | "Where is the abnormality located?" |

**Preprocessing:**
- Images converted to RGB (handles grayscale DICOM-sourced images)
- Answers lowercased and whitespace-normalised
- Samples with missing image or empty answer are dropped
- 90/10 train/test split if no official test split exists

---

## Metrics

Three complementary metrics are used, listed in increasing clinical relevance:

### 1. ROUGE-L
Measures the longest common subsequence overlap between prediction and ground truth. Fast and interpretable but penalises valid synonyms (e.g. "opacity" vs "consolidation").

```python
rouge_fn = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
```

### 2. BERTScore F1
Embeds both strings with a pretrained BERT and computes cosine similarity. More tolerant of paraphrase — better suited to the medical domain where multiple phrasings are clinically equivalent.

```python
bertscore_metric.compute(predictions=preds, references=gts, lang='en')
```

### 3. LLM-as-a-Judge (Claude API)
Scores each prediction 0–10 for **clinical correctness**, not just lexical overlap. Requires an `ANTHROPIC_API_KEY` (Kaggle Secret or environment variable).

```python
# Enabled when ANTHROPIC_API_KEY is set
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
```

> **Priority:** LLM-Judge ≈ BERTScore > ROUGE-L for medical evaluation. ROUGE-L is reported first for compatibility with prior literature.

---

## Key Findings

### Visual Shortcut Effect
BLIP-2's higher closed-ended accuracy stems from pixel-pattern matching rather than anatomical understanding. A large white region in the image correlates with keywords like "effusion" or "opacity", giving the appearance of clinical knowledge.

### Hallucination Profile
On open-ended questions, the OPT decoder generates responses based on language model probability — independent of the image. Predictions may be fluent and superficially plausible while being clinically incorrect.

### Error taxonomy (200-sample test set)

| Error type | Condition | Impact |
|---|---|---|
| Visual Shortcut (benign) | Closed, Yes/No correct | Inflated metrics — not true understanding |
| Semantic Misalignment | Open, ROUGE-L < 0.3 | Correct meaning, wrong terminology |
| Partial Match | Open, 0.3 ≤ ROUGE-L < 0.6 | Partially correct |
| Total Collapse | Closed, Yes/No wrong | Fully incorrect |

### Clinical implication
> BLIP-2 zero-shot is acceptable for **binary screening triage** (flagging obvious abnormalities) but **must not be used** for differential diagnosis, anatomical localisation, or any output that requires clinical specificity.

---

## Installation

### Requirements
- Python 3.8 or higher
- CUDA-capable GPU recommended (see VRAM guide below); CPU works but is slow

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
gradio
transformers
bitsandbytes
accelerate
pillow
sentencepiece
rouge-score
torch
```

For the notebook, additional packages are needed:
```bash
pip install datasets evaluate bert-score nltk anthropic tqdm pandas matplotlib
```

---

## Running the Demo App

`app.py` provides a Gradio web interface that mirrors the notebook's B1 zero-shot pipeline.

### Basic launch (auto-detects GPU)
```bash
python app.py
# → http://localhost:7860
```

### All launch options

| Flag | Description |
|---|---|
| `--port 7861` | Change port (default: 7860) |
| `--share` | Generate a public Gradio link (useful for Colab/remote) |
| `--cpu` | Force CPU inference (no GPU required, ~60 s/query) |
| `--small` | Force BLIP-vqa-base (~400 MB weights, 1 GB VRAM) |

```bash
python app.py --port 7861 --share   # public link on port 7861
python app.py --cpu                 # CPU-only machine
python app.py --small               # low-VRAM GPU (< 6 GB)
```

### Using the app

1. Upload an X-ray, CT, MRI, or ultrasound image.
2. Type a clinical question in English.
3. *(Optional)* Enter the ground truth answer to see a live ROUGE-L score.
4. Click **Predict**.

The **Details** panel shows the predicted answer, inference time, active model, and ROUGE-L score (if ground truth was provided).

**Sample questions built in:**
- "Is there any abnormality in this image?"
- "Which organ is shown in the image?"
- "Is there a fracture?"
- "What type of scan is this?"
- "Is there evidence of pneumonia?"

---

## Running the Notebook

The notebook `task2-medical-vqa.ipynb` is designed for **Kaggle** but also runs locally.

### On Kaggle
1. Upload the notebook to Kaggle.
2. Enable a GPU accelerator (P100 or T4).
3. Add your `ANTHROPIC_API_KEY` under **Settings → Secrets** to enable LLM-as-a-Judge.
4. Run all cells sequentially.

### Locally
```bash
jupyter notebook task2-medical-vqa.ipynb
```

Set environment variable for LLM Judge:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Notebook section map

| Section | Content |
|---|---|
| 0 | Library installation |
| 1 | Imports & reproducibility config (SEED=42) |
| 2 | VQA-RAD loading from HuggingFace |
| 3 | Preprocessing & train/val/test split |
| 4 | Exploratory data analysis (distributions, answer lengths) |
| 5 | Metric definitions (ROUGE-L, BERTScore, LLM-Judge) |
| 6 | BLIP-2 model loading (8-bit GPU) |
| 7 | Zero-shot inference loop over 200 test samples |
| 8 | Visual Shortcut vs Hallucination analysis |
| 9 | Qualitative output visualisation (image grid) |
| 10 | Error taxonomy & classification |
| 11 | BLIP-2 architecture diagram |
| 12 | Conclusions & improvement roadmap |
| 13 | Save predictions CSV + metrics JSON + zip outputs |

---

## Model Strategy & VRAM Guide

The app and notebook auto-select the best strategy based on available VRAM:

| VRAM available | Model loaded | Approx. size | CLI flag |
|---|---|---|---|
| ≥ 8 GB | BLIP-2 OPT-2.7B (8-bit GPU) | ~8 GB | *(auto)* |
| 1 – 6 GB | BLIP-vqa-base (fp16) | ~1 GB | `--small` |
| No GPU | BLIP-2 OPT-2.7B (CPU fp32) | ~11 GB RAM | `--cpu` |

**Inference speed (approximate):**

| Mode | Time per query |
|---|---|
| BLIP-2 8-bit GPU (T4 16 GB) | ~2–5 s |
| BLIP-vqa-base fp16 GPU | < 1 s |
| BLIP-2 CPU fp32 | ~60 s |

> `bitsandbytes` 8-bit quantisation is Linux-only. On Windows use `--small` or `--cpu`.

---

## Output Files

After running the notebook, the `outputs/` directory contains:

| File | Description |
|---|---|
| `predictions_zeroshot.csv` | Per-sample: question, ground truth, predicted answer, predicted explanation |
| `metrics_zeroshot.json` | Aggregated ROUGE-L, BERTScore F1, LLM-Judge score |
| `qualitative_results.png` | 2×3 grid of sample images annotated with predictions vs ground truth |
| `working_files.zip` | Archive of all outputs (Kaggle download) |

---

## Limitations & Future Work

### Known limitations
- **No fine-tuning:** Zero-shot BLIP-2 lacks medical domain knowledge; results reflect general vision-language capability, not radiology expertise.
- **Short answer constraint:** Prompting for 1–2 word answers degrades open-ended responses that require clinical detail.
- **English only:** VQA-RAD is English; model performance on other languages is untested.
- **200-sample cap:** Inference is limited to 200 test samples due to compute constraints; full-set results may differ.

### Recommended improvements (priority order)

1. **QLoRA fine-tuning on VQA-RAD** — Fine-tune only ~1–2 % of parameters via LoRA adapters. Reduces VRAM by 65 %, directly improves both closed and open-ended accuracy.

2. **Medical-specific few-shot prompting** — Add 3–5 radiology QA examples with images to the prompt to reduce Hallucination without retraining.

3. **Specialised medical models** — Replace BLIP-2 with [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) or [BioViL-T](https://huggingface.co/microsoft/BioViL-T), which are pretrained on clinical imaging data.

4. **Iterative decoding** — Apply contrastive decoding or nucleus sampling with temperature tuning to reduce repetitive or semantically empty outputs.

5. **Structured output parsing** — Post-process free-form outputs through a medical NER model to extract canonical clinical terms, improving ROUGE-L and BERTScore.

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