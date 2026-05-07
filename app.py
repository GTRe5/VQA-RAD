"""
app.py - Medical VQA Demo (English) — localhost
================================================
Synchronized with: task2-medical-vqa.ipynb (B1 zero-shot only)

Install:
  pip install gradio transformers bitsandbytes accelerate pillow sentencepiece rouge-score

Run:
  python app.py                  # auto-detect GPU VRAM, pick best strategy
  python app.py --port 7861
  python app.py --share          # public Gradio link
  python app.py --cpu            # force CPU (works on any machine, slower)
  python app.py --small          # force small model (BLIP-vqa-base, ~400 MB, fits 4 GB GPU)

VRAM requirements:
  BLIP-2 OPT-2.7B  fp16   -> ~10 GB  (notebook-compatible, identical outputs)
  BLIP-2 OPT-2.7B  8-bit  ->  ~8 GB  (slightly different logits, saves VRAM)
  BLIP-vqa-base    fp16   ->  ~1 GB  (separate smaller model, faster)
  CPU fallback             ->   0 GB  (slow ~60s/query)
"""

import argparse
import time

import gradio as gr
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    BitsAndBytesConfig,
)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Medical VQA Demo — B1 Zero-shot")
parser.add_argument("--port",  type=int,        default=7860)
parser.add_argument("--share", action="store_true", help="Create public Gradio link")
parser.add_argument("--cpu",   action="store_true", help="Force CPU (no GPU needed)")
parser.add_argument("--small", action="store_true", help="Force BLIP-vqa-base (~1 GB VRAM)")
args = parser.parse_args()

# ── VRAM detection ────────────────────────────────────────────────────────────
USE_GPU = torch.cuda.is_available() and not args.cpu

if USE_GPU:
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU 0 : {torch.cuda.get_device_name(0)}  ({vram_gb:.1f} GB VRAM)")
else:
    vram_gb = 0.0
    print("[INFO] No GPU — running on CPU")

# Strategy selection:
#   < 6 GB VRAM  -> small BLIP model (fits comfortably)
#   6-10 GB      -> BLIP-2 8-bit quantised (saves ~2 GB vs fp16)
#   >= 10 GB     -> BLIP-2 fp16 (notebook-identical outputs)
#   no GPU       -> BLIP-2 CPU float32
USE_SMALL = args.small or (USE_GPU and vram_gb < 6.0)

if USE_SMALL:
    STRATEGY = "small"
elif USE_GPU and vram_gb < 10.0:
    STRATEGY = "blip2-8bit"
elif USE_GPU:
    STRATEGY = "blip2-fp16"
else:
    STRATEGY = "blip2-cpu"

print(f"[INFO] Strategy : {STRATEGY}")

# ── Model IDs ─────────────────────────────────────────────────────────────────
BLIP2_ID = "Salesforce/blip2-opt-2.7b"   # matches notebook MODEL_ID
SMALL_ID = "Salesforce/blip-vqa-base"    # ~400 MB, ~1 GB VRAM

# ── Load model ────────────────────────────────────────────────────────────────
if STRATEGY == "small":
    print(f"[INFO] Loading {SMALL_ID} ...")
    processor = BlipProcessor.from_pretrained(SMALL_ID)
    model = BlipForQuestionAnswering.from_pretrained(
        SMALL_ID,
        torch_dtype=torch.float16 if USE_GPU else torch.float32,
    ).to("cuda:0" if USE_GPU else "cpu").eval()
    MODEL_LABEL = "BLIP-vqa-base (fp16)"

elif STRATEGY == "blip2-8bit":
    # 8-bit saves ~2 GB VRAM; logits differ slightly from notebook fp16
    print(f"[INFO] Loading {BLIP2_ID} (8-bit quantised) ...")
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    processor = AutoProcessor.from_pretrained(BLIP2_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    MODEL_LABEL = "BLIP-2 OPT-2.7B (8-bit GPU)"

elif STRATEGY == "blip2-fp16":
    # fp16 — produces outputs identical to the notebook
    print(f"[INFO] Loading {BLIP2_ID} (fp16, notebook-compatible) ...")
    processor = AutoProcessor.from_pretrained(BLIP2_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID,
        torch_dtype=torch.float16,
    ).to("cuda:0").eval()
    MODEL_LABEL = "BLIP-2 OPT-2.7B (fp16 GPU)"

else:  # blip2-cpu
    print(f"[INFO] Loading {BLIP2_ID} (CPU float32, slow) ...")
    processor = AutoProcessor.from_pretrained(BLIP2_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID,
        torch_dtype=torch.float32,
    ).eval()
    MODEL_LABEL = "BLIP-2 OPT-2.7B (CPU)"

print(f"[OK]   {MODEL_LABEL} ready.")

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pil_image: Image.Image, question: str) -> tuple[str, float]:
    """
    Run inference and return (answer, elapsed_seconds).

    Prompt and generation params exactly match notebook Section 6 (Cell 17):
      - Prompt  : "Question: {question} Answer:"
      - Greedy  : num_beams=1, do_sample=False
      - Length  : max_new_tokens=80
    """
    img = pil_image.convert("RGB")
    t0  = time.time()

    if STRATEGY == "small":
        # BLIP-vqa-base has its own simpler API
        inputs = processor(images=img, text=question, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        pred = processor.decode(out[0], skip_special_tokens=True).strip()

    else:
        # ── Exactly matches notebook Cell 17 ─────────────────────────
        prompt = f"Question: {question} Answer:"
        inputs = processor(images=img, text=prompt, return_tensors="pt")

        target = "cpu" if STRATEGY == "blip2-cpu" else "cuda:0"
        inputs = {k: v.to(target) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,   # same as notebook
                num_beams=1,         # greedy — same as notebook
                do_sample=False,     # same as notebook
            )

        pred = processor.decode(out[0], skip_special_tokens=True)
        if "Answer:" in pred:
            pred = pred.split("Answer:")[-1]
        pred = pred.strip()

    return pred, time.time() - t0


# ── Gradio callback ───────────────────────────────────────────────────────────
def run_demo(
    image,
    question: str,
    ground_truth: str,
) -> tuple[str, str]:
    if image is None:
        return "Please upload a medical image.", ""
    if not question.strip():
        return "Please enter a question.", ""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    answer, elapsed = predict(image, question.strip())

    lines = [
        f"Question  : {question.strip()}",
        f"Answer    : {answer}",
        f"Inference : {elapsed:.2f}s",
        f"Model     : {MODEL_LABEL}",
    ]

    # Optional ROUGE-L when ground truth is provided
    if ground_truth.strip():
        from rouge_score import rouge_scorer as rouge_lib
        rouge_fn = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        rL = rouge_fn.score(
            ground_truth.strip().lower(),
            answer.lower(),
        )["rougeL"].fmeasure
        lines.append(f"ROUGE-L   : {rL:.4f}  (vs ground truth \"{ground_truth.strip()}\")")

    return answer, "\n".join(lines)


# ── UI ────────────────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "Is there any abnormality in this image?",
    "Which organ is shown in the image?",
    "Is there a fracture?",
    "What type of scan is this?",
    "Is there evidence of pneumonia?",
    "What is the orientation of this image?",
    "Which side is affected?",
    "Is there a tumor present?",
    "Is this a normal or abnormal finding?",
    "What body part is depicted?",
    "Is there pleural effusion?",
    "Where is the abnormality located?",
]

STRATEGY_NOTE = {
    "small"     : "BLIP-vqa-base — GPU VRAM < 6 GB detected, using small model",
    "blip2-8bit": "BLIP-2 OPT-2.7B — 8-bit quantised GPU (6-10 GB VRAM)",
    "blip2-fp16": "BLIP-2 OPT-2.7B — fp16 GPU ≥ 10 GB (notebook-identical outputs)",
    "blip2-cpu" : "BLIP-2 OPT-2.7B — CPU mode (slow, ~60s per query)",
}

CSS = """
#title    { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #64748b; font-size: 0.88em; margin-bottom: 4px; }
#strategy { text-align: center; font-size: 0.85em; margin-bottom: 12px; }
"""

with gr.Blocks(title="Medical VQA", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML(f"""
        <h1 id='title'>🏥 Medical Visual Question Answering</h1>
        <p id='subtitle'>VQA-RAD · Zero-Shot · Task 2 — Deep Learning</p>
        <p id='strategy'>{STRATEGY_NOTE[STRATEGY]}</p>
    """)

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="Medical Image (X-ray / MRI / CT / Ultrasound)",
                type="pil",
                height=300,
            )
            q_input = gr.Textbox(
                label="Clinical Question",
                placeholder='E.g.: "Is there pleural effusion?" or "Where is the opacity?"',
                lines=2,
            )
            gt_input = gr.Textbox(
                label="Ground Truth Answer (optional — enables ROUGE-L score)",
                placeholder='E.g.: "yes"  or  "left lower lobe"',
                lines=1,
            )
            gr.Examples(
                examples=[[None, q, ""] for q in SAMPLE_QUESTIONS],
                inputs=[img_input, q_input, gt_input],
                label="Sample questions (click to fill)",
            )
            predict_btn = gr.Button("🔍  Run Inference", variant="primary", size="lg")

        # ── Right column: outputs ─────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Prediction")
            ans_out = gr.Textbox(
                label=f"Answer  [{MODEL_LABEL}]",
                interactive=False,
                lines=2,
            )
            summary_out = gr.Textbox(
                label="Details",
                interactive=False,
                lines=8,
            )
            gr.Markdown("""
### ℹ️ Interpreting results
| Score | Meaning |
|-------|---------|
| ROUGE-L ≥ 0.60 | Good lexical match |
| ROUGE-L 0.30-0.59 | Partial match |
| ROUGE-L < 0.30 | Poor match |

**Visual Shortcut** — Yes/No answers score higher not because the model
understands anatomy, but due to pixel-pattern matching.

**Hallucination** — Open-ended answers may sound fluent but be clinically wrong.

> ⚠️ Research prototype — do **not** use for clinical decision-making.
""")

    predict_btn.click(
        fn=run_demo,
        inputs=[img_input, q_input, gt_input],
        outputs=[ans_out, summary_out],
    )
    # Shift+Enter also submits
    q_input.submit(
        fn=run_demo,
        inputs=[img_input, q_input, gt_input],
        outputs=[ans_out, summary_out],
    )

    gpu_info = (
        f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)"
        if USE_GPU else "CPU only"
    )
    gr.Markdown(f"""
---
### How to use
1. Upload an X-ray, CT, MRI, or ultrasound image.
2. Type a clinical question in English.
3. Optionally provide the ground truth answer to see a live ROUGE-L score.
4. Click **Run Inference** (or press Shift+Enter).

### Model strategy (auto-selected by VRAM)
| VRAM available | Model loaded | Notebook-compatible |
|----------------|-------------|---------------------|
| ≥ 10 GB | BLIP-2 OPT-2.7B fp16 | ✅ Identical outputs |
| 6-10 GB | BLIP-2 OPT-2.7B 8-bit | ⚠️ Slight logit diff |
| 1-6 GB  | BLIP-vqa-base fp16 | ❌ Different model |
| No GPU  | BLIP-2 OPT-2.7B CPU | ✅ Same, just slow |

> **Running on**: {gpu_info} → **{MODEL_LABEL}**
""")

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[INFO] Starting demo → http://localhost:{args.port}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=False,
        show_error=True,
    )