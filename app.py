"""
app.py – Medical VQA Demo (English) — localhost
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
  BLIP-2 OPT-2.7B  8-bit  -> ~8 GB  (full model, notebook-compatible)
  BLIP-vqa-base    fp16   -> ~1 GB  (smaller, faster, lower accuracy)
  CPU fallback             -> 0 GB VRAM (slow ~60s/query)
"""

import argparse
import time

import gradio as gr
import torch
from PIL import Image
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForQuestionAnswering,
    BlipProcessor,
    BitsAndBytesConfig,
)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Medical VQA Demo - B1 Zero-shot")
parser.add_argument("--port",  type=int, default=7860)
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
    vram_gb = 0
    print("[INFO] No GPU — running on CPU")

# Auto-select strategy based on available VRAM
# < 6 GB  -> use small BLIP model (fits comfortably)
# >= 6 GB -> use BLIP-2 OPT-2.7B 8-bit (matches notebook)
USE_SMALL = args.small or (USE_GPU and vram_gb < 6.0)

if USE_SMALL:
    STRATEGY = "small"
elif USE_GPU:
    STRATEGY = "blip2-gpu"
else:
    STRATEGY = "blip2-cpu"

print(f"[INFO] Strategy : {STRATEGY}")

# ── Load model ────────────────────────────────────────────────────────────────
BLIP2_ID = "Salesforce/blip2-opt-2.7b"
SMALL_ID = "Salesforce/blip-vqa-base"   # ~400 MB weights, ~1 GB VRAM

if STRATEGY == "small":
    print(f"[INFO] Loading {SMALL_ID} (small model) ...")
    processor = BlipProcessor.from_pretrained(SMALL_ID)
    model = BlipForQuestionAnswering.from_pretrained(
        SMALL_ID,
        torch_dtype=torch.float16 if USE_GPU else torch.float32,
    )
    model = model.to("cuda:0" if USE_GPU else "cpu").eval()
    MODEL_LABEL = "BLIP-vqa-base (small)"

elif STRATEGY == "blip2-gpu":
    print(f"[INFO] Loading {BLIP2_ID} (8-bit GPU) ...")
    bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)
    processor = Blip2Processor.from_pretrained(BLIP2_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID,
        quantization_config=bnb_8bit,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    MODEL_LABEL = "BLIP-2 OPT-2.7B (8-bit GPU)"

else:
    print(f"[INFO] Loading {BLIP2_ID} (CPU float32) ...")
    processor = Blip2Processor.from_pretrained(BLIP2_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_ID,
        torch_dtype=torch.float32,
    ).eval()
    MODEL_LABEL = "BLIP-2 OPT-2.7B (CPU)"

print(f"[OK] {MODEL_LABEL} loaded.")

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pil_image, question):
    img = pil_image.convert("RGB")
    t0  = time.time()

    if STRATEGY == "small":
        inputs = processor(images=img, text=question, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        pred = processor.decode(out[0], skip_special_tokens=True).strip().lower()

    else:
        # Same prompt & params as notebook Cell 18
        prompt = (
            "You are a radiology expert. "
            "Answer the question with one or two words only.\n"
            f"Question: {question}\n"
            "Answer:"
        )
        inputs = processor(images=img, text=prompt, return_tensors="pt")
        target = "cuda:0" if STRATEGY == "blip2-gpu" else "cpu"
        inputs = {k: v.to(target) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                num_beams=5,
                do_sample=False,
                repetition_penalty=1.3,
                length_penalty=0.5,
                early_stopping=True,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        pred = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in pred:
            pred = pred.split("Answer:")[-1]
        pred = pred.strip().split("\n")[0].strip().lower()

    return pred, time.time() - t0


# ── Gradio callback ───────────────────────────────────────────────────────────
def run_demo(image, question, ground_truth):
    if image is None:
        return "Please upload a medical image.", ""
    if not question.strip():
        return "Please enter a question.", ""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    answer, elapsed = predict(image, question.strip())

    lines = [
        f"Question : {question.strip()}",
        f"Answer   : {answer}",
        f"Inference: {elapsed:.2f}s",
        f"Model    : {MODEL_LABEL}",
    ]

    if ground_truth.strip():
        from rouge_score import rouge_scorer as rouge_lib
        rouge_fn = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
        rL = rouge_fn.score(
            ground_truth.strip().lower(), answer
        )["rougeL"].fmeasure
        lines.append(f"ROUGE-L  : {rL:.3f}  (vs ground truth)")

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
]

STRATEGY_NOTE = {
    "small"    : "BLIP-vqa-base — GPU VRAM < 6 GB detected, using small model",
    "blip2-gpu": "BLIP-2 OPT-2.7B — 8-bit GPU (notebook-compatible)",
    "blip2-cpu": "BLIP-2 OPT-2.7B — CPU mode (slow, ~60s per query)",
}

CSS = """
#title    { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #64748b; font-size: 0.88em; margin-bottom: 4px; }
#strategy { text-align: center; font-size: 0.85em; margin-bottom: 12px; }
"""

with gr.Blocks(title="Medical VQA", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML(f"""
        <h1 id='title'>Medical Visual Question Answering</h1>
        <p id='subtitle'>VQA-RAD (English) | Zero-shot | Task 2 - Deep Learning</p>
        <p id='strategy'>{STRATEGY_NOTE[STRATEGY]}</p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="Medical Image (X-ray / MRI / CT / Ultrasound)",
                type="pil",
                height=300,
            )
            q_input = gr.Textbox(
                label="Clinical Question (English)",
                placeholder="E.g.: Is there any abnormality in this image?",
                lines=2,
            )
            gt_input = gr.Textbox(
                label="Ground Truth Answer (optional - enables ROUGE-L score)",
                placeholder="E.g.: yes",
                lines=1,
            )
            gr.Examples(
                examples=[[None, q, ""] for q in SAMPLE_QUESTIONS],
                inputs=[img_input, q_input, gt_input],
                label="Sample questions (click to fill)",
            )
            predict_btn = gr.Button("Predict", variant="primary", size="lg")

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
                lines=7,
            )

    predict_btn.click(
        fn=run_demo,
        inputs=[img_input, q_input, gt_input],
        outputs=[ans_out, summary_out],
    )

    gpu_info = (
        f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)"
        if USE_GPU else "CPU"
    )
    gr.Markdown(f"""
---
### How to use
1. Upload an X-ray, CT, MRI, or ultrasound image.
2. Type a clinical question in English.
3. Optionally provide the ground truth answer to see a live ROUGE-L score.
4. Click **Predict**.

### Model strategy (auto-selected by VRAM)
| VRAM available | Model loaded | Run flag |
|----------------|-------------|----------|
| >= 8 GB | BLIP-2 OPT-2.7B 8-bit | *(auto)* |
| 1 – 6 GB | BLIP-vqa-base fp16 | `--small` |
| No GPU | BLIP-2 OPT-2.7B CPU | `--cpu` |

> **Running on**: {gpu_info} → **{MODEL_LABEL}**
""")

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n[INFO] Starting demo -> http://localhost:{args.port}\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=False,
        show_error=True,
    )