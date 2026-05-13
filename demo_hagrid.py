"""
demo_hagrid.py — HaGRID Live Gesture Detection
───────────────────────────────────────────────
Runs real-time hand-gesture recognition using a trained EfficientNet-B0
checkpoint from TRAINING_HaGRID_HuggingFace.ipynb.

Usage
-----
    python demo_hagrid.py                                 # uses default checkpoint path
    python demo_hagrid.py --ckpt path/to/efficientnet_b0_hagrid_final.pth
    python demo_hagrid.py --ckpt path/to/model.pth --cam 1 --threshold 0.6 --topk 3

Controls
--------
    [Q] key          — quit
    [QUIT] button    — click the red button drawn on screen to quit
    [S] key          — save current frame as screenshot
"""

import argparse
import colorsys
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import timm
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
#  CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description='HaGRID Live Gesture Detection')
    parser.add_argument(
        '--ckpt', type=str,
        default='hagrid_checkpoints/efficientnet_b0_hagrid_final.pth',
        help='Path to the .pth checkpoint (default: hagrid_checkpoints/efficientnet_b0_hagrid_final.pth)',
    )
    parser.add_argument(
        '--cam', type=int, default=0,
        help='Camera device index (default: 0)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.50,
        help='Minimum confidence to display a label (default: 0.50)',
    )
    parser.add_argument(
        '--topk', type=int, default=3,
        help='Number of top predictions shown in side panel (default: 3)',
    )
    parser.add_argument(
        '--width', type=int, default=640,
        help='Camera capture width in pixels (default: 640)',
    )
    parser.add_argument(
        '--height', type=int, default=480,
        help='Camera capture height in pixels (default: 480)',
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
ALL_CLASSES = sorted([
    'call', 'dislike', 'fist', 'four', 'like', 'mute', 'no_gesture',
    'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop',
    'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted',
])
CLASS2IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX2CLASS  = {i: c for c, i in CLASS2IDX.items()}

# Label shown on screen (ASCII-safe for OpenCV text rendering)
GESTURE_LABEL = {
    'call'           : 'Call',
    'dislike'        : 'Dislike',
    'fist'           : 'Fist',
    'four'           : 'Four',
    'like'           : 'Like',
    'mute'           : 'Mute',
    'no_gesture'     : 'No Gesture',
    'ok'             : 'OK',
    'one'            : 'One',
    'palm'           : 'Palm',
    'peace'          : 'Peace',
    'peace_inverted' : 'Peace (inv)',
    'rock'           : 'Rock',
    'stop'           : 'Stop',
    'stop_inverted'  : 'Stop (inv)',
    'three'          : 'Three',
    'three2'         : 'Three 2',
    'two_up'         : 'Two Up',
    'two_up_inverted': 'Two Up (inv)',
}


def _cls_color(idx: int, total: int = 19):
    """Return a vivid BGR colour unique to each class index."""
    h = idx / total
    r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR

CLASS_COLORS = {cls: _cls_color(i) for i, cls in enumerate(ALL_CLASSES)}

# ── Design tokens ─────────────────────────────────────────────────────────────
_BG     = (14, 14, 20)       # near-black navy (BGR)
_TXT1   = (235, 238, 245)    # primary text
_TXT2   = (120, 130, 150)    # secondary / dim text
_TRACK  = (42, 44, 56)       # progress-bar track
_GREEN  = (80, 210, 120)     # FPS good
_AMBER  = (60, 160, 230)     # FPS low (BGR orange)
_FONT_D = cv2.FONT_HERSHEY_DUPLEX
_FONT_S = cv2.FONT_HERSHEY_SIMPLEX


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _alpha_rect(img, pt1, pt2, color, alpha):
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    panel = np.zeros_like(roi)
    panel[:] = color
    cv2.addWeighted(panel, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _filled_rrect(img, pt1, pt2, color, r=7):
    x1, y1 = pt1; x2, y2 = pt2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)):
        cv2.circle(img, (cx, cy), r, color, -1)


def _stroke_rrect(img, pt1, pt2, color, r=7, t=1):
    x1, y1 = pt1; x2, y2 = pt2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(img,  (x1+r, y1),  (x2-r, y1),  color, t)
    cv2.line(img,  (x1+r, y2),  (x2-r, y2),  color, t)
    cv2.line(img,  (x1, y1+r),  (x1, y2-r),  color, t)
    cv2.line(img,  (x2, y1+r),  (x2, y2-r),  color, t)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180,  0, 90, color, t)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270,  0, 90, color, t)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90,  0, 90, color, t)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0,  0, 90, color, t)


def _hud_corners(img, pt1, pt2, color, size=18, t=1):
    x1, y1 = pt1; x2, y2 = pt2
    for cx, cy, sx, sy in ((x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)):
        cv2.line(img, (cx, cy), (cx + sx*size, cy), color, t)
        cv2.line(img, (cx, cy), (cx, cy + sy*size), color, t)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_model(ckpt_path: str, device: torch.device):
    path = Path(ckpt_path)
    if not path.exists():
        print(f'[ERROR] Checkpoint not found: {path}')
        print('        Run TRAINING_HaGRID_HuggingFace.ipynb first,')
        print('        or pass --ckpt /path/to/your/model.pth')
        sys.exit(1)

    print(f'[INFO] Loading checkpoint: {path}')
    ckpt = torch.load(path, map_location=device)

    # Restore class mapping saved inside the checkpoint
    idx2class = ckpt.get('idx2class', IDX2CLASS)
    class2idx = ckpt.get('class2idx', CLASS2IDX)
    num_classes = ckpt.get('num_classes', len(idx2class))
    model_name  = ckpt.get('model_name',  'efficientnet_b0')
    img_size    = ckpt.get('img_size',    224)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    val_acc = ckpt.get('val_acc', float('nan'))
    epoch   = ckpt.get('epoch',   '?')
    print(f'[INFO] Model    : {model_name}')
    print(f'[INFO] Epoch    : {epoch}')
    print(f'[INFO] Val acc  : {val_acc:.4f}' if val_acc == val_acc else '[INFO] Val acc  : unknown')
    print(f'[INFO] Classes  : {num_classes}')
    print(f'[INFO] Device   : {device}')

    return model, idx2class, class2idx, img_size


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def build_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(bgr_frame: np.ndarray, model, transform, device, idx2class, top_k: int = 3):
    """
    Run inference on a BGR numpy frame.
    Returns list of (class_name, probability) sorted by confidence descending.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0].cpu()
    top_p, top_i = probs.topk(top_k)
    return [(idx2class[i.item()], p.item()) for i, p in zip(top_i, top_p)]


# ══════════════════════════════════════════════════════════════════════════════
#  OVERLAY DRAWING
# ══════════════════════════════════════════════════════════════════════════════
# Quit button rectangle (x1, y1, x2, y2) — computed at draw time, stored here
# so the mouse callback can check it.
_QUIT_BTN_RECT = [0, 0, 0, 0]   # [x1, y1, x2, y2]


def draw_overlay(frame: np.ndarray,
                 top_preds: list,
                 threshold: float,
                 top_k: int,
                 fps: float = 0.0) -> np.ndarray:
    """
    Draw prediction overlay onto a BGR frame (copy returned).

    Layout
    ──────
    • Top banner (68 px)  — accent stripe · icon · gesture name · confidence bar
    • Right panel (210 px) — top-K ranked predictions with progress bars
    • Bottom bar  (44 px)  — FPS · model tag · [Q] QUIT button
    • HUD corners          — L-bracket decorations on the camera feed area
    """
    out = frame.copy()
    h, w = out.shape[:2]

    pred_class, confidence = top_preds[0]
    above   = confidence >= threshold
    accent  = CLASS_COLORS.get(pred_class, (160, 160, 160))
    dim_acc = tuple(max(0, c // 3) for c in accent)   # dimmed accent for borders

    BANNER_H = 68
    BBAR_H   = 44
    PANEL_W  = 210
    bbar_y0  = h - BBAR_H
    panel_x0 = w - PANEL_W

    # ── 1. TOP BANNER ─────────────────────────────────────────────────────────
    _alpha_rect(out, (0, 0), (w, BANNER_H), _BG, 0.82)

    # Left accent stripe
    cv2.rectangle(out, (0, 0), (7, BANNER_H), accent, -1)

    # Gesture icon circle
    ic_cx, ic_cy = 36, BANNER_H // 2
    cv2.circle(out, (ic_cx, ic_cy), 20, accent, -1)
    cv2.circle(out, (ic_cx, ic_cy), 20, _TXT1, 1)
    init = GESTURE_LABEL.get(pred_class, pred_class)[0].upper()
    cv2.putText(out, init, (ic_cx - 7, ic_cy + 7), _FONT_D, 0.65, _BG, 2, cv2.LINE_AA)

    # Gesture name
    display = GESTURE_LABEL.get(pred_class, pred_class)
    label   = display if above else 'Uncertain...'
    lcol    = _TXT1 if above else _TXT2
    cv2.putText(out, label, (66, 31), _FONT_D, 0.92, lcol, 2, cv2.LINE_AA)

    # Raw class key (sub-label)
    cv2.putText(out, pred_class, (68, 52), _FONT_S, 0.36, _TXT2, 1, cv2.LINE_AA)

    # Confidence bar (right of banner)
    cb_w, cb_h = 170, 10
    cb_x = w - PANEL_W - cb_w - 20
    cb_y = 18
    cv2.putText(out, 'CONFIDENCE', (cb_x, cb_y - 2), _FONT_S, 0.32, _TXT2, 1, cv2.LINE_AA)
    _filled_rrect(out, (cb_x, cb_y + 4), (cb_x + cb_w, cb_y + 4 + cb_h), _TRACK, r=5)
    fill_w = max(0, int(cb_w * confidence))
    if fill_w:
        fill_col = accent if above else _AMBER
        _filled_rrect(out, (cb_x, cb_y + 4), (cb_x + fill_w, cb_y + 4 + cb_h), fill_col, r=5)
    cv2.putText(out, f'{confidence * 100:.1f}%', (cb_x + cb_w + 8, cb_y + 14),
                _FONT_D, 0.55, _TXT1, 1, cv2.LINE_AA)
    cv2.putText(out, '#1 DETECTION', (cb_x, cb_y + 32), _FONT_S, 0.32, _TXT2, 1, cv2.LINE_AA)

    # Bottom border of banner
    cv2.line(out, (0, BANNER_H), (w, BANNER_H), dim_acc, 1)

    # ── 2. BOTTOM STATUS BAR ──────────────────────────────────────────────────
    _alpha_rect(out, (0, bbar_y0), (w, h), _BG, 0.84)
    cv2.line(out, (0, bbar_y0), (w, bbar_y0), dim_acc, 1)

    # FPS
    fps_col = _GREEN if fps > 20 else _AMBER
    cv2.putText(out, f'{fps:.0f}', (14, bbar_y0 + 29), _FONT_D, 0.78, fps_col, 2, cv2.LINE_AA)
    cv2.putText(out, 'FPS', (52, bbar_y0 + 29), _FONT_S, 0.36, _TXT2, 1, cv2.LINE_AA)
    cv2.circle(out, (78, bbar_y0 + 22), 2, _TRACK, -1)

    # Centre label
    ctr = 'HaGRID  |  EfficientNet-B0'
    (tw, _), _ = cv2.getTextSize(ctr, _FONT_S, 0.40, 1)
    cv2.putText(out, ctr, ((w - tw) // 2, bbar_y0 + 29), _FONT_S, 0.40, _TXT2, 1, cv2.LINE_AA)

    # QUIT button
    btn_w2, btn_h2 = 90, 28
    bx1 = w - btn_w2 - 12;  by1 = bbar_y0 + (BBAR_H - btn_h2) // 2
    bx2 = bx1 + btn_w2;     by2 = by1 + btn_h2
    _QUIT_BTN_RECT[:] = [bx1, by1, bx2, by2]
    _filled_rrect(out, (bx1, by1), (bx2, by2), (38, 38, 195), r=6)
    _stroke_rrect(out, (bx1, by1), (bx2, by2), (90, 90, 245), r=6, t=1)
    cv2.putText(out, '[Q] QUIT', (bx1 + 7, by1 + 19), _FONT_S, 0.40, _TXT1, 1, cv2.LINE_AA)

    # ── 3. RIGHT SIDE PANEL ───────────────────────────────────────────────────
    _alpha_rect(out, (panel_x0, BANNER_H), (w, bbar_y0), _BG, 0.72)
    cv2.line(out, (panel_x0, BANNER_H), (panel_x0, bbar_y0), dim_acc, 1)

    avail_h  = bbar_y0 - BANNER_H
    row_h    = avail_h // top_k

    for rank, (cls, prob) in enumerate(top_preds):
        ry0  = BANNER_H + rank * row_h
        ry1  = ry0 + row_h
        mid  = (ry0 + ry1) // 2
        bcol = CLASS_COLORS.get(cls, _TRACK)

        if rank > 0:
            cv2.line(out, (panel_x0 + 8, ry0), (w - 8, ry0), _TRACK, 1)

        px = panel_x0 + 10

        # Rank badge + name
        cv2.putText(out, f'#{rank + 1}', (px, mid - 8), _FONT_D, 0.46, bcol, 1, cv2.LINE_AA)
        short = GESTURE_LABEL.get(cls, cls)[:15]
        cv2.putText(out, short, (px + 26, mid - 8), _FONT_S, 0.43, _TXT1, 1, cv2.LINE_AA)

        # Progress bar
        bar_x0  = px + 26
        bar_y_c = mid + 4
        bar_ww  = PANEL_W - 50
        _filled_rrect(out, (bar_x0, bar_y_c), (bar_x0 + bar_ww, bar_y_c + 8), _TRACK, r=4)
        fill2 = max(0, int(bar_ww * prob))
        if fill2:
            dim = 1 if rank == 0 else 2
            fc2 = tuple(max(0, min(255, c // dim + 30*(2-dim))) for c in bcol)
            _filled_rrect(out, (bar_x0, bar_y_c), (bar_x0 + fill2, bar_y_c + 8), fc2, r=4)

        # Probability
        cv2.putText(out, f'{prob * 100:.1f}%', (bar_x0, mid + 26),
                    _FONT_S, 0.36, _TXT2, 1, cv2.LINE_AA)

    # ── 4. HUD CORNER DECORATIONS ─────────────────────────────────────────────
    feed_x2 = panel_x0 - 4
    feed_y1 = BANNER_H + 4
    feed_y2 = bbar_y0 - 4
    _hud_corners(out, (4, feed_y1), (feed_x2, feed_y2), dim_acc, size=20, t=1)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MOUSE CALLBACK
# ══════════════════════════════════════════════════════════════════════════════
class _State:
    quit_requested = False

def mouse_callback(event, x, y, flags, param):
    """Set quit flag when the user left-clicks inside the QUIT button."""
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = _QUIT_BTN_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            _State.quit_requested = True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using device: {device}')

    # Load model
    model, idx2class, class2idx, img_size = load_model(args.ckpt, device)
    transform = build_transform(img_size)

    # Open camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {args.cam}')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] Camera {args.cam} opened at {actual_w}×{actual_h}')
    print('[INFO] Press  Q  or click  QUIT  to exit')
    print('[INFO] Press  S  to save a screenshot')

    # OpenCV window
    WIN = 'HaGRID — Live Gesture Detection'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, actual_w, actual_h)
    # NOTE: setMouseCallback is registered after the first imshow() below.
    # Qt's window handler is NULL until the window is actually rendered;
    # calling setMouseCallback before any imshow causes a -27 crash.

    frame_count   = 0
    fps           = 0.0
    t_prev        = time.perf_counter()
    _callback_set = False   # guard: register the mouse callback exactly once

    # Cache last prediction so overlay never flickers on inference errors
    top_preds_cache = [(cls, 0.0) for cls in list(idx2class.values())[:args.topk]]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('[WARN] Failed to read frame — retrying...')
                time.sleep(0.05)
                continue

            # ── Inference ─────────────────────────────────────────────────
            try:
                top_preds = predict(
                    frame, model, transform, device, idx2class, top_k=args.topk
                )
                top_preds_cache = top_preds
            except Exception as exc:
                print(f'[WARN] Inference error: {exc}')
                top_preds = top_preds_cache

            # ── FPS ───────────────────────────────────────────────────────
            frame_count += 1
            t_now = time.perf_counter()
            elapsed = t_now - t_prev
            if elapsed >= 0.5:                         # update FPS every 0.5 s
                fps    = frame_count / elapsed
                frame_count = 0
                t_prev = t_now

            # ── Draw & show ───────────────────────────────────────────────
            out = draw_overlay(
                frame, top_preds,
                threshold=args.threshold,
                top_k=args.topk,
                fps=fps,
            )
            cv2.imshow(WIN, out)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            # Best-effort mouse callback — skip silently if the Qt window
            # handler is unavailable (broken OpenCV Qt build, missing fonts, etc.)
            if not _callback_set:
                try:
                    cv2.setMouseCallback(WIN, mouse_callback)
                except cv2.error:
                    pass   # Q key / Esc still quit; QUIT button just won't be clickable
                _callback_set = True
            if key == ord('q') or key == ord('Q') or key == 27:   # Q / Esc
                print('[INFO] Q pressed — exiting.')
                break
            if key == ord('s') or key == ord('S'):
                fname = f'hagrid_screenshot_{int(time.time())}.png'
                cv2.imwrite(fname, out)
                print(f'[INFO] Screenshot saved → {fname}')

            # ── Quit button ───────────────────────────────────────────────
            if _State.quit_requested:
                print('[INFO] QUIT button clicked — exiting.')
                break

            # ── Window closed by OS ───────────────────────────────────────
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                print('[INFO] Window closed — exiting.')
                break

    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user.')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print('[INFO] Camera released. Bye!')


if __name__ == '__main__':
    main()