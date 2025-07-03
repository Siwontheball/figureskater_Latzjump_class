# src/annotate_with_jump.py

import sys
import cv2
import torch
import numpy as np

# 1) replace these with your actual module paths
from extract_skeletons import process_video  
from model import LSTMClassifier as JumpModel          

CLASS_NAMES = ["Inside", "Right", "Flat"]

def load_classifier(weights_path="best_model.pt", device="cpu"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = JumpModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device

def classify_clip(video_path, model, device):
    """
    Extracts a single (60,feat_dim) skeleton sequence and returns the predicted class.
    """
    seq = process_video(video_path)           # shape (60, feature_dim)
    x   = torch.from_numpy(seq[None]).float().to(device)
    with torch.no_grad():
        logits = model(x)
        pred   = logits.argmax(dim=1).item()
    return CLASS_NAMES[pred]

def annotate_video(input_path, output_path, model, device):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {input_path}")

    # prepare writer
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # get the predicted jump class once (whole‐clip classification)
    label = classify_clip(input_path, model, device)

    # overlay it on every frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw the label in red at top-left
        cv2.putText(
            frame,
            f"Lutz edge: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[✓] Annotated video saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/annotate_with_jump.py input.mp4 output.mp4")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    model, device = load_classifier("best_model.pt")
    annotate_video(inp, outp, model, device)
