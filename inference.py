import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.model.GenD import GenD
from src import config as C
from src.retinaface import prepare_model


# --- Face Alignment Function (from detector.py) ---
def align_face(
    img: np.ndarray,
    landmarks: np.ndarray,
    target_size=(224, 224),
    scale=1.3,
):
    dst = np.array(
        [
            [0.34, 0.46],
            [0.66, 0.46],
            [0.5, 0.64],
            [0.37, 0.82],
            [0.63, 0.82],
        ],
        dtype=np.float32,
    )

    dst[:, 0] = dst[:, 0] * target_size[0]
    dst[:, 1] = dst[:, 1] * target_size[1]

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.0
    y_margin = target_size[1] * margin_rate / 2.0

    # move
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    # resize
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmarks.astype(np.float32)

    M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]

    img = cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)

    return img


# --- File Processing ---
def process_image_file(file_path, detector, target_size=(224, 224)):
    try:
        img = cv2.imread(file_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = detector.detect(img)
        xyxy, landmarks = preds

        if len(xyxy) == 0:
            return None

        # Select largest face
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = np.argmax(areas)
        selected_landmarks = landmarks[idx]

        aligned_img = align_face(img, selected_landmarks, target_size=target_size)
        return Image.fromarray(aligned_img)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_video_file(file_path, detector, target_size=(224, 224), num_frames=32):
    frames = []
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return []

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = detector.detect(frame_rgb)
            xyxy, landmarks = preds

            if len(xyxy) > 0:
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                idx = np.argmax(areas)
                selected_landmarks = landmarks[idx]

                aligned_img = align_face(frame_rgb, selected_landmarks, target_size=target_size)
                frames.append(Image.fromarray(aligned_img))

        cap.release()
    except Exception as e:
        print(f"Error processing video {file_path}: {e}")

    return frames


def main():
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default=str(root / "test_data"))
    parser.add_argument("--output_csv", type=str, default=str(root / "output.csv"))
    parser.add_argument("--checkpoint", type=str, default=str(root / "model" / "model.pt"))
    parser.add_argument("--config", type=str, default=str(root / "config" / "config.yaml"))
    parser.add_argument("--det_thres", type=float, default=0.4)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="How to aggregate frame predictions for videos: mean, max, or median",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = C.load_config(args.config)

    # Initialize model
    model = GenD(config, verbose=False)

    # Load checkpoint weights
    print("Loading GenD model from checkpoint...")
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")

    print("Loading RetinaFace detector...")
    detector = prepare_model(args.det_thres)

    # Prepare Output CSV
    with open(args.output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prob"])

    # Process Files
    if not os.path.exists(args.input_folder):
        print(f"Input folder {args.input_folder} does not exist.")
        return

    files = sorted(
        [f for f in os.listdir(args.input_folder) if os.path.isfile(os.path.join(args.input_folder, f))]
    )

    results = {}

    print(f"Processing {len(files)} files...")

    # Statistics counters
    stats = {"image": {"total": 0, "detected": 0}, "video": {"total": 0, "detected": 0}}

    for filename in tqdm(files):
        file_path = os.path.join(args.input_folder, filename)
        ext = os.path.splitext(filename)[1].lower()

        face_images = []
        is_image = False
        is_video = False

        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".jfif"]:
            is_image = True
            stats["image"]["total"] += 1
            face_img = process_image_file(file_path, detector)
            if face_img:
                face_images.append(face_img)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            is_video = True
            stats["video"]["total"] += 1
            face_images = process_video_file(file_path, detector)

        prob = 0.0  # Default to Real if no face found or error

        if face_images:
            if is_image:
                stats["image"]["detected"] += 1
            if is_video:
                stats["video"]["detected"] += 1
            try:
                # Preprocess and Inference
                tensors = torch.stack([model.get_preprocessing()(img) for img in face_images]).to(device)

                with torch.no_grad():
                    logits = model(tensors).logits_labels
                    probs = logits.softmax(dim=-1)

                # Aggregate probabilities across frames (for video)
                # Assuming index 1 is Fake
                fake_probs = probs[:, 1]

                if args.aggregation == "mean":
                    prob = fake_probs.mean().item()
                elif args.aggregation == "max":
                    prob = fake_probs.max().item()
                elif args.aggregation == "median":
                    prob = fake_probs.median().item()

            except Exception as e:
                print(f"Inference error for {filename}: {e}")

        results[filename] = prob

    # Write Results
    with open(args.output_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        for filename in files:
            writer.writerow([filename, results.get(filename, 0)])

    print(f"Done! Results saved to {args.output_csv}")

    # Print Statistics
    print("\n=== Processing Statistics ===")
    print(
        f"Images: {stats['image']['detected']}/{stats['image']['total']} "
        f"({(stats['image']['detected']/stats['image']['total']*100 if stats['image']['total']>0 else 0):.1f}%)"
    )
    print(
        f"Videos: {stats['video']['detected']}/{stats['video']['total']} "
        f"({(stats['video']['detected']/stats['video']['total']*100 if stats['video']['total']>0 else 0):.1f}%)"
    )
    total_files = stats["image"]["total"] + stats["video"]["total"]
    total_detected = stats["image"]["detected"] + stats["video"]["detected"]
    print(
        f"Total: {total_detected}/{total_files} "
        f"({(total_detected/total_files*100 if total_files>0 else 0):.1f}%)"
    )
    print("=============================\n")


if __name__ == "__main__":
    main()
