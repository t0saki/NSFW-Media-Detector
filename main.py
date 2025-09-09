import os
import argparse
import csv
from pathlib import Path
import time
import abc
import torch
import timm
import traceback
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Dependency for the alternative model ---
try:
    from transformers import AutoModelForImageClassification, ViTImageProcessor
except ImportError:
    print("Warning: `transformers` library not found. The 'falconsai' model will not be available.")
    print("Install it with: pip install transformers")
    AutoModelForImageClassification = None
    ViTImageProcessor = None

# --- HEIC/AVIF Support (Optional Dependencies) ---
# We register openers for these formats. If the libraries are not installed,
# the script will still run but will skip these file types.
try:
    import pillow_heif
    import pillow_avif
    pillow_heif.register_heif_opener()
    print("HEIC/HEIF support enabled.")
except ImportError:
    print("Warning: `pillow-heif` not found. HEIC/HEIF files will raise an error on open.")
    print("To process them, run: pip install pillow-heif")
# --- End of new code block ---


# --- Constants ---
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg",
                              ".png", ".bmp", ".webp", ".heic", ".heif", ".avif"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
MIN_VIDEO_SAMPLES = 2
MAX_VIDEO_SAMPLES = 10

# --- Model Abstraction ---


class NsfwDetector(abc.ABC):
    """Abstract base class for NSFW detectors."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    @abc.abstractmethod
    def predict(self, image_batch: list[Image.Image]) -> np.ndarray:
        """
        Analyzes a batch of PIL images and returns NSFW probabilities.
        Args:
            image_batch: A list of RGB PIL.Image.Image objects.
        Returns:
            A numpy array of NSFW probabilities, one for each image.
        """
        pass


class TimmMarqoDetector(NsfwDetector):
    """Detector for the Marqo/nsfw-image-detection-384 model using timm."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        model_id = "hf_hub:Marqo/nsfw-image-detection-384"
        print(f"Loading {model_id} model (timm)...")
        self.model = timm.create_model(
            model_id, pretrained=True).to(self.device).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(
            **data_config, is_training=False)
        self.nsfw_idx = self.model.pretrained_cfg["label_names"].index("NSFW")
        print("Marqo model loaded successfully.")

    def predict(self, image_batch: list[Image.Image]) -> np.ndarray:
        if not image_batch:
            return np.array([])
        tensors = torch.stack([self.transforms(img)
                              for img in image_batch]).to(self.device)
        with torch.no_grad():
            output = self.model(tensors).softmax(dim=-1)
        return output[:, self.nsfw_idx].cpu().numpy()


class HfFalconsaiDetector(NsfwDetector):
    """Detector for the Falconsai/nsfw_image_detection model using transformers."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        if ViTImageProcessor is None:
            raise ImportError(
                "`transformers` library is required for the 'falconsai' model.")
        model_id = "Falconsai/nsfw_image_detection"
        print(f"Loading {model_id} model (transformers)...")
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id).to(self.device).eval()
        self.nsfw_idx = int(self.model.config.label2id["nsfw"])
        print(f"{model_id} model loaded successfully.")

    def predict(self, image_batch: list[Image.Image]) -> np.ndarray:
        if not image_batch:
            return np.array([])
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return outputs.logits.softmax(dim=-1)[:, self.nsfw_idx].cpu().numpy()


DETECTOR_MAPPING = {
    "marqo": TimmMarqoDetector,
    "falconsai": HfFalconsaiDetector,
}

# --- Global variable for worker processes ---
worker_detector = None


def analyze_image_batch(image_paths: list, detector: NsfwDetector):
    """
    Analyzes a batch of images for NSFW content with a fallback to individual processing.
    """
    images, valid_paths, batch_results = [], [], []
    for path in image_paths:
        try:
            # With the registered openers, Image.open now supports HEIC/AVIF
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(
                f"Warning: Could not open image {path}, skipping. Error: {e}")
            batch_results.append({"path": path, "prob": -1.0})

    if not images:
        return batch_results

    try:
        # Attempt batch processing first for efficiency
        nsfw_probs = detector.predict(images)
        for i, path in enumerate(valid_paths):
            batch_results.append({"path": path, "prob": nsfw_probs[i]})
        return batch_results
    except Exception as e:
        print(
            f"Warning: Batch processing failed. Error: {e}. Retrying files individually...")
        # Fallback to individual processing
        # First, add back the results for files that failed to open initially
        final_results = [res for res in batch_results if res['prob'] == -1.0]
        for i, img in enumerate(images):
            path = valid_paths[i]
            try:
                nsfw_prob = detector.predict([img])[0]
                final_results.append({"path": path, "prob": nsfw_prob})
            except Exception as individual_e:
                print(
                    f"Error processing individual file {path} after batch failure. Error: {individual_e}")
                final_results.append({"path": path, "prob": -1.0})
        return final_results

# --- Video and Main Logic (No changes needed here) ---


def _perform_video_analysis(video_path: str, detector: NsfwDetector) -> dict:
    """
    Core logic for analyzing a video with dynamic frame sampling.
    """
    max_nsfw_prob = 0.0
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return {"path": video_path, "prob": -1.0}
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            return {"path": video_path, "prob": 0.0}
        num_samples = min(MAX_VIDEO_SAMPLES, max(
            MIN_VIDEO_SAMPLES, total_frames))
        start_frame = int(total_frames * 0.05)
        end_frame = int(total_frames * 0.95)
        if start_frame >= end_frame:
            start_frame, end_frame = 0, max(0, total_frames - 1)

        sample_indices = np.unique(np.linspace(
            start_frame, end_frame, num=num_samples, dtype=int))

        for frame_index in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_nsfw_prob = detector.predict([img])[0]
            if current_nsfw_prob > max_nsfw_prob:
                max_nsfw_prob = current_nsfw_prob
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return {"path": video_path, "prob": -1.0}
    finally:
        if cap:
            cap.release()
    return {"path": video_path, "prob": max_nsfw_prob}


def analyze_video_worker(video_path: str, model_name: str) -> dict:
    """Worker function for parallel video analysis (uses CPU)."""
    global worker_detector
    if worker_detector is None:
        device = torch.device("cpu")
        detector_class = DETECTOR_MAPPING[model_name]
        print(
            f"Initializing detector '{model_name}' in worker {os.getpid()}...")
        worker_detector = detector_class(device)
    return _perform_video_analysis(video_path, worker_detector)


def yield_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance NSFW Image/Video Detection with resume capability."
    )
    parser.add_argument("input_folder", type=str,
                        help="Path to the folder to scan recursively.")
    parser.add_argument("--output-file", type=str,
                        default="nsfw_detection_results.csv", help="Path to output CSV file.")
    parser.add_argument("--model-name", type=str, default="falconsai",
                        choices=DETECTOR_MAPPING.keys(), help="Name of the detector model to use.")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Probability threshold to classify as NSFW (0.0 to 1.0).")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of images to process in a single batch (for GPU).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU cores for video processing. Set to 0 to disable parallelism.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA and force CPU usage.")

    args = parser.parse_args()
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Model Initialization ---
    try:
        detector_class = DETECTOR_MAPPING[args.model_name]
        detector = detector_class(device)
    except (ImportError, ValueError, KeyError) as e:
        print(f"Error initializing model '{args.model_name}': {e}")
        return

    # --- Resume Logic ---
    processed_files = set()
    if os.path.exists(args.output_file):
        print(f"Resuming from existing output file: {args.output_file}")
        try:
            with open(args.output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    if row:
                        processed_files.add(row[0])
            print(
                f"Found {len(processed_files)} already processed files. They will be skipped.")
        except Exception as e:
            print(
                f"Warning: Could not read existing output file. Starting from scratch. Error: {e}")
            processed_files.clear()

    # --- File Discovery ---
    print(f"Scanning for media files in '{args.input_folder}'...")
    all_image_files, all_video_files = [], []
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            ext = Path(file).suffix.lower()
            full_path = os.path.join(root, file)
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                all_image_files.append(full_path)
            elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                all_video_files.append(full_path)

    # Filter out already processed files
    images_to_process = [
        f for f in all_image_files if f not in processed_files]
    videos_to_process = [
        f for f in all_video_files if f not in processed_files]

    total_new_files = len(images_to_process) + len(videos_to_process)
    print(
        f"Found {len(all_image_files)} total images and {len(all_video_files)} total videos.")
    print(f"Processing {total_new_files} new files this session.")

    if total_new_files == 0:
        print("No new files to process. Exiting.")
        return

    # --- Open output file for appending and process files ---
    files_processed_this_session = 0
    nsfw_found_this_session = 0
    fieldnames = ["File Path", "Prediction", "NSFW Probability"]

    try:
        with open(args.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header only if the file is new/empty
            csvfile.seek(0, 2)  # Go to the end of the file
            if csvfile.tell() == 0:
                writer.writeheader()

            # --- Process Images ---
            if images_to_process:
                pbar = tqdm(total=len(images_to_process),
                            desc="Processing Images")
                for batch_paths in yield_batches(images_to_process, args.batch_size):
                    batch_results = analyze_image_batch(batch_paths, detector)
                    for res in batch_results:
                        prob = res['prob']
                        pred = "ERROR" if prob < 0 else "NSFW" if prob >= args.threshold else "SFW"
                        if pred == "NSFW":
                            nsfw_found_this_session += 1
                        writer.writerow(
                            {"File Path": res['path'], "Prediction": pred, "NSFW Probability": f"{prob:.4f}" if prob >= 0 else "N/A"})
                    csvfile.flush()  # Ensure data is written to disk
                    pbar.update(len(batch_paths))
                    files_processed_this_session += len(batch_paths)
                pbar.close()

            # --- Process Videos ---
            if videos_to_process:
                use_parallel = device.type == 'cpu' and args.workers != 0
                if use_parallel:
                    num_workers = args.workers or os.cpu_count()
                    print(
                        f"Processing {len(videos_to_process)} videos in parallel with {num_workers} workers...")
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        futures = {executor.submit(
                            analyze_video_worker, path, args.model_name): path for path in videos_to_process}
                        for future in tqdm(as_completed(futures), total=len(videos_to_process), desc="Processing Videos"):
                            res = future.result()
                            prob = res['prob']
                            pred = "ERROR" if prob < 0 else "NSFW" if prob >= args.threshold else "SFW"
                            if pred == "NSFW":
                                nsfw_found_this_session += 1
                            writer.writerow(
                                {"File Path": res['path'], "Prediction": pred, "NSFW Probability": f"{prob:.4f}" if prob >= 0 else "N/A"})
                            csvfile.flush()
                            files_processed_this_session += 1
                else:
                    print(
                        f"Processing {len(videos_to_process)} videos serially on {device.type}...")
                    for path in tqdm(videos_to_process, desc="Processing Videos"):
                        res = _perform_video_analysis(path, detector)
                        prob = res['prob']
                        pred = "ERROR" if prob < 0 else "NSFW" if prob >= args.threshold else "SFW"
                        if pred == "NSFW":
                            nsfw_found_this_session += 1
                        writer.writerow(
                            {"File Path": res['path'], "Prediction": pred, "NSFW Probability": f"{prob:.4f}" if prob >= 0 else "N/A"})
                        csvfile.flush()
                        files_processed_this_session += 1
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved. Run the script again to resume.")

    # --- Final Summary ---
    end_time = time.time()
    total_nsfw_count = 0
    with open(args.output_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Prediction"] == "NSFW":
                total_nsfw_count += 1

    print("\n--- Detection Complete ---")
    print(f"Processed {files_processed_this_session} new files this session.")
    print(
        f"Total files in output: {len(processed_files) + files_processed_this_session}")
    print(
        f"Total NSFW files found in output: {total_nsfw_count} (threshold >= {args.threshold})")
    print(
        f"Processing time for this session: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # 'spawn' is recommended for CUDA compatibility in multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
