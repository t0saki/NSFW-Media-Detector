import os
import argparse
import csv
from pathlib import Path
import time
import abc
import torch
from torch.utils.data import Dataset, DataLoader
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
try:
    import pillow_heif
    if hasattr(pillow_heif, 'register_heif_opener'):
        pillow_heif.register_heif_opener()
        print("HEIC/HEIF support enabled via register_heif_opener.")
    else:
        from PIL import HeifImagePlugin
        print("HEIC/HEIF support enabled via legacy HeifImagePlugin.")
except ImportError:
    print("Warning: `pillow-heif` not found. HEIC/HEIF files may not be processed.")
    print("To process them, run: pip install pillow-heif")

try:
    import pillow_avif
    if hasattr(pillow_avif, 'register_avif_opener'):
        pillow_avif.register_avif_opener()
        print("AVIF support enabled.")
except ImportError:
    print("Warning: `pillow-avif` not found. AVIF files may not be processed.")

# --- Constants ---
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg",
                              ".png", ".bmp", ".webp", ".heic", ".heif", ".avif"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
MIN_VIDEO_SAMPLES = 2
MAX_VIDEO_SAMPLES = 10

# --- NEW: Picklable Transform Class for Hugging Face Processor ---


class HfProcessorTransform:
    """
    A picklable transform class for Hugging Face processors.
    This replaces the unpicklable lambda function.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        # The logic from the original lambda function is now here
        return self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)

# --- Model Abstraction ---


class NsfwDetector(abc.ABC):
    """Abstract base class for NSFW detectors."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    @abc.abstractmethod
    def get_transforms(self):
        """Returns the appropriate transformation function for the model."""
        pass

    @abc.abstractmethod
    def predict_tensor_batch(self, tensor_batch: torch.Tensor) -> np.ndarray:
        """
        Analyzes a batch of tensors and returns NSFW probabilities.
        """
        pass

    def predict(self, image_batch: list[Image.Image]) -> np.ndarray:
        """
        Analyzes a batch of PIL images. Convenience wrapper for single images.
        """
        if not image_batch:
            return np.array([])
        # This path is more complex for HF models, so we let the subclass handle it if needed
        transforms = self.get_transforms()
        tensors = torch.stack([transforms(img)
                              for img in image_batch]).to(self.device)
        return self.predict_tensor_batch(tensors)


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

    def get_transforms(self):
        return self.transforms

    def predict_tensor_batch(self, tensor_batch: torch.Tensor) -> np.ndarray:
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor_batch).softmax(dim=-1)
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

    def get_transforms(self):
        # --- FIX: Return an instance of our new picklable class ---
        return HfProcessorTransform(self.processor)

    def predict(self, image_batch: list[Image.Image]) -> np.ndarray:
        # Override for direct processor use, which is more efficient for this model
        if not image_batch:
            return np.array([])
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits
        return logits.softmax(dim=-1)[:, self.nsfw_idx].cpu().numpy()

    def predict_tensor_batch(self, tensor_batch: torch.Tensor) -> np.ndarray:
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(pixel_values=tensor_batch).logits
        return logits.softmax(dim=-1)[:, self.nsfw_idx].cpu().numpy()


DETECTOR_MAPPING = {
    "marqo": TimmMarqoDetector,
    "falconsai": HfFalconsaiDetector,
}

# --- PyTorch Dataset for efficient, parallel image loading ---


class ImageFileDataset(Dataset):
    """
    A PyTorch Dataset to load images from a list of file paths.
    """

    def __init__(self, file_paths: list[str], transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, path
        except Exception:
            # If an image is corrupt, return None. Filtered by safe_collate.
            return None, path


def safe_collate(batch):
    """
    A custom collate_fn that filters out None values from a batch.
    """
    batch = [(img, path) for img, path in batch if img is not None]
    if not batch:
        return torch.tensor([]), []
    images, paths = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(paths)


# --- Global variable for worker processes ---
worker_detector = None


def _perform_video_analysis(video_path: str, detector: NsfwDetector) -> dict:
    max_nsfw_prob = 0.0
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return {"path": video_path, "prob": -1.0}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    global worker_detector
    if worker_detector is None:
        device = torch.device("cpu")
        detector_class = DETECTOR_MAPPING[model_name]
        print(
            f"Initializing detector '{model_name}' in worker {os.getpid()}...")
        worker_detector = detector_class(device)
    return _perform_video_analysis(video_path, worker_detector)


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
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of images to process in a single batch on the GPU.")
    parser.add_argument("--image-workers", type=int, default=4,
                        help="Number of CPU cores for parallel image loading. Set to 0 to disable.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU cores for parallel video processing. Defaults to all available cores. Set to 0 to disable.")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA and force CPU usage.")
    args = parser.parse_args()
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    try:
        detector_class = DETECTOR_MAPPING[args.model_name]
        detector = detector_class(device)
    except (ImportError, ValueError, KeyError) as e:
        print(f"Error initializing model '{args.model_name}': {e}")
        return

    processed_files = set()
    if os.path.exists(args.output_file):
        print(f"Resuming from existing output file: {args.output_file}")
        try:
            with open(args.output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if row:
                        processed_files.add(row[0])
            print(
                f"Found {len(processed_files)} already processed files. They will be skipped.")
        except Exception as e:
            print(
                f"Warning: Could not read existing output file. Starting from scratch. Error: {e}")
            processed_files.clear()

    print(f"Scanning for media files in '{args.input_folder}'...")
    all_image_files, all_video_files = [], []
    for root, _, files in os.walk(args.input_folder):
        for file in files:
            ext = Path(file).suffix.lower()
            full_path = os.path.join(root, file)
            if full_path in processed_files:
                continue
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                all_image_files.append(full_path)
            elif ext in SUPPORTED_VIDEO_EXTENSIONS:
                all_video_files.append(full_path)

    images_to_process = all_image_files
    videos_to_process = all_video_files

    total_new_files = len(images_to_process) + len(videos_to_process)
    print(
        f"Found {len(images_to_process)} new images and {len(videos_to_process)} new videos to process.")
    if total_new_files == 0:
        print("No new files to process. Exiting.")
        return

    files_processed_this_session = 0
    nsfw_found_this_session = 0
    fieldnames = ["File Path", "Prediction", "NSFW Probability"]
    try:
        with open(args.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csvfile.seek(0, 2)
            if csvfile.tell() == 0:
                writer.writeheader()

            if images_to_process:
                print(
                    f"Processing {len(images_to_process)} images using {args.image_workers} parallel loader(s)...")
                dataset = ImageFileDataset(
                    file_paths=images_to_process,
                    transform=detector.get_transforms()
                )
                image_loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.image_workers,
                    collate_fn=safe_collate,
                    pin_memory=True if device.type == 'cuda' else False
                )

                # Increased batch_size based on user's screenshot
                pbar = tqdm(image_loader, desc="Processing Images",
                            total=len(image_loader))
                for tensor_batch, path_batch in pbar:
                    if not path_batch:
                        continue
                    tensor_batch = tensor_batch.to(device)
                    nsfw_probs = detector.predict_tensor_batch(tensor_batch)
                    for i, path in enumerate(path_batch):
                        prob = nsfw_probs[i]
                        pred = "NSFW" if prob >= args.threshold else "SFW"
                        if pred == "NSFW":
                            nsfw_found_this_session += 1
                        writer.writerow(
                            {"File Path": path, "Prediction": pred,
                                "NSFW Probability": f"{prob:.4f}"}
                        )
                    csvfile.flush()
                    files_processed_this_session += len(path_batch)

            if videos_to_process:
                use_parallel = args.workers != 0
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
    finally:
        end_time = time.time()
        total_nsfw_count = 0
        total_processed_count = 0
        if os.path.exists(args.output_file):
            try:
                with open(args.output_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        total_processed_count += 1
                        if row["Prediction"] == "NSFW":
                            total_nsfw_count += 1
            except Exception as e:
                print(f"\nCould not read final stats from output file: {e}")

        print("\n--- Detection Complete ---")
        print(
            f"Processed {files_processed_this_session} new files this session.")
        print(f"Total files in output: {total_processed_count}")
        print(
            f"Total NSFW files found in output: {total_nsfw_count} (threshold >= {args.threshold})")
        print(
            f"Processing time for this session: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
