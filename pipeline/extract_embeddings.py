"""
Extract layer-wise VLM hidden states for a set of images and one prompt.

For every image, the hidden state of the final prompt token is stored for the
embedding layer and every decoder block (layers/layer_0 .. layer_36 for
Qwen3-VL-8B) in <output-dir>/<model_folder>/embeddings.h5.
Usually called once per persona by pipeline/run_persona_analysis.py.

Usage:
    python pipeline/extract_embeddings.py --model Qwen/Qwen3-VL-8B-Instruct --pooling last \
        --extract-layers all --image-ids-file configs/image_ids_100_with_hidden_states.txt \
        --output-dir out/persona_1 --persona "<description>" --prompt-condition original_persona
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse
import logging
from datetime import datetime

import numpy as np
import h5py
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paths import IMAGE_DIR, OUTPUT
from src.vlm_wrapper import VLMWrapper
from pipeline.prompt_conditions import (
    CONDITIONS,
    resolve_condition,
)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("extract_embeddings")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = output_dir / f"extract_embeddings_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def load_image_safely(image_path: Path) -> Optional[Image.Image]:
    """Load image with error handling."""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception:
        return None


def _pool(hidden: torch.Tensor, pooling: str) -> np.ndarray:
    """Pool a [1, seq_len, hidden_dim] hidden state over its tokens."""
    if pooling == "mean":
        return hidden.mean(dim=1).squeeze(0).numpy()
    elif pooling == "cls":
        return hidden[:, 0, :].squeeze(0).numpy()
    return hidden[:, -1, :].squeeze(0).numpy()


def extract_embeddings_streaming(
    model: VLMWrapper,
    image_paths: List[Path],
    output_path: Path,
    prompt: str,
    pooling: str = "last",
    layer_indices: Optional[List[int]] = None,
    extract_all_layers: bool = False,
    batch_size: int = 100,
    logger: Optional[logging.Logger] = None,
    resume: bool = True,
) -> Tuple[int, Path]:
    """
    Extract embeddings in batches and write them directly to HDF5.

    Supports resumption: if output_path exists, skips already-processed images.

    Args:
        model: VLM wrapper instance
        image_paths: List of image file paths
        output_path: Path to output HDF5 file
        prompt: Prompt text
        pooling: Pooling strategy for embeddings
        layer_indices: Which layers to extract
        extract_all_layers: If True, extract all layers
        batch_size: Number of images to process before writing to disk
        logger: Logger instance
        resume: If True, skip already-processed images (default True)

    Returns:
        Tuple of (number of processed images, output path)
    """
    do_layerwise = extract_all_layers or (layer_indices is not None and len(layer_indices) > 0)
    indices_to_pass = None if extract_all_layers else layer_indices

    # Check for existing file and get already-processed image IDs for resumption
    existing_ids = set()
    if resume and output_path.exists():
        try:
            with h5py.File(output_path, "r") as f:
                if "image_ids" in f:
                    existing_ids = set(
                        id.decode() if isinstance(id, bytes) else id
                        for id in f["image_ids"][:]
                    )
                    if logger:
                        logger.info(f"Resume mode: found {len(existing_ids)} already-processed images")
        except Exception as e:
            if logger:
                logger.warning(f"Could not read existing file for resume: {e}. Starting fresh.")
            existing_ids = set()

    # Filter out already-processed images
    if existing_ids:
        original_count = len(image_paths)
        image_paths = [p for p in image_paths if p.stem not in existing_ids]
        if logger:
            logger.info(f"Skipping {original_count - len(image_paths)} already-processed images")

        if not image_paths:
            if logger:
                logger.info("All images already processed!")
            return len(existing_ids), output_path

    # Process first image to get embedding dimensions
    first_emb = None
    first_layer_dims = {}
    initialization_failures = 0
    max_initialization_retries = 10

    for img_path in image_paths:
        img = load_image_safely(img_path)
        if img is None:
            continue
        try:
            first_emb = model.encode_image(img, pooling=pooling, prompt=prompt).numpy()
            if do_layerwise:
                hidden_states = model.get_hidden_states(img, prompt=prompt, layer_indices=indices_to_pass)
                for layer_idx, hidden in hidden_states.items():
                    first_layer_dims[f"layer_{layer_idx}"] = _pool(hidden, pooling).shape[0]
            break
        except Exception as e:
            initialization_failures += 1
            if logger:
                logger.warning(f"Failed to process {img_path.name} ({initialization_failures}/{max_initialization_retries}): {e}")

            if initialization_failures >= max_initialization_retries:
                if logger:
                    logger.error(f"Aborting after {max_initialization_retries} consecutive initialization failures.")
                break
            continue

    if first_emb is None:
        if logger:
            logger.error("Could not process any images to determine embedding dimensions")
        return 0, output_path

    emb_dim = first_emb.shape[0]
    n_images = len(image_paths)

    if logger:
        logger.info(f"Embedding dimension: {emb_dim}")
        logger.info(f"Processing {n_images} images in batches of {batch_size}")
        if first_layer_dims:
            logger.info(f"Layer dimensions: {list(first_layer_dims.keys())}")

    # Determine file mode: append if resuming and file exists, otherwise write
    file_mode = "a" if (resume and existing_ids and output_path.exists()) else "w"

    with h5py.File(output_path, file_mode) as f:
        if "embeddings" in f:
            # Resume case
            emb_ds = f["embeddings"]
            ids_ds = f["image_ids"]
            ci_ds = f["ci_scores"]
            trend_ds = f["trend_groups"]

            layer_datasets = {}
            if do_layerwise and "layers" in f:
                for layer_name in f["layers"].keys():
                    layer_datasets[layer_name] = f["layers"][layer_name]

            if logger:
                logger.info(f"Resuming: appending to existing file with {emb_ds.shape[0]} embeddings")
        else:
            # Last-layer embedding (the same values as layers/layer_<last>)
            emb_ds = f.create_dataset(
                "embeddings",
                shape=(0, emb_dim),
                maxshape=(None, emb_dim),
                dtype=np.float32,
                chunks=(min(batch_size, 100), emb_dim),
                compression="gzip",
                compression_opts=4,
            )

            dt = h5py.special_dtype(vlen=str)
            ids_ds = f.create_dataset(
                "image_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
            )

            # CI scores and trend groups are not known here (NaN / -1); the
            # analysis takes the CI scores from ci_scores_100_uniform.csv.
            ci_ds = f.create_dataset(
                "ci_scores",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float32,
            )
            trend_ds = f.create_dataset(
                "trend_groups",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
            )

            layer_datasets = {}
            if do_layerwise:
                layers_group = f.create_group("layers")
                for layer_name, dim in first_layer_dims.items():
                    layer_datasets[layer_name] = layers_group.create_dataset(
                        layer_name,
                        shape=(0, dim),
                        maxshape=(None, dim),
                        dtype=np.float32,
                        chunks=(min(batch_size, 100), dim),
                        compression="gzip",
                        compression_opts=4,
                    )

        # Process images in batches
        batch_embeddings = []
        batch_ids = []
        batch_layers = {k: [] for k in first_layer_dims}
        processed_count = 0

        for img_path in tqdm(image_paths, desc="Extracting embeddings"):
            img = load_image_safely(img_path)
            if img is None:
                if logger:
                    logger.warning(f"Failed to load image: {img_path}")
                continue

            try:
                emb = model.encode_image(img, pooling=pooling, prompt=prompt).numpy()
                batch_embeddings.append(emb)
                batch_ids.append(img_path.stem)

                if do_layerwise:
                    hidden_states = model.get_hidden_states(img, prompt=prompt, layer_indices=indices_to_pass)
                    for layer_idx, hidden in hidden_states.items():
                        layer_key = f"layer_{layer_idx}"
                        if layer_key in batch_layers:
                            batch_layers[layer_key].append(_pool(hidden, pooling))

                processed_count += 1

                # Write batch to disk when batch is full
                if len(batch_embeddings) >= batch_size:
                    _write_batch_to_hdf5(emb_ds, ids_ds, ci_ds, trend_ds, layer_datasets,
                                         batch_embeddings, batch_ids, batch_layers)
                    batch_embeddings = []
                    batch_ids = []
                    batch_layers = {k: [] for k in first_layer_dims}

            except Exception as e:
                if logger:
                    logger.warning(f"Error processing {img_path}: {e}")
                continue

        # Write remaining batch
        if batch_embeddings:
            _write_batch_to_hdf5(emb_ds, ids_ds, ci_ds, trend_ds, layer_datasets,
                                 batch_embeddings, batch_ids, batch_layers)

    if logger:
        logger.info(f"Processed {processed_count} images")

    return processed_count, output_path


def _write_batch_to_hdf5(
    emb_ds: h5py.Dataset,
    ids_ds: h5py.Dataset,
    ci_ds: h5py.Dataset,
    trend_ds: h5py.Dataset,
    layer_datasets: Dict[str, h5py.Dataset],
    batch_embeddings: List[np.ndarray],
    batch_ids: List[str],
    batch_layers: Dict[str, List[np.ndarray]],
) -> None:
    """Append a batch of embeddings to the HDF5 datasets."""
    n_new = len(batch_embeddings)
    n_existing = emb_ds.shape[0]

    emb_ds.resize(n_existing + n_new, axis=0)
    emb_ds[n_existing:] = np.stack(batch_embeddings)

    ids_ds.resize(n_existing + n_new, axis=0)
    ids_ds[n_existing:] = batch_ids

    ci_ds.resize(n_existing + n_new, axis=0)
    ci_ds[n_existing:] = np.full(n_new, np.nan, dtype=np.float32)

    trend_ds.resize(n_existing + n_new, axis=0)
    trend_ds[n_existing:] = np.full(n_new, -1, dtype=np.int32)

    for layer_name, layer_embs in batch_layers.items():
        if layer_embs and layer_name in layer_datasets:
            ds = layer_datasets[layer_name]
            ds.resize(n_existing + n_new, axis=0)
            ds[n_existing:] = np.stack(layer_embs)


def get_model_folder_name(model_name: str, pooling: str) -> str:
    """Folder name {model_shortname}_fp16_{pooling}, e.g. qwen3-vl-8b-instruct_fp16_last."""
    if "qwen3-vl-8b" in model_name.lower():
        short_name = "qwen3-vl-8b-instruct"
    else:
        short_name = model_name.split("/")[-1].lower().replace(".", "-")
    return f"{short_name}_fp16_{pooling}"


def main():
    parser = argparse.ArgumentParser(description="Extract layer-wise Qwen3-VL hidden states from images")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Hugging Face path of a Qwen3-VL checkpoint")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for embeddings")
    parser.add_argument("--pooling", type=str, default="last", choices=["mean", "cls", "last"],
                        help="Pooling over tokens (the paper uses the final prompt token: last)")
    parser.add_argument("--extract-layers", type=str, default="all",
                        help="all, last, first, or comma-separated layer indices (e.g., '0,6,12,-1')")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"],
                        help="Model dtype")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of images to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for streaming extraction (saves to disk every N images)")
    parser.add_argument("--persona", type=str, default=None,
                        help="Persona description to use for prompt construction")
    parser.add_argument("--use-baseline-prompt", action="store_true",
                        help="Use the baseline prompt (question only) without persona")
    parser.add_argument("--prompt-condition", type=str, default=None,
                        choices=sorted(CONDITIONS.keys()),
                        help="Prompt-condition id from the registry "
                             "(original_baseline, original_persona, D1, D2, P1-P4). "
                             "If unset, inferred from --use-baseline-prompt / --persona.")
    parser.add_argument("--image-ids-file", type=str, default=None,
                        help="File containing image IDs to process (one per line), for exact subset matching")

    args = parser.parse_args()

    image_dir = Path(args.image_dir) if args.image_dir else IMAGE_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("VLM Embedding Extraction")
    logger.info("=" * 60)

    # Get list of images to process
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    image_paths.sort()
    logger.info(f"Found {len(image_paths)} images in directory")

    if args.limit:
        image_paths = image_paths[:args.limit]
        logger.info(f"Limited to {len(image_paths)} images")

    # Filter by explicit image IDs file if provided
    if args.image_ids_file:
        ids_file = Path(args.image_ids_file)
        if ids_file.exists():
            with open(ids_file, 'r') as f:
                explicit_ids = set(line.strip() for line in f if line.strip())
            image_paths = [p for p in image_paths if p.stem in explicit_ids]
            logger.info(f"Filtered to {len(image_paths)} images from --image-ids-file ({ids_file})")
        else:
            logger.warning(f"Image IDs file {args.image_ids_file} not found. Skipping explicit filter.")

    if not image_paths:
        logger.error("No images found!")
        return 1

    # Parse layer indices
    layer_indices = None
    extract_all_layers = False
    if args.extract_layers.lower() == "all":
        extract_all_layers = True
        logger.info("Extracting ALL layers")
    elif args.extract_layers.lower() == "last":
        layer_indices = [-1]
        logger.info("Extracting LAST layer only")
    elif args.extract_layers.lower() == "first":
        layer_indices = [0]
        logger.info("Extracting FIRST layer only")
    else:
        try:
            layer_indices = [int(x.strip()) for x in args.extract_layers.split(",")]
            logger.info(f"Extracting layers: {layer_indices}")
        except ValueError:
            logger.error(f"Invalid extract-layers format: {args.extract_layers}")
            return 1

    # Prompt from the registry; there is no default prompt.
    condition_id = resolve_condition(
        explicit=args.prompt_condition,
        use_baseline_prompt=args.use_baseline_prompt,
        has_persona=bool(args.persona),
    )
    condition = CONDITIONS[condition_id]
    prompt = condition.build(args.persona if condition.requires_persona else None)
    if condition.requires_persona:
        logger.info(f"Using prompt condition '{condition_id}'. Persona: {args.persona[:50]}...")
    else:
        logger.info(f"Using prompt condition '{condition_id}' (no persona)")

    # Initialize model
    logger.info(f"Loading model: {args.model}")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        model = VLMWrapper(model_name=args.model, device=args.device, dtype=dtype_map[args.dtype])
        logger.info("Model loaded successfully")
        logger.info(f"  Embedding dim: {model.get_embedding_dim()}")
        logger.info(f"  Num layers: {model.get_num_layers()}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    logger.info("Starting embedding extraction...")
    start_time = datetime.now()

    model_folder = get_model_folder_name(args.model, args.pooling)
    model_output_dir = output_dir / model_folder
    model_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {model_folder}")
    output_path = model_output_dir / "embeddings.h5"

    logger.info(f"Using streaming mode (batch size: {args.batch_size})")
    n_processed, _ = extract_embeddings_streaming(
        model=model,
        image_paths=image_paths,
        output_path=output_path,
        prompt=prompt,
        pooling=args.pooling,
        layer_indices=layer_indices,
        extract_all_layers=extract_all_layers,
        batch_size=args.batch_size,
        logger=logger,
    )

    # Extraction settings and prompt, as in the archived files
    with h5py.File(output_path, "a") as f:
        meta_group = f["metadata"] if "metadata" in f else f.create_group("metadata")
        meta_group.attrs["model"] = args.model
        meta_group.attrs["pooling"] = args.pooling
        meta_group.attrs["device"] = args.device
        meta_group.attrs["dtype"] = args.dtype
        meta_group.attrs["quantize"] = "none"
        meta_group.attrs["sampling_suffix"] = "none"
        meta_group.attrs["n_images"] = n_processed
        meta_group.attrs["extraction_date"] = datetime.now().isoformat()
        meta_group.attrs["layers_extracted"] = "all" if extract_all_layers else str(layer_indices)
        meta_group.attrs["streaming_batch_size"] = args.batch_size
        meta_group.attrs["extraction_type"] = "llm"
        meta_group.attrs["prompt_type"] = "baseline" if args.use_baseline_prompt else ("persona" if args.persona else "default")
        meta_group.attrs["prompt_condition"] = condition_id
        if args.persona:
            meta_group.attrs["persona"] = args.persona

    logger.info(f"Saved LLM embeddings to {output_path}")
    logger.info(f"Extraction completed in {datetime.now() - start_time}")
    logger.info(f"  Processed: {n_processed} images")
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
