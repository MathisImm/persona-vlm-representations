import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import IMAGE_DIR
from src.vlm_wrapper import VLMWrapper
from pipeline.prompt_conditions import CONDITIONS

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_text")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File
    fh = logging.FileHandler(output_dir / "extraction.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="Extract textual responses from VLM for personas")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face path of a Qwen3-VL checkpoint")
    parser.add_argument("--image-ids-file", type=str, required=True, help="File with image IDs")
    parser.add_argument("--personas-config", type=str, required=True, help="JSON file with persona descriptions")
    parser.add_argument("--image-dir", type=str, default=str(IMAGE_DIR), help="Directory with images (default: $PERSONA_IMAGES)")
    parser.add_argument("--output-dir", type=str, default="results/persona_text", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--dtype", type=str, default="float16", help="Precision (float16, float32, bfloat16)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument(
        "--prompt-condition",
        type=str,
        default="original_persona",
        choices=sorted(CONDITIONS.keys()),
        help="Prompt-condition id from prompt_conditions.CONDITIONS. The 'baseline' "
             "entry inside the personas JSON always uses 'original_baseline' "
             "regardless of this flag (so existing data is reproducible). For "
             "no-persona conditions (D1, D2), pass --prompt-condition D1/D2 with a "
             "single-entry personas JSON.",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    
    logger.info("Starting Persona Text Extraction")
    logger.info(f"Model: {args.model}")
    
    # Load Image IDs
    with open(args.image_ids_file, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(image_ids)} image IDs")
    
    # Load Personas
    with open(args.personas_config, "r") as f:
        personas_dict = json.load(f)
    logger.info(f"Loaded {len(personas_dict)} personas")
    
    # Initialize Model
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    model = VLMWrapper(model_name=args.model, device=args.device, dtype=dtype_map[args.dtype])
    
    results = []
    
    image_dir = Path(args.image_dir)
    
    # Loop images
    for img_id in tqdm(image_ids, desc="Images"):
        img_path = image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            # Try search other extensions
            possible = list(image_dir.glob(f"{img_id}.*"))
            if possible:
                img_path = possible[0]
            else:
                logger.warning(f"Image {img_id} not found in {image_dir}")
                continue
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_id}: {e}")
            continue
            
        # Loop personas
        for p_id, p_info in personas_dict.items():
            description = p_info.get("description", "")

            # The 'baseline' entry inside a personas JSON has historically used
            # the no-persona prompt — preserve that behavior so existing CSVs
            # remain reproducible. All other entries use the requested condition.
            if p_id == "baseline" or not description:
                condition_id = "original_baseline"
                prompt = CONDITIONS[condition_id].build()
            else:
                condition_id = args.prompt_condition
                condition = CONDITIONS[condition_id]
                if condition.requires_persona:
                    prompt = condition.build(description)
                else:
                    prompt = condition.build()

            try:
                response_text = model.generate(image, prompt, max_new_tokens=args.max_new_tokens)

                results.append({
                    "image_id": img_id,
                    "persona_id": p_id,
                    "persona_name": p_info.get("name", p_id),
                    "prompt_condition": condition_id,
                    "raw_response": response_text
                })

            except Exception as e:

                logger.error(f"Error generating response for img {img_id}, persona {p_id}: {e}")
                
    # Save results
    output_path = output_dir / "persona_text_outputs.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(results)} responses to {output_path}")
    
    # Also save as JSON for better structure
    output_json = output_dir / "persona_text_outputs.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {output_json}")

if __name__ == "__main__":
    main()
