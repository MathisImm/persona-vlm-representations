"""
Qwen3-VL wrapper: text generation (ratings) and layer-wise hidden states.

The image and the prompt text form a single user turn, image first, and the
processor's chat template is applied on top (no system message).
"""

import os
from typing import Optional, List, Dict, Union
from pathlib import Path

import torch
from PIL import Image


class VLMWrapper:
    """Loads Qwen3-VL and runs generation and hidden-state extraction."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_name: Hugging Face path of a Qwen3-VL checkpoint
            device: Device to load model on
            dtype: Model dtype
        """
        if "qwen3-vl" not in model_name.lower():
            raise ValueError(f"Only Qwen3-VL checkpoints are supported, got {model_name}")
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.model_path = model_name
        self._load_model()

    def _load_model(self):
        """Load the processor and model (transformers >= 4.57)."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        try:
            is_offline = os.environ.get("HF_HUB_OFFLINE") == "1"

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=is_offline,
            )
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=True,
                local_files_only=is_offline,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen3-VL model: {e}")

    def _inputs(self, image: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        """Chat-template the image + prompt turn and move it to the model's device."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, str, Path],
        pooling: str,
        prompt: str,
    ) -> torch.Tensor:
        """
        Pooled hidden state of the last decoder layer.

        Args:
            image: PIL Image or path to image
            pooling: Pooling over tokens ('mean', 'cls' = first, 'last')
            prompt: Prompt text

        Returns:
            Embedding tensor [hidden_dim]
        """
        hidden_states = self.get_hidden_states(image, prompt=prompt, layer_indices=[-1])
        last_hidden = list(hidden_states.values())[0]  # [1, seq_len, hidden_dim]

        if pooling == "mean":
            embedding = last_hidden.mean(dim=1).squeeze(0)
        elif pooling == "cls":
            embedding = last_hidden[:, 0, :].squeeze(0)
        elif pooling == "last":
            embedding = last_hidden[:, -1, :].squeeze(0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return embedding.cpu()

    @torch.no_grad()
    def get_hidden_states(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Hidden states of the embedding layer (0) and every decoder block (1..N).

        Args:
            image: PIL Image or path
            prompt: Prompt text
            layer_indices: Layer indices to extract (None = all)

        Returns:
            Dict mapping layer index to hidden states tensor [1, seq_len, hidden_dim]
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        outputs = self.model(**self._inputs(image, prompt), output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer

        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))

        result = {}
        for idx in layer_indices:
            # Handle negative indices
            actual_idx = idx if idx >= 0 else len(hidden_states) + idx
            if 0 <= actual_idx < len(hidden_states):
                result[idx] = hidden_states[actual_idx].cpu()

        return result

    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> str:
        """
        Generate a text response for an image and prompt (greedy by default).

        Args:
            image: PIL Image or path
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (unused when do_sample is False)
            do_sample: Whether to use sampling

        Returns:
            Generated text string
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self._inputs(image, prompt)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Decode only the generated tokens
        generated = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return generated.strip()

    def get_embedding_dim(self) -> int:
        """Hidden dimension of the language model."""
        return self.model.config.text_config.hidden_size

    def get_num_layers(self) -> int:
        """Number of decoder blocks of the language model."""
        return self.model.config.text_config.num_hidden_layers
