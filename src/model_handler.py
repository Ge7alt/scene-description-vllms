from typing import List, Tuple
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM,
    LlavaForConditionalGeneration
)
from PIL.Image import Image
from omegaconf import DictConfig
from utils import setup_logger

logger = setup_logger()

def load_model(
    model_id: str, model_type: str, device: str, dtype_str: str
) -> Tuple[AutoProcessor, torch.nn.Module]:
    """
    Load a processor and model dynamically based on model_type.

    Args:
        model_id (str): Hugging Face model repo ID.
        model_type (str): One of ["blip", "blip2", "llava", "generic"].
        device (str): "cuda" or "cpu".
        dtype_str (str): torch dtype ("float16", "float32", etc).

    Returns:
        (processor, model)
    """
    dtype = torch.float16 if dtype_str == "float16" and device == "cuda" else torch.float32

    if model_type == "blip":
        processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map='auto'
        ).to(device)

    elif model_type == "blip2":
        processor = Blip2Processor.from_pretrained(model_id, use_fast=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map='auto'
        ).to(device)

    elif model_type == "llava":
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map='auto'
        ).to(device)

    else:  # generic fallback
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map='auto'
        ).to(device)

    return processor, model

def prepare_inputs(processor, image, prompt, model_type, device, dtype):
    """
    Normalize inputs across BLIP, BLIP2, LLaVA, Qwen-VL families.
    """

    # --- BLIP ---
    if model_type == "blip":
        return processor(image, text=prompt, return_tensors="pt").to(device, dtype)

    # --- BLIP2 ---
    elif model_type == "blip2":
        return processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)

    # --- LLaVA ---
    elif model_type == "llava":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]
        chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=chat_prompt, return_tensors="pt").to(device, dtype)
        return inputs

    # # --- Qwen-VL ---
    # elif model_type == "qwen":
    #     from qwen_vl_utils import process_vision_info

    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": image},
    #                 {"type": "text", "text": prompt},
    #             ],
    #         }
    #     ]
    #     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )
    #     return inputs.to(device)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")



def generate_captions(
    image: Image,
    model: torch.nn.Module,
    model_type: str,
    processor: AutoProcessor,
    generation_config: DictConfig,
    device: str,
) -> List[str]:
    """
    Generate captions for an image using a given model.
    Keeps generation arguments consistent across model families.
    """

    prompt = generation_config.prompt or ""

    inputs = prepare_inputs(
        processor, image, prompt,
        model_type=model_type,
        device=device,
        dtype=model.dtype,
    )

    # Unified generation call
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=generation_config.num_captions_per_image,
        max_new_tokens=generation_config.max_new_tokens,
        min_length=generation_config.min_length,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        repetition_penalty=generation_config.repetition_penalty,
    )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    logger.info(f"Raw generated texts: {generated_texts}")
    # Clean outputs
    # captions = []
    # for text in generated_texts:
    #     clean_text = text.replace(prompt, "") if prompt else text
    #     captions.append(clean_text.strip())

    # return captions
    captions = []
    for text in generated_texts:
        if model_type == "llava":
            # Split the output by "ASSISTANT:" and take the last part.
            parts = text.split("ASSISTANT:")
            if len(parts) > 1:
                clean_text = parts[-1].strip()
            else:
                # If for some reason the marker isn't there, use the whole text
                clean_text = text.strip()
        else:
            clean_text = text.replace(prompt, "") if prompt else text
    
        captions.append(clean_text)

    return captions
