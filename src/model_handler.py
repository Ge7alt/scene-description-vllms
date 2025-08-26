from typing import List, Tuple
import torch
from transformers import (
    BitsAndBytesConfig, PreTrainedModel, 
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq,
    LlavaForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
)
from PIL.Image import Image
from omegaconf import DictConfig
from utils import setup_logger

logger = setup_logger()

def load_model(
    model_id: str, model_type: str, device: str, dtype_str: str
) -> Tuple[AutoProcessor, PreTrainedModel]:
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
    dtype = torch.float32 # default
    if device == "cuda" and torch.cuda.is_available():
        if dtype_str == "bfloat16":
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                logger.info("Using bfloat16 precision on CUDA.")
            else:
                logger.warning("bfloat16 is not supported on this device, falling back to float16.")
                dtype = torch.float16
        elif dtype_str == "float16":
            dtype = torch.float16
    try:
        if model_type == "blip":
            processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
            model = BlipForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, 
                device_map='auto'
            )

        elif model_type == "blip2":
            processor = Blip2Processor.from_pretrained(model_id, use_fast=True)
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, 
                device_map='auto'
            ).to(device)

        elif model_type == "llava":
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, 
                device_map='auto'
            )

        elif model_type == "qwen":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True,)
            processor = processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, quantization_config=bnb_config, 
                device_map='auto',
                # attn_implementation="flash_attention_2"
            )

        elif model_type == "smolvlm":
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, torch_dtype=dtype, 
                device_map='auto'
            )

        else:  # generic fallback
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id, torch_dtype=dtype, 
                device_map='auto'
            )

        return processor, model

    except Exception as e:
        logger.error(f"Error loading model {model_id} of type {model_type}: {e}")
        raise e

def prepare_inputs(processor, image, prompt, model_type, device, dtype):
    """
    Normalize inputs across BLIP, BLIP2, LLaVA, SmolVLM Qwen-VL families.
    """
    try:
        if model_type in ["llava", "smolvlm"]:
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

        elif model_type == "qwen":
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
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            return inputs.to(device)

        else:
            # Generic processor call
            return processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
            # raise ValueError(f"Unsupported model_type: {model_type}")
    except Exception as e:
        logger.error(f"Error preparing inputs for model type {model_type}: {e}")
        raise e


def generate_captions(
    image: Image,
    model: torch.nn.Module,
    model_type: str,
    processor: AutoProcessor,
    generation_config: DictConfig,
    device: str,
) -> Tuple[List[str], int]:
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
    input_token_length = inputs.input_ids.shape[1]
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

    if model_type == "qwen":
        # Qwen-VL generates some extra tokens at the end; trim to max_new_tokens
        generated_ids_trimmed = [
             out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids
        ]

        generated_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    else:
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # logger.info(f"Raw generated texts: {generated_texts}")
    # Clean outputs
    # captions = []
    # for text in generated_texts:
    #     clean_text = text.replace(prompt, "") if prompt else text
    #     captions.append(clean_text.strip())
    num_new_tokens = len(generated_ids[0]) - input_token_length

    # return captions
    captions = []
    # breakpoint()
    for text in generated_texts:
        if model_type == "llava":
            # Split the output by "ASSISTANT:" and take the last part.
            parts = text.split("ASSISTANT:")
            if len(parts) > 1:
                clean_text = parts[-1].strip()
        elif model_type == "smolvlm":
            parts = text.split("Assistant:")
            clean_text = parts[-1].strip()

        else:
            clean_text = text.replace(prompt, "") if prompt else text
    
        captions.append(clean_text)

    return captions, num_new_tokens
