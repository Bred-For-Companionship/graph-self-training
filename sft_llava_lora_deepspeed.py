import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser, LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk


@dataclass
class ModelArgs:
    MODEL_ID: str = field(default="llava-hf/llava-v1.6-mistral-7b-hf", metadata={"help": "Model identifier"})


@dataclass
class DataArgs:
    TRAIN_DATA_PATH: str = field(default="llava1.5_instr_6k", metadata={"help": "HF ds"})


@dataclass
class LoraArgs:
    r: int = field(default=128, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=256, metadata={"help": "LoRA alpha"})


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


parser = HfArgumentParser((LoraArgs, ModelArgs, DataArgs))
lora_args, model_args, data_args = parser.parse_args_into_dataclasses()

processor = LlavaNextProcessor.from_pretrained(model_args.MODEL_ID)
processor.tokenizer.padding_side = "right"
model = LlavaNextForConditionalGeneration.from_pretrained(model_args.MODEL_ID,
                                                          torch_dtype=torch.float16,
                                                          low_cpu_mem_usage=True,
                                                          load_in_4bit=False,
                                                          attn_implementation="flash_attention_2")

lora_config = LoraConfig(
    r=lora_args.r,
    lora_alpha=lora_args.lora_alpha,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = get_peft_model(model, lora_config)

ds = load_from_disk(data_args.TRAIN_DATA_PATH)








