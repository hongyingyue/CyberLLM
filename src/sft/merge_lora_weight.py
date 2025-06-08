import os
import shutil
import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--adapter_model_path", type=str, required=True, help="Path to adapter model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output model")
    parser.add_argument("--save_dtype", type=str, choices=['bf16', 'fp32', 'fp16'], 
                        default='fp32', help="In which dtype to save, fp32, bf16 or fp16.")
    args = parser.parse_args()
    name2dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, device_map='cpu', 
        trust_remote_code=True, torch_dtype=name2dtype[args.save_dtype]
    )
    model = PeftModel.from_pretrained(model, args.adapter_model_path, trust_remote_code=True)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_path, safe_serialization=False)

    shutil.copy(
        os.path.join(args.base_model_path, 'generation_config.json'), 
        os.path.join(args.output_path, 'generation_config.json')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'hy.tiktoken'), 
        os.path.join(args.output_path, 'hy.tiktoken')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'tokenizer_config.json'), 
        os.path.join(args.output_path, 'tokenizer_config.json')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'config.json'), 
        os.path.join(args.output_path, 'config.json')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'modeling.py'), 
        os.path.join(args.output_path, 'modeling.py')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'configuration.py'), 
        os.path.join(args.output_path, 'configuration.py')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'tokenization_hy.py'), 
        os.path.join(args.output_path, 'tokenization_hy.py')
    )

    print(f'Merged model weight is saved to {args.output_path}')