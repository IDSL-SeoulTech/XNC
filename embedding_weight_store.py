import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig
import os
import numpy as np
import argparse

def save_model_weights(model_path, save_dir, flatten=False):
    """
    Extract and save embedding-related parameters (embed_tokens, lm_head) from the specified model.
    """
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type.lower()

    if "vl" in model_type or "vlm" in model_path.lower():
        model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    os.makedirs(save_dir, exist_ok=True)
    
    for name, param in model.named_parameters():
        if "embed_tokens" in name.lower() or "lm_head" in name.lower():
            file_path = os.path.join(save_dir, f"{name.replace('.', '_')}.txt")
            with open(file_path, 'w') as file:
                weights = param.data.cpu().numpy()  
                
                if param.dtype == torch.float16:                
                    binary_weights = [format(int.from_bytes(np.float16(x).tobytes(), 'little'), '016b') for x in weights.flatten()]
                else:
                    binary_weights = [format(int.from_bytes(np.float32(x).tobytes(), 'little'), '032b') for x in weights.flatten()]
                
                weight_str = "\n".join(binary_weights)
                file.write(weight_str)
    
    print(f"All embedding parameters have been saved as text files in: {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save embedding weights from a model.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model ID or path to load')
    parser.add_argument('--save_dir', type=str, default="/XNC/weight/Llama-3.2-1B", help='Directory to save weights')
    
    args = parser.parse_args()
    save_model_weights(args.model, args.save_dir)