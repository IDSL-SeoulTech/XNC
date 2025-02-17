import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoConfig

def save_model_weights(model_path, save_dir, flatten=True):
    """
    Extract and save embedding-related parameters (embed_tokens, lm_head) from the specified model.
    If flatten=True, the tensor is flattened into a 1D array and the original shape information is saved along with it.
    """
    config = AutoConfig.from_pretrained(model_path)
    model_type = config.model_type.lower()

    if "vl" in model_type or "vlm" in model_path.lower():
        model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    os.makedirs(save_dir, exist_ok=True)
    saved_files = []

    for name, param in model.named_parameters():
        if "embed_tokens" in name.lower() or "lm_head" in name.lower():
            file_path = os.path.join(save_dir, f"{name.replace('.', '_')}.pt")
            tensor = param.data.cpu()
            if flatten:
                original_shape = tensor.shape
                flat_tensor = tensor.flatten()
                save_data = {"original_shape": original_shape, "data": flat_tensor}
                torch.save(save_data, file_path)
            else:
                torch.save(tensor, file_path)
            saved_files.append(file_path)
            print(f"Saved {name} to {file_path}")

    print("Saved files:")
    for file in saved_files:
        print(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save embedding weights from a model.')
    parser.add_argument('--model', type=str, default="HuggingFaceTB/SmolLM-135M", help='Model ID or path to load')
    parser.add_argument('--save_dir', type=str, default="/path/embedding_weight/SmolLM-135M", help='Directory to save weights')
    parser.add_argument('--flatten', action='store_true', help='Flatten weights before saving')

    args = parser.parse_args()
    save_model_weights(args.model, args.save_dir, flatten=args.flatten)
