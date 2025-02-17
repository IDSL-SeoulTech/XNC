import torch
import argparse
import numpy as np

def xnc_transform(weight_data):
    """
    Transforms the saved weight data using XNC transformation.
    If the data was saved with flatten=True, weight_data will be a dictionary containing
    the keys "data" and "original_shape".
    """
    if isinstance(weight_data, dict) and "data" in weight_data:
        original_shape = weight_data["original_shape"]
        tensor = weight_data["data"]
    else:
        tensor = weight_data
        original_shape = tensor.shape

    np_array = tensor.numpy().flatten()
    uint16_array = np_array.view(np.uint16)

    result = np.empty_like(uint16_array)
    sign_mask = 0x8000
    invert_mask = 0x7800

    for i in range(len(uint16_array)):
        val = uint16_array[i] if i == 0 else np.bitwise_xor(uint16_array[i-1], uint16_array[i])
        if (val & sign_mask) != 0:
            val ^= invert_mask
        result[i] = val

    transformed_array = result.view(np.float16).reshape(original_shape)
    return torch.from_numpy(transformed_array)

def main():
    parser = argparse.ArgumentParser(description='Transforms a saved pt file using XNC transformation and saves it.')
    parser.add_argument('--input', type=str, default='/xnc/embedding_weight/SmolLM-135M/model_embed_tokens_weight.pt', help='Input pt file path')
    parser.add_argument('--output', type=str, default='/xnc/xnc_embedding_weight/SmolLM-135M/SmolLM-135M_embed_tokens_XNC.pt', help='Output pt file path')
    args = parser.parse_args()

    loaded_data = torch.load(args.input, map_location='cpu', weights_only=True)
    saved_files = []

    if isinstance(loaded_data, list):
        transformed_list = [xnc_transform(item) if isinstance(item, (torch.Tensor, dict)) else item for item in loaded_data]
        torch.save(transformed_list, args.output)
        saved_files.append(args.output)
    elif isinstance(loaded_data, dict):
        transformed_dict = {key: xnc_transform(value) if isinstance(value, (torch.Tensor, dict)) else value for key, value in loaded_data.items()}
        torch.save(transformed_dict, args.output)
        saved_files.append(args.output)
    elif isinstance(loaded_data, (torch.Tensor, dict)):
        torch.save(xnc_transform(loaded_data), args.output)
        saved_files.append(args.output)
    else:
        raise ValueError(f"Error: Unsupported data type {type(loaded_data)}")

    print("Saved files:")
    for file in saved_files:
        print(file)

if __name__ == '__main__':
    main()
