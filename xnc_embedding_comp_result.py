import torch
import os
import argparse
import numpy as np

def read_fp16_tensor_file(input_file):
    """
    Loads an FP16 tensor and flattens it into a 1D array.
    """
    tensor = torch.load(input_file, map_location='cpu', weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Loaded object is not a torch.Tensor.")
    return tensor.view(-1)

def fp16_to_bits(tensor):
    """
    Converts a 1D FP16 tensor into a 2D binary representation tensor.
    Each row represents an element as a 16-bit binary list.
    """
    np_arr = tensor.cpu().numpy().view(np.uint16)
    bits = np.array([[int(bit) for bit in np.binary_repr(x, width=16)] for x in np_arr], dtype=np.int16)
    return torch.from_numpy(bits)

def read_tensor_in_chunks(tensor, chunk_size=32):
    """
    Reads a 2D tensor (each row representing a 16-bit block) in chunks of the specified size.
    """
    for i in range(0, tensor.shape[0], chunk_size):
        yield tensor[i:i + chunk_size]

def determine_compression(blocks):
    """
    Determines the compression type based on the structure of 32-bit blocks.
    """
    zero_columns = {
        2: (blocks[:, 1] == blocks[:, 0]).all(),
        14: (blocks[:, 13] == 0).all(),
        15: (blocks[:, 14] == 0).all(),
        16: (blocks[:, 15] == 0).all()
    }
    num_zero_columns = sum(zero_columns.values())
    return (12, '1111') if num_zero_columns == 4 else (16, '0000')

def compress_tensor(blocks, metadata):
    """
    Applies compression rules to each block (row) based on metadata.
    """
    compressed_blocks = []
    for row in blocks:
        if metadata == '1111':
            if row[:5].eq(torch.tensor([1, 1, 1, 1, 1], dtype=torch.int16)).all() or \
               row[:5].eq(torch.tensor([0, 0, 0, 0, 0], dtype=torch.int16)).all():
                compressed_row = torch.cat((row[:1], row[5:13]))  # 9-bit compression
            else:
                compressed_row = torch.cat((row[:1], row[2:13]))  # 12-bit compression
        else:
            compressed_row = row  # Keep 16-bit format
        compressed_blocks.append(compressed_row)
    return compressed_blocks

def save_compressed_txt(output_file, layer_data):
    """
    Saves compression statistics to a text file.
    """
    with open(output_file, "w") as file:
        for layer_number, stats in sorted(layer_data.items()):
            compression_ratio = stats["Uncompressed bits"] / stats["Compressed bits"]
            compression_percent = (1 - stats["Compressed bits"] / stats["Uncompressed bits"]) * 100
            file.write(f"Uncompressed bits: {stats['Uncompressed bits']} bits\n")
            file.write(f"Compressed bits: {stats['Compressed bits']} bits\n")
            file.write(f"Compression ratio: {compression_ratio:.4f}, Compression percent: {compression_percent:.4f}%\n")
            file.write(f"Metadata bits: {stats['Metadata bits']} bits\n")

def compress_file(input_file, output_txt):
    """
    Loads a 1D FP16 tensor, converts elements into binary representation,
    analyzes compression by chunks, and saves results.
    """
    tensor = read_fp16_tensor_file(input_file)
    bits_tensor = fp16_to_bits(tensor)
    layer_data = {}
    total_uncompressed_bits = 0
    total_compressed_bits = 0
    total_metadata_bits = 0
    
    for idx, blocks in enumerate(read_tensor_in_chunks(bits_tensor)):
        bits_per_block, metadata = determine_compression(blocks)
        compressed_blocks = compress_tensor(blocks, metadata)
        total_uncompressed_bits += blocks.numel()
        chunk_compressed_bits = sum(block.numel() for block in compressed_blocks)
        total_compressed_bits += chunk_compressed_bits + len(compressed_blocks)
    
    layer_data[0] = {
        "Uncompressed bits": total_uncompressed_bits,
        "Compressed bits": total_compressed_bits,
        "Metadata bits": total_metadata_bits
    }
    
    save_compressed_txt(output_txt, layer_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saves compression statistics of a 1D FP16 tensor to a text file.')
    parser.add_argument('--input_pt', type=str, default='/xnc/xnc_embedding_weight/SmolLM-135M/SmolLM-135M_embed_tokens_XNC.pt', help='Input pt file path')
    parser.add_argument('--output_txt', type=str, default='/xnc/result/SmolLM-135M_text_model_embed_tokens_XNC_compressed_output.txt', help='Output txt file path')
    args = parser.parse_args()
    
    compress_file(args.input_pt, args.output_txt)
