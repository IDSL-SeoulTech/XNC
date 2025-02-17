import torch
import os
import argparse
import numpy as np

def read_fp16_tensor_file(input_file):
    """Loads an FP16 tensor and reshapes it into a 1D tensor."""
    tensor = torch.load(input_file, map_location='cpu', weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("The loaded object is not a torch.Tensor.")
    return tensor.view(-1)  # Reshape to 1D

def fp16_to_bits(tensor):
    """
    Converts a 1D FP16 tensor into a 2D tensor with 16-bit binary representation.
    Each row represents an element as a list of 16-bit (0,1) values.
    """
    # Convert FP16 tensor to numpy array and view it as uint16
    np_arr = tensor.cpu().numpy().view(np.uint16)
    # Convert each element to a 16-bit binary string, then into a list of 0s and 1s
    bits = np.array([[int(bit) for bit in np.binary_repr(x, width=16)] for x in np_arr], dtype=np.int16)
    return torch.from_numpy(bits)

def read_tensor_in_chunks(tensor, chunk_size=32):
    """Reads a 2D tensor (each row is a 16-bit block) in chunks of specified size."""
    for i in range(0, tensor.shape[0], chunk_size):
        yield tensor[i:i + chunk_size]

def determine_compression(blocks):
    """
    Determines compression type based on specific bit columns of 32-bit blocks.
    blocks: tensor of shape (n, 16)
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
    Applies compression rules to each row based on metadata.
    Each row represents a 16-bit block, and the result varies in length.
    Returns a list of compressed blocks.
    """
    compressed_blocks = []
    for row in blocks:
        if metadata == '1111':
            if row[0:5].eq(torch.tensor([1, 1, 1, 1, 1], dtype=torch.int16)).all() or \
               row[0:5].eq(torch.tensor([0, 0, 0, 0, 0], dtype=torch.int16)).all():
                compressed_row = torch.cat((row[:1], row[5:13]))  # 9-bit compression
            else:
                compressed_row = torch.cat((row[:1], row[2:13]))  # 12-bit compression
        else:
            if row[0:5].eq(torch.tensor([1, 1, 1, 1, 1], dtype=torch.int16)).all() or \
               row[0:5].eq(torch.tensor([0, 0, 0, 0, 0], dtype=torch.int16)).all():
                compressed_row = torch.cat((row[:1], row[5:13]))  # 9-bit compression
            else:
                compressed_row = row  # Keep as 16-bit
        compressed_blocks.append(compressed_row)
    return compressed_blocks  # Return as a list

def save_compressed_txt(output_file, layer_data):
    """Saves compression statistics as a text file."""
    with open(output_file, "w") as file:
        for layer_number, stats in sorted(layer_data.items()):
            compression_ratio = stats["Uncompressed bits"] / stats["Compressed bits"]
            compression_percent = (1 - stats["Compressed bits"] / stats["Uncompressed bits"]) * 100
            file.write(f"  Uncompressed bits: {stats['Uncompressed bits']} bits\n")
            file.write(f"  Compressed bits: {stats['Compressed bits']} bits\n")
            file.write(f"  Compression ratio: {compression_ratio:.4f}, Compression percent: {compression_percent:.4f}%\n")
            file.write(f"  Metadata bits: {stats['Metadata bits']} bits\n")

            if stats["Total blocks"] > 0:
                ratio_9bit = (stats["9-bit blocks"] / stats["Total blocks"]) * 100
                ratio_12bit = (stats["12-bit blocks"] / stats["Total blocks"]) * 100
                ratio_16bit = (stats["16-bit blocks"] / stats["Total blocks"]) * 100

                file.write(f"  9-bit blocks: {stats['9-bit blocks']} ({ratio_9bit:.2f}%)\n")
                file.write(f"  12-bit blocks: {stats['12-bit blocks']} ({ratio_12bit:.2f}%)\n")
                file.write(f"  16-bit blocks: {stats['16-bit blocks']} ({ratio_16bit:.2f}%)\n")

def compress_file(input_file, output_txt):
    """
    Loads a 1D FP16 tensor, converts each element into a 16-bit binary representation,
    analyzes compression for each chunk (e.g., 32-block size), and saves results to a text file.
    """
    tensor = read_fp16_tensor_file(input_file)  # Load 1D FP16 tensor
    bits_tensor = fp16_to_bits(tensor)  # Convert to (N, 16) tensor
    layer_data = {}
    total_uncompressed_bits = 0
    total_compressed_bits = 0
    total_metadata_bits = 0
    count_9bit, count_12bit, count_16bit = 0, 0, 0
    
    for idx, blocks in enumerate(read_tensor_in_chunks(bits_tensor)):
        bits_per_block, metadata = determine_compression(blocks)
        compressed_blocks = compress_tensor(blocks, metadata)  # Returns list
        total_uncompressed_bits += blocks.numel()
        chunk_compressed_bits = sum(block.numel() for block in compressed_blocks)
        total_compressed_bits += chunk_compressed_bits + len(compressed_blocks)
        
        for block in compressed_blocks:
            if block.numel() == 9:
                count_9bit += 1
            elif block.numel() == 12:
                count_12bit += 1
            elif block.numel() == 16:
                count_16bit += 1
        
    layer_data[0] = {
        "Uncompressed bits": total_uncompressed_bits,
        "Compressed bits": total_compressed_bits,
        "Metadata bits": total_metadata_bits,
        "9-bit blocks": count_9bit,
        "12-bit blocks": count_12bit,
        "16-bit blocks": count_16bit,
        "Total blocks": count_9bit + count_12bit + count_16bit,
    }
    
    save_compressed_txt(output_txt, layer_data)
    print(f"Compression statistics saved to {output_txt}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saves compression statistics of a 1D FP16 tensor to a text file.')
    parser.add_argument('--input_pt', type=str, default='/xnc/xnc_embedding_weight/SmolLM-135M/SmolLM-135M_embed_tokens_XNC.pt', help='Input pt file path')
    parser.add_argument('--output_txt', type=str, default='/xnc/result/SmolLM-135M_embed_tokens_XNC_result', help='Output txt file path')
    args = parser.parse_args()
    
    compress_file(args.input_pt, args.output_txt)
