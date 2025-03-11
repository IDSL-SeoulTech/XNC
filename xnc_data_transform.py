import os
import argparse

# Custom binary_data_save function
def binary_data_save(output_file, data_block):
    """Saves a single block of binary data to the specified file."""
    with open(output_file, 'a') as f:
        f.write(data_block + '\n')

def read_fp16_binary_file_in_chunks(input_file, chunk_size=32):
    """Reads the binary data from the input file in chunks of 64B."""
    with open(input_file, 'r') as file:
        while True:
            chunk = [file.readline().strip() for _ in range(chunk_size)]
            if not any(chunk):
                break
            yield chunk

def not_bit_MSB_with_bits_2_to_5(chunk):
    """Inverts bits 2 to 5 if the MSB is 1 for each row."""
    result = []
    for binary_str in chunk:
        bits = list(binary_str)
        if bits[0] == '1':
            for i in range(1, 5):
                bits[i] = str(1 - int(bits[i]))
        result.append(''.join(bits))
    return result

def xor_adjacent_rows_in_chunk(chunk):
    """Performs XOR on adjacent rows in the chunk."""
    chunk_size = len(chunk)
    if chunk_size == 0:
        return None
    
    result_rows = [chunk[0]]
    for i in range(1, chunk_size):
        row1 = list(map(int, list(chunk[i - 1])))
        row2 = list(map(int, list(chunk[i])))
        xor_row = [row1[j] ^ row2[j] for j in range(len(row1))]
        result_rows.append(''.join(map(str, xor_row)))
    
    return result_rows

def apply_xor_bit_to_result_rows(xor_rows):
    """Applies not_bit_MSB_with_bits_2_to_5 on the XORed rows."""
    return not_bit_MSB_with_bits_2_to_5(xor_rows)

def save_compressed_data_block_by_block(input_file, output_file):
    """Processes the input file chunk by chunk, XORs adjacent rows, applies xor_bit_3, and saves the result."""
    for chunk in read_fp16_binary_file_in_chunks(input_file):
        xor_chunk = xor_adjacent_rows_in_chunk(chunk)
        if xor_chunk:
            final_chunk = apply_xor_bit_to_result_rows(xor_chunk)
            for row in final_chunk:
                binary_data_save(output_file, row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and compress floating point binary weights.')
    parser.add_argument('--original_weight', type=str, default="/XNC/weight/Llama-3.2-1B/model_embed_tokens_weight.txt", help='Path to the original weight file.')
    parser.add_argument('--xnc_comp_weight', type=str, default="/XNC/weight/Llama-3.2-1B/xnc_model_embed_tokens_weight.txt", help='Path to save the processed compressed weight file.')
    args = parser.parse_args()
    
    if os.path.exists(args.xnc_comp_weight):
        os.remove(args.xnc_comp_weight)
    
    save_compressed_data_block_by_block(args.original_weight, args.xnc_comp_weight)
