import argparse

def read_fp16_binary_file_in_chunks(input_file, chunk_size=32):
    """Reads the binary data from the input file in chunks of 64B."""
    with open(input_file, 'r') as file:
        while True:
            chunk = [file.readline().strip() for _ in range(chunk_size)]
            if not any(chunk):
                break
            yield chunk

def determine_compression(blocks):
    """
    Checks the 2nd, 14th, 15th, and 16th bits vertically across the 64B.
    Returns the number of bits to compressible block and the corresponding metadata.
    """
    zero_columns = {2: True, 14: True, 15: True, 16: True}
    for block in blocks:
        if block[1] != block[0]:
            zero_columns[2] = False
        if block[13] != '0':
            zero_columns[14] = False
        if block[14] != '0':
            zero_columns[15] = False
        if block[15] != '0':
            zero_columns[16] = False
    num_zero_columns = sum(zero_columns.values())
    return (12, '1') if num_zero_columns == 4 else (16, '0')

def check_zero_columns(block):
    return block[1] == block[0] and block[13] == '0' and block[14] == '0' and block[15] == '0'

def compress_block_remove_specific_columns(blocks, metadata):
    """
    Removes columns based on the metadata and applies special compression rules:
    - If metadata == '1', compress values starting with '11111' or '00000' to 9 bits (8 bits + 1 bit metadata),
      and keep other values at 12 bits + 1 bit metadata.
    - If metadata == '0', apply zero_columns condition:
        - If the block passes the zero_columns condition, compress it to 12 bits.
        - Otherwise, keep the block as 16 bits.
    """
    compressed_blocks = []
    global count_9bit, count_12bit, count_16bit
    for block in blocks:
        if metadata == '1':
            if block.startswith('11111') or block.startswith('00000'):
                compressed_block = block[0] + block[5:13]
                count_9bit += 1
            else:
                compressed_block = block[0] + block[2:13]
                count_12bit += 1
        elif metadata == '0':
            if check_zero_columns(block):
                compressed_block = block[0] + block[2:13]
                count_12bit += 1
            else:
                compressed_block = block
                count_16bit += 1
        compressed_blocks.append(compressed_block)
    return compressed_blocks

def compress_file(input_file, output_file):
    compressed_data = []
    metadata = []

    uncompressed_bits = 0
    compressed_bits = 0
    metadata_bits = 0
    layer_data = {}

    global count_9bit, count_12bit, count_16bit
    count_9bit = count_12bit = count_16bit = 0

    chunk_generator = read_fp16_binary_file_in_chunks(input_file)

    for blocks in chunk_generator:
        uncompressed_bits += 16 * len(blocks)
        bits_per_block, block_metadata = determine_compression(blocks)
        compressed_blocks = compress_block_remove_specific_columns(blocks, block_metadata)
        compressed_data.extend(compressed_blocks)
        metadata.append(block_metadata)

        if block_metadata == '0':
            for block in blocks:
                metadata_bits += 1
        else:
            metadata_bits += len(blocks)
        metadata_bits += 1

    compressed_bits = metadata_bits + (count_9bit * 9) + (count_12bit * 12) + (count_16bit * 16)

    total_blocks = count_9bit + count_12bit + count_16bit
    layer_data[0] = {
        "Uncompressed bits": uncompressed_bits,
        "Compressed bits": compressed_bits,
        "Metadata bits": metadata_bits,
        "9-bit blocks": count_9bit,
        "12-bit blocks": count_12bit,
        "16-bit blocks": count_16bit,
        "Total blocks": total_blocks,
    }

    with open(output_file, "w") as output_file:
        for layer_number, stats in sorted(layer_data.items()):
            compression_ratio = stats["Uncompressed bits"] / stats["Compressed bits"]
            compression_percent = (1 - stats["Compressed bits"] / stats["Uncompressed bits"]) * 100
            output_file.write(f"  Uncompressed bits: {stats['Uncompressed bits']} bits\n")
            output_file.write(f"  Compressed bits: {stats['Compressed bits']} bits\n")
            output_file.write(f"  Compression ratio: {compression_ratio:.4f}, Compression percent: {compression_percent:.4f}%\n")
            output_file.write(f"  Metadata bits: {stats['Metadata bits']} bits\n")
            
            if stats["Total blocks"] > 0:
                ratio_9bit = (stats["9-bit blocks"] / stats["Total blocks"]) * 100
                ratio_12bit = (stats["12-bit blocks"] / stats["Total blocks"]) * 100
                ratio_16bit = (stats["16-bit blocks"] / stats["Total blocks"]) * 100
                output_file.write(f"  9-bit blocks: {stats['9-bit blocks']} ({ratio_9bit:.2f}%)\n")
                output_file.write(f"  12-bit blocks: {stats['12-bit blocks']} ({ratio_12bit:.2f}%)\n")
                output_file.write(f"  16-bit blocks: {stats['16-bit blocks']} ({ratio_16bit:.2f}%)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress floating point 16-bit binary weights.')
    parser.add_argument('--xnc_comp_weight', type=str, default="/XNC/weight/Llama-3.2-1B/xnc_model_embed_tokens_weight.txt", help='Path to the original weight file.')
    parser.add_argument('--result_dir', type=str, default="/XNC/result/Llama-3.2-1B_comp_result.txt", help='Path to save the compressed output file.')
    args = parser.parse_args()
    compress_file(args.xnc_comp_weight, args.result_dir)
