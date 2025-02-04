# XNC: XNC: XOR and NOT-Based Lossless Compression for Optimizing Unquantized Embedding Layers in Large Language Models
![image](https://github.com/user-attachments/assets/bb39dda5-b5c8-4192-a740-936a5417ab63)

# Abstart
Although 4-bit quantized small LLMs have been proposed recently, many studies have retained FP16 precision for embedding layers, as they constitute a relatively small proportion of the overall model in existing LLMs and suffer from severe accuracy degradation when quantized. However, in quantized small LLMs, the embedding layer accounts for a substantial proportion of the total model parameters, necessitating its compression. Since embedding layers are sensitive to approximation, lossless compression is more desirable than lossy compression methods such as quantization. While existing lossless compression methods efficiently compress patterns such as zeros, narrow values, or frequently occurring values, embedding layers typically lack these patterns, making effective compression more challenging. In this paper, we propose XOR and NOT-based lossless compression (XNC), which applies XOR operations between adjacent 16-bit blocks and then performs a NOT operation on the result, effectively truncating the upper and lower bits to compress the embedding layer to 9-bit without any loss. The proposed method leverages XOR and NOT operations, enabling easy hardware implementation, with only four cycles required for compression and three cycles for decompression, ensuring efficient data compression without performance degradation. As a result, the proposed compression technique achieves an average compression ratio of 1.34× for the embedding layers of small LLMs without any loss, effectively reducing the model size of 4- bit quantized LLMs by an average of 9.91%.

# Installation

# Experiments

# Citation
If you find this repo useful in your research, please consider citing the following paper:
```
