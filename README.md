# XNC: XNC: XOR and NOT-Based Lossless Compression for Optimizing Unquantized Embedding Layers in Large Language Models
![image](https://github.com/user-attachments/assets/bb39dda5-b5c8-4192-a740-936a5417ab63)

## Abstart
Although 4-bit quantized small LLMs have been proposed recently, many studies have retained FP16 precision for embedding layers, as they constitute a relatively small proportion of the overall model in existing LLMs and suffer from severe accuracy degradation when quantized. However, in quantized small LLMs, the embedding layer accounts for a substantial proportion of the total model parameters, necessitating its compression. Since embedding layers are sensitive to approximation, lossless compression is more desirable than lossy compression methods such as quantization. While existing lossless compression methods efficiently compress patterns such as zeros, narrow values, or frequently occurring values, embedding layers typically lack these patterns, making effective compression more challenging. In this paper, we propose XOR and NOT-based lossless compression (XNC), which applies XOR operations between adjacent 16-bit blocks and then performs a NOT operation on the result, effectively truncating the upper and lower bits to compress the embedding layer to 9-bit without any loss. The proposed method leverages XOR and NOT operations, enabling easy hardware implementation, with only four cycles required for compression and three cycles for decompression, ensuring efficient data compression without performance degradation. As a result, the proposed compression technique achieves an average compression ratio of 1.34× for the embedding layers of small LLMs without any loss, effectively reducing the model size of 4- bit quantized LLMs by an average of 9.91%.

## Installation
```
conda create -n xnc python=3.11
conda activate xnc
```

## Results

|Model|All Zero|Narrow Pattern|Non Pattern|2nd,14th,15th,16th bit All Zero|2nd,14th,15th,16th bit Non-All Zero|
|------|---|---|---|---|---|
|Llama-3.2-1B|≈ 0%|0.26%|99.73%|99.83%|0.17%|
|Llama-3.2-3B|%|%|%|%|%|
|Gemma-2-2B|≈ 0%|0.26%|99.73%|99.90%|0.10%|
|Qwen-2.5-0.5B|≈ 0%|0.26%|99.73%|99.79%|0.21%|
|Qwen-2.5-1.5B|%|%|%|%|%|
|Qwen-2.5-3B|%|%|%|%|%|
|Phi-3.5-mini|≈ 0%|0.26%|99.73%|99.90%|0.10%|
|SmolLM-135M|0%|0.26%|99.74%|99.96%|0.04%|
|SmolLM-360M|%|%|%|%|%|
|SmolLM-1.7B|%|%|%|%|%|
|SmolVLM-256M|%|%|%|%|%|
|Qwen-2.5-VL-3B|%|%|%|%|%|

![image](https://github.com/user-attachments/assets/b5ed038e-b184-424a-bf31-fca4d0ef6466)

![image](https://github.com/user-attachments/assets/099e8af7-6c1d-41ab-b2a3-d3f555c772bb)


## Citation
If you find this repo useful in your research, please consider citing the following paper:
```

```
