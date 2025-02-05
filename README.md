# XNC: XNC: XOR and NOT-Based Lossless Compression for Optimizing Unquantized Embedding Layers in Large Language Models
![image](https://github.com/user-attachments/assets/bb39dda5-b5c8-4192-a740-936a5417ab63)

## Abstart
Although 4-bit quantized small LLMs have been proposed recently, many studies have retained FP16 precision for embedding layers, as they constitute a relatively small proportion of the overall model in existing LLMs and suffer from severe accuracy degradation when quantized. However, in quantized small LLMs, the embedding layer accounts for a substantial proportion of the total model parameters, necessitating its compression. Since embedding layers are sensitive to approximation, lossless compression is more desirable than lossy compression methods such as quantization. While existing lossless compression methods efficiently compress patterns such as zeros, narrow values, or frequently occurring values, embedding layers typically lack these patterns, making effective compression more challenging. In this paper, we propose XOR and NOT-based lossless compression (XNC), which applies XOR operations between adjacent 16-bit blocks and then performs a NOT operation on the result, effectively truncating the upper and lower bits to compress the embedding layer to 9-bit without any loss. The proposed method leverages XOR and NOT operations, enabling easy hardware implementation, with only four cycles required for compression and three cycles for decompression, ensuring efficient data compression without performance degradation. As a result, the proposed compression technique achieves an average compression ratio of 1.34× for the embedding layers of small LLMs without any loss, effectively reducing the model size of 4- bit quantized LLMs by an average of 9.91%.

## Installation
```
conda create -n xnc python=3.11
conda activate xnc

pip install -e .
```

## Results
- ss

|model|All Zero|Narrow Pattern|Non Pattern|2<sup>nd</sup>, 14<sup>th</sup>, 15<sup>th</sup>, <br> 16<sup>th</sup> bit all zero|2<sup>nd</sup>, 14<sup>th</sup>, 15<sup>th</sup>,  <br> 16<sup>th</sup> bit Non-All Zero|
|:----------------:|:---:|:---:|:---:|:---:|:---:|
|Llama-3.2-1B|≈ 0%|0.26%|99.73%|99.83%|0.17%|
|Llama-3.2-3B|≈ 0%|0.25%|99.74%|99.83%|0.17%|
|Gemma-2-2B|≈ 0%|0.26%|99.73%|99.90%|0.10%|
|Qwen-2.5-0.5B|≈ 0%|0.26%|99.73%|99.79%|0.21%|
|Qwen-2.5-1.5B|≈ 0%|0.26%|99.73%|99.86%|0.14%|
|Qwen-2.5-3B|≈ 0%|0.26%|99.73%|99.86%|0.14%|
|Phi-3.5-mini|≈ 0%|0.26%|99.73%|99.90%|0.10%|
|SmolLM-135M|0%|0.26%|99.74%|99.96%|0.04%|
|SmolLM-360M|≈ 0%|0.26%|99.73%|99.98%|0.02%|
|SmolLM-1.7B|≈ 0%|0.26%|99.73%|99.97%|0.03%|
|SmolVLM-256M|0.02%|0.26%|99.72%|99.98%|0.02%|
|Qwen-2.5-VL-3B|≈ 0%|0.26%|99.73%|99.85%|0.15%|

- ss
![image](https://github.com/user-attachments/assets/b5ed038e-b184-424a-bf31-fca4d0ef6466)

- ss
![image](https://github.com/user-attachments/assets/099e8af7-6c1d-41ab-b2a3-d3f555c772bb)


## Citation
If you find this repo useful in your research, please consider citing the following paper:
```

```
