# Scene Description using Vision-Language Models (TASK 2)

A pipeline to generate descriptive captions for images using various open-source Vision-Language Models (VLLMs). 

The system accepts a folder of images, processes them through a selected VLLM, and outputs multiple descriptive captions per image. Also included is comparison of the performance and output quality of different models on datasets like COCO and NoCaps.

---

## Features

-   **Supported Models**: Easily configurable to run several state-of-the-art VLLMs available on Hugging Face, including:
    -   BLIP (`blip-base`, `blip-large`)
    -   BLIP-2 (`blip2-opt`)
    -   LLaVA (`llava-1.5-7b-hf`)
    -   Qwen-VL (`Qwen2.5-VL-3B-Instruct`)
    -   SmolVLM (`SmolVLM-Instruct`)
    -   GIT (`git-base`)
-   All model and generation parameters (prompts, token limits, sampling methods, etc.) are managed through simple YAML configuration files.
-   **Performance Tracking**: Automatically measures and logs key performance indicators for each model run:
    -   Latency per image (ms)
    -   Peak GPU memory usage (MB)
    -   Throughput (tokens per second)
-   **Output**: Generates a structured JSON file for each experiment run, containing:
    -   Three Generated captions for each image.
    -   Performance metrics for each image.
    -   Aggregated summary statistics (average latency, memory usage, etc.).
-   **Visualization**: Saves visualizations of the first five images with their top generated caption for quick qualitative assessment.
-   **Quantitative Evaluation**: Includes a standalone script (`src/evaluation.py`)  to calculate standard NLP metrics against reference captions (e.g., from COCO or NoCaps datasets), including:
    -   **BLEU**
    -   **ROUGE**
    -   **CIDEr**, 
    -   **SPICE**, 
    -   **METEOR**
    -   **BERTScore**
    -   **CLIPScore** (reference-free metric)
---

## Getting Started

### Prerequisites

-   Python 3.12
-   NVIDIA GPU with CUDA support
-   Conda (recommended for environment management)

### Installation

1.  **Create and activate the Conda environment:**
    ```bash
    conda create -n scene-description python=3.12.3
    conda activate scene-description
    ```

2.  **Install PyTorch with CUDA support:**
    It's recommended to install PyTorch separately to ensure the correct CUDA version is used.
    ```bash
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    ```

3.  **Install the remaining requirements:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) Download Datasets for Evaluation:**
    For the bonus task of comparing models, this project supports the COCO and NoCaps datasets.
    * Download the [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip).
    * Extract them to a known location, for example, `datasets/coco/val2017`.
    *(The NoCaps dataset will be downloaded automatically by the script when first used.)*

---
### ⚠️ Important Note on Dependencies


To run the evaluation script (`src/evaluation.py`) and calculate metrics like **CIDEr** and **SPICE**, **Java** should be installed. They rely on the `pycocoevalcap` library, which has a Java dependency.

---

## How to Run

The script can be run in two primary modes: generating captions for a custom folder of images or evaluating models on a standard dataset.

### Quick Start: Generating Captions for a Custom Folder

to generate descriptions for any folder of images on your machine.

**Arguments:**
* `--image_folder`: **(Required)** Path to your folder of images.
* `--config`: **(Required)** Path to the model's YAML configuration file (e.g., `configs/smolvlm.yaml`).

**Example Command:**
```bash
python src/main.py \
    --config configs/smolvlm.yaml \
    --image_folder /path/to/your/images
```    
---
### Model Evaluation on Standard Datasets

To reproduce the metric evaluations and compare different models, you can run the script on the COCO or NoCaps datasets.

**Arguments:**

* ``--dataset``: **(Required)**  The dataset to use (coco or nocaps).

* ``--config``: **(Required)** Path to the model's YAML configuration file.

* ``--coco_path``: Path to the COCO images folder (required if dataset is coco).

* ``--nocaps_split``: The split for the NoCaps dataset (validation or test) (if dataset is nocaps).

```bash
python src/main.py \
    --config configs/llava-hf.yaml \
    --dataset coco \
    --coco_path /datasets/coco/val2017
```

## Model Comparison & Results

Experiments were run to compare the performance and output quality of several Vision-Language Models on the COCO validation dataset. A summary of the key findings is presented below.

### Performance Metrics

This table shows the average performance metrics across all processed images for each model. Performance was measured on an NVIDIA RTX 4090 GPU.

| Model | Avg Latency (ms) | Avg GPU Mem (MB) | Tokens/sec |
| :--- | :--- | :--- | :--- |
| SmolVLM-Instruct | 2292.64 | 5394.81 | 16.43 |
| llava-1.5-7b-hf | 3680.83 | 14703.8 | 18.92 |
| Qwen2.5-VL-3B-Instruct | 52112.64 | 4149.9 | 2.42 |
| blip-image-captioning-base | 450.02 | 1093.91 | 41.85 |
| blip-image-captioning-large | 469.19 | 1943.85 | 40.36 |
| blip2-opt-2.7b | 341.38 | 14396.8 | 27.97 |

*(Note: These results are based on a sample run on a specific hardware configuration. Performance will vary.)*

**Analysis:**

Based on the models evaluated in this analysis, the following conclusions can be drawn:

* **Efficiency:** The **BLIP** models are by far the fastest and most efficient in terms of latency and throughput.

* **Resource Usage:** The larger instruction-tuned models, LLaVA and BLIP2, consume significantly more GPU memory. In contrast, **Qwen-VL** has a remarkably low memory footprint for its size, a result of being loaded with 8-bit quantization (`load_in_8bit=True`).


### Quantitative Evaluation Metrics

The following tables show the results from standard evaluation metrics against the COCO and NoCaps ground-truth captions.

#### 📊 COCO Evaluation Metrics

| Model | BLEU-4 | CIDEr | SPICE | METEOR | ROUGE-L | BERTScore-F1 | CLIPScore |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolVLM-Instruct** | 0.0542 | 0.1430 | 0.3834 | 0.1912 | 0.3162 | 0.9026 | 18.038 |
| **llava-1.5-7b-hf** | 0.0363 | 0.1718 | 0.0104 | 0.1874 | 0.197 | 0.8829 | 20.4585 |
| **Qwen2.5-VL-3B-Instruct** | 0.0264 | 0.1288 | 0.0 | 0.1508 | 0.1531 | 0.8541 | 19.422 |
| **blip-image-captioning-base** | 0.1426 | 0.1886 | 0.4752 | 0.2427 | 0.4086 | 0.9113 | 20.9779 |
| **blip-image-captioning-large** | 0.2109 | 0.2249 | 0.6479 | 0.2839 | 0.4815 | 0.9239 | 21.1418 |
| **blip2-opt-2.7b** | 0.2632 | 0.1154 | 1.09 | 0.2471 | 0.5243 | 0.9377 | 22.4577 |

#### 📊 NoCaps Evaluation Metrics

| Model | BLEU-4 | CIDEr | SPICE | METEOR | ROUGE-L | BERTScore-F1 | CLIPScore |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolVLM-Instruct** | 0.1175 | 0.1224 | 0.5135 | 0.2112 | 0.3739 | 0.9133 | 17.7388 |
| **llava-1.5-7b-hf** | 0.0554 | 0.1448 | 0.0164 | 0.201 | 0.2322 | 0.8884 | 20.702 |
| **Qwen2.5-VL-3B-Instruct** | 0.0451 | 0.143 | 0.0 | 0.1755 | 0.1807 | 0.8602 | 19.3543 |
| **blip-image-captioning-base** | 0.1718 | 0.1188 | 0.4643 | 0.2271 | 0.409 | 0.9034 | 20.8648 |
| **blip-image-captioning-large** | 0.2718 | 0.1627 | 0.6475 | 0.284 | 0.4978 | 0.9225 | 21.2237 |
| **blip2-opt-2.7b** | 0.3359 | 0.1196 | 0.9084 | 0.2481 | 0.5425 | 0.9334 | 22.3054 |

---

### Qualitative Analysis & Discussion

While the quantitative metrics provide a useful baseline, a manual review of the generated captions reveals that the models with the highest metrics scores aren't always the most accurate or helpful in real-world scenarios.

* **Qwen-VL** and **SmolVLM**, excelled at following the specific, detailed prompts used in the experiment.
    * **Qwen-VL** was particularly impressive in its ability to generate structured, bullet-point descriptions as requested, providing clean and factual output.
    * **SmolVLM** consistently produced concise, relevant, and highly accurate captions that directly described the image content without adding extraneous details.
* **LLaVA** and the **BLIP family**, were significantly more prone to hallucination.
    * **LLaVA** often generated very descriptive and fluent sentences, but it frequently included objects, actions, or relationships that were not actually present in the image.
* The **BLIP2** model, did not respond well to the complex prompt. Its strength lies in simpler, unconditional captioning, and its performance suffered when given detailed instructions.
---

### Qualitative Examples

Here are some side-by-side examples of the captions generated by each model for the same image.

<table>
  <tr>
    <td align="center"><strong>SmolVLM-Instruct</strong></td>
    <td align="center"><strong>Qwen2.5-VL-3B-Instruct</strong></td>
  </tr>
  <tr>
    <td><img src="sample_outputs\HuggingFaceTB_SmolVLM-Instruct_custom_data_2025-08-26_18-41-14\visualizations\0.png" width="400"></td>
    <td><img src="sample_outputs\Qwen_Qwen2.5-VL-3B-Instruct_custom_data_2025-08-26_18-46-20\visualizations\0.png" width="400"></td>
  </tr>
  <tr>
    <td align="center"><strong>llava-1.5-7b-hf</strong></td>
    <td align="center"><strong>blip2-opt</strong></td>
  </tr>
  <tr>
    <td><img src="sample_outputs\llava-hf_llava-1.5-7b-hf_custom_data_2025-08-26_18-52-16\visualizations\0.png" width="400"></td>
    <td><img src="sample_outputs\Salesforce_blip2-opt-2.7b_custom_data_2025-08-26_18-51-42\visualizations\0.png" width="400"></td>
  </tr>
    <tr>
    <td align="center"><strong>blip-base</strong></td>
    <td align="center"><strong>blip-large</strong></td>
  </tr>
  <tr>
    <td><img src="sample_outputs\Salesforce_blip-image-captioning-base_custom_data_2025-08-26_18-50-12\visualizations\0.png" width="400"></td>
    <td><img src="sample_outputs\Salesforce_blip-image-captioning-large_custom_data_2025-08-26_18-50-30\visualizations\0.png" width="400"></td>
  </tr>
    
</table>
The visualisation shows clear hallucination by llava-hf and blip-large model, with both models saying that there are two persons.

---

### Conclusion & Recommendations

If the primary goal is to generate a caption that has high word overlap with a standard dataset like COCO, the **BLIP2** and **BLIP-Large** models are statistically the best.

However, for a practical application requiring **factual accuracy, reliability, and the ability to follow specific instructions**, the traditional metrics are a poor indicator of success. Based on qualitative review, **SmolVLM** and **Qwen-VL** are the better models for this task.

---
