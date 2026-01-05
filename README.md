# 📐 Math VLM Pipeline: Handwritten Math to LaTeX

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

> **A robust, modular Deep Learning pipeline for fine-tuning Vision-Language Models (VLMs) to transcribe handwritten mathematical expressions into accurate LaTeX code.**

Built on **Google's PaliGemma-3B**, utilizing **4-bit QLoRA (Quantized Low-Rank Adaptation)** for efficient training on consumer GPUs (e.g., T4/RTX 3060).

---

## 🚀 Key Features

* **🧩 Fully Modular Architecture:** Decoupled logic for Data, Training, and Inference. No hard-coded paths or prompts.
* **⚙️ Config-Driven:** Control everything (Hyperparameters, Prompts, Datasets) from a single `config.py` dataclass.
* **💾 Memory Efficient:** Implements **4-bit NF4 Quantization** and **Gradient Checkpointing** to train 3B parameters on <15GB VRAM.
* **🧠 Smart Inference:** Custom token-slicing logic to prevent hallucinated prompts in the final output.
* **📉 Production Ready:** Includes robust error handling, OOM prevention strategies (Gradient Accumulation), and inference cleaning.

---

## 📂 Project Structure

This project follows a "Library vs. Driver" design pattern for maximum maintainability.

```text
📁 math-vlm-pipeline/
├── 📂 finetune_hub/          # 🧠 THE CORE LIBRARY
│   ├── 📄 config.py          # Single Source of Truth (Dataclasses)
│   ├── 📄 engine.py          # Model Loading, QLoRA & 4-bit Quantization
│   ├── 📄 data.py            # Dataset Streaming & Dynamic Processing
│   ├── 📄 trainer.py         # Custom Hugging Face Trainer Wrapper
│   └── 📄 inference.py       # Production Inference Engine (Clean output)
├── 📓 Fine_Tuning.ipynb      # 🎛️ Master Experiment Driver (Notebook)
├── 📄 requirements.txt       # Dependencies
└── 📄 README.md              # Documentation
```

---

## 🛠️ Installation & Setup
### 1. Clone the Repository
```bash

git clone [https://github.com/nabeelshan78/math-vlm-pipeline.git](https://github.com/nabeelshan78/math-vlm-pipeline.git)
cd math-vlm-pipeline

```

### 2. Install Dependencies
```bash
pip install torch torchvision transformers datasets peft bitsandbytes accelerate
```

### 3. Authentication
You must have a Hugging Face token with access to gated models (PaliGemma).
```bash
huggingface-cli login
# Paste your token when prompted
```

---



## Quick Start

### 1. Configure Your RunOpen finetune_hub/config.py to set your parameters. The defaults are optimized for free Colab T4 GPUs.
```python
Python@dataclass
class ModelConfig:
    dataset_id: str = "deepcopy/MathWriting-human"
    prompt_text: str = "Convert this handwritten math to LaTeX."
    batch_size: int = 2          # Kept low for 16GB VRAM
    gradient_accumulation_steps: int = 8  # Effective Batch Size = 16
    num_train_epochs: int = 3
```
    
### 2. Train the Model
Run the Fine_Tuning.ipynb notebook

```python
# Initialize
config = ModelConfig()
engine = VLMEngine(config)
engine.load_model()
model = engine.apply_adapter()

# Load Data
data_proc = DataProcessor(engine.processor, config)
train_dataset = data_proc.load_data(limit=None)

# Train
trainer = TrainerWrapper(model, engine.processor, train_dataset, config, data_proc.collate_fn)
trainer.train()
```

### 3. Run Inference
```python
from finetune_hub import InferenceEngine

inference = InferenceEngine(base_model_id="google/paligemma-3b-pt-224", adapter_path="./math_vlm_adapter")
latex_code = inference.generate("my_handwritten_equation.png", prompt_text="Convert to LaTeX.")

print(latex_code)
# Output: \int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
```python

---

## 📊 Performance & Results
