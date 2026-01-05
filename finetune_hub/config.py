from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """
    Configuration class acting as the 'Single Source of Truth' for the entire pipeline.
    This centralizes all hyperparameters, ensuring consistency across training and inference.
    """

    # --- Model & Data Settings ---
    # The Hugging Face ID of the base Vision-Language Model (VLM) to fine-tune.
    model_id: str = "google/paligemma-3b-pt-224"
    
    # The identifier for the dataset on Hugging Face Hub (contains images + latex labels).
    dataset_id: str = "deepcopy/MathWriting-human"
    
    # The standard instruction prompt used during training. 
    # Must be consistent during inference to trigger the correct behavior.
    prompt_text: str = "Convert this handwritten math to LaTeX."
    
    # --- Quantization & LoRA (Memory Efficiency) ---
    # If True, loads the model in 4-bit precision (NF4) to drastically reduce VRAM usage.
    use_4bit: bool = True
    
    # The rank 'r' of the low-rank adaptation matrices. 
    # Higher = more capacity/smarter but uses more VRAM. 16 is a standard balance.
    lora_rank: int = 16
    
    # Scaling factor for LoRA updates. Rule of thumb: typically set to 2x the rank.
    lora_alpha: int = 32
    
    # Dropout probability applied to LoRA layers to prevent overfitting on small datasets.
    lora_dropout: float = 0.05
    
    # List of model modules to apply adapters to. Targeting all linear layers (q, v, k, o, etc.)
    # generally yields better results than just attention heads.
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # --- Training Hyperparameters ---
    # The step size for the optimizer. 2e-4 is standard for QLoRA fine-tuning.
    learning_rate: float = 2e-4
    
    # Number of samples per GPU per step. Increase if you have more VRAM.
    batch_size: int = 4
    
    # Accumulate gradients over N steps before updating weights.
    # Effective Batch Size = batch_size * gradient_accumulation_steps.
    # Helps simulate large batch training on small GPUs.
    gradient_accumulation_steps: int = 4
    
    # How many times the model sees the entire dataset. 
    # 1 epoch is often enough for simple syntax learning; increase for complex reasoning.
    num_train_epochs: int = 1
    
    # Save a checkpoint of the model weights every N steps.
    save_steps: int = 25
    
    # --- System & Task Config ---
    # Maximum token length for inputs/outputs. Shorten this if OOM occurs.
    max_seq_length: int = 512
    
    # Directory where checkpoints and the final adapter will be saved.
    output_dir: str = "./math_vlm_adapter"
    
    # How often to print training stats (loss) to the console.
    logging_steps: int = 5
    
    # Specifies the task structure for the PEFT library (Causal Language Modeling).
    task_type: str = "CAUSAL_LM"