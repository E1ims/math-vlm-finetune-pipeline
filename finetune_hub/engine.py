import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import get_peft_model, prepare_model_for_kbit_training
from .config import ModelConfig
from .adapter import AdapterFactory

class VLMEngine:
    """
    The Core Engine responsible for the lifecycle of the Vision-Language Model.
    
    Responsibilities:
    1. Loading the massive base model (PaliGemma) into memory efficiently.
    2. applying 4-bit quantization (QLoRA) to fit it on consumer GPUs.
    3. Attaching the trainable Low-Rank Adapters (LoRA) for fine-tuning.
    """
    
    def __init__(self, cfg: ModelConfig):
        """
        Initialize the engine with the configuration.
        
        Args:
            cfg (ModelConfig): The global configuration containing model IDs, 
                               quantization settings, and hyperparameters.
        """
        self.cfg = cfg
        self.processor = None  # Handle for the tokenizer and image processor
        self.model = None      # Handle for the actual neural network

    def load_model(self):
        """
        Loads the base model with 4-bit quantization and prepares it for training.
        This is the most memory-intensive step.
        """
        print(f"Loading Base Model: {self.cfg.model_id}...")
        
        # 1. Get the Quantization Config (BitsAndBytes)
        # This tells the model to load weights in 4-bit NF4 format to save VRAM.
        bnb_config = AdapterFactory.get_qlora_config(self.cfg)

        # 2. Load the Pre-trained Model
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            quantization_config=bnb_config, # Apply the 4-bit compression
            device_map="auto",              # Automatically split model across available GPUs (or CPU if needed)
            dtype=torch.float16             # Compute in Half Precision (FP16) for speed
        )

        # 3. Load the Processor
        # This handles converting text to tokens and images to pixel tensors.
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_id)

        # 4. Enable Gradient Checkpointing
        # CRITICAL OPTIMIZATION: This trades a small amount of computation speed for massive VRAM savings.
        # It discards intermediate activations during the forward pass and re-computes them 
        # during the backward pass, preventing OOM errors on long sequences.
        self.model.gradient_checkpointing_enable()
        
        # 5. Prepare for k-bit Training
        # This helper function stabilizes 4-bit training by:
        # - Freezing base model layers
        # - Casting LayerNorms to float32 (for numeric stability)
        # - Requiring gradients on inputs (needed for gradient checkpointing)
        self.model = prepare_model_for_kbit_training(self.model)

    def apply_adapter(self):
        """
        Injects the trainable LoRA adapters into the frozen base model.
        After this step, the model is ready for fine-tuning.
        """
        # 1. Get the LoRA Configuration
        # Defines rank (r), alpha, and which modules (q_proj, v_proj, etc.) to target.
        lora_config = AdapterFactory.get_lora_config(self.cfg)
        
        # 2. Wrap the Base Model
        # This adds the tiny trainable adapter matrices alongside the frozen 4-bit weights.
        self.model = get_peft_model(self.model, lora_config)

        # 3. Verification
        # Prints exactly how many parameters we are training (usually < 1-2% of total).
        print("\n--- Trainable Parameters ---")
        self.model.print_trainable_parameters()
        
        return self.model