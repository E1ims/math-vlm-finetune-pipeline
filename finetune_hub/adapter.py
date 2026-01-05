import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from .config import ModelConfig

class AdapterFactory:
    """
    Factory class for creating parameter-efficient fine-tuning (PEFT) configurations.
    
    This handles two critical components of efficient training:
    1. QLoRA Quantization: Compressing the base model to 4-bit precision to fit in VRAM.
    2. LoRA Adapters: Defining the low-rank matrices that will actually be trained.
    """

    @staticmethod
    def get_qlora_config(cfg: ModelConfig) -> BitsAndBytesConfig:
        """
        Creates the BitsAndBytes configuration for 4-bit quantization (QLoRA).
        
        Args:
            cfg (ModelConfig): The global model configuration object.
            
        Returns:
            BitsAndBytesConfig: Configuration that forces the base model to load in 4-bit NF4 format.
        """

        return BitsAndBytesConfig(
            # Enable 4-bit loading (Reducing memory usage by ~4x)
            load_in_4bit=cfg.use_4bit,
            
            # "nf4" (Normal Float 4) is an information-theoretically optimal data type for normally distributed weights, offering better precision than standard FP4.
            bnb_4bit_quant_type="nf4",
            
            # While storage is 4-bit, computations (like matrix multiplication) happen in Float16 for speed and stability.
            bnb_4bit_compute_dtype=torch.float16,
            
            # Double Quantization saves an extra 0.4 bits per parameter by quantizing the quantization constants. This is critical for squeezing large models onto small RAM GPUs.
            bnb_4bit_use_double_quant=True
        )

    @staticmethod
    def get_lora_config(cfg: ModelConfig) -> LoraConfig:
        """
        Creates the Low-Rank Adaptation (LoRA) configuration.
        
        LoRA freezes the pre-trained model weights and injects trainable rank decomposition matrices
        into each layer of the Transformer architecture.
        
        Args:
            cfg (ModelConfig): The global model configuration object.
            
        Returns:
            LoraConfig: Configuration defining the rank, alpha, and target modules for the adapter.
        """

        return LoraConfig(
            # 'r' is the rank of the low-rank update matrices. Higher 'r' (e.g., 64) means more trainable parameters/capacity but higher VRAM usage.
            r=cfg.lora_rank,

            # 'lora_alpha' is a scaling factor. The update is scaled by alpha/r. A common heuristic is setting alpha = 2 * rank.
            lora_alpha=cfg.lora_alpha,

            # Which specific layers to apply adapters to. For VLM/LLMs, targeting all linear layers. (q_proj, k_proj, v_proj, o_proj, etc.) generally yields better performance than just attention.
            target_modules=cfg.target_modules,

            # Dropout probability for the LoRA layers to prevent overfitting on small datasets.
            lora_dropout=cfg.lora_dropout,

            # 'bias'="none" means we do not train bias terms, saving parameters.
            bias="none",

            # Defines the task type (e.g., CAUSAL_LM) to ensure the adapter attaches correctly to the model head.
            task_type=cfg.task_type
        )