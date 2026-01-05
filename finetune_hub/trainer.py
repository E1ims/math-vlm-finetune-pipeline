import transformers
from transformers import Trainer, TrainingArguments
from .config import ModelConfig

class TrainerWrapper:
    """
    A wrapper class to manage the Hugging Face Trainer execution.
    
    This class bridges the gap between our Custom Config, Data Pipeline, and the 
    standard Hugging Face training loop. It ensures all hyperparameters and 
    memory-saving tricks are applied correctly.
    """
    
    def __init__(self, model, processor, train_dataset, cfg: ModelConfig, data_collator):
        """
        Initialize the trainer wrapper.
        
        Args:
            model: The PEFT/LoRA wrapped model ready for training.
            processor: The VLM processor (tokenizer + image handler).
            train_dataset: The processed dataset containing images and text.
            cfg (ModelConfig): Our global configuration object.
            data_collator: The function that batches and pads our data.
        """
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.data_collator = data_collator

    def train(self):
        """
        Configures and executes the training loop.
        
        This method:
        1. Sets up TrainingArguments (The 'Control Panel').
        2. Initializes the Hugging Face Trainer.
        3. Runs the training.
        4. Saves the final adapter weights and processor configuration.
        
        Returns:
            trainer: The trainer object (useful for extracting logs/loss history).
        """
        
        # Define the training hyperparameters (The Control Panel)
        args = TrainingArguments(
            # Where to save checkpoints and final model
            output_dir=self.cfg.output_dir,
            
            # Batch size per GPU. Lower this if you hit OOM (Out of Memory).
            per_device_train_batch_size=self.cfg.batch_size,
            
            # Gradient Accumulation: Simulates a larger batch size by waiting X steps 
            # before updating weights. Essential for training on small GPUs.
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            
            # Warmup: Slowly ramps up learning rate to prevent shock to the model.
            warmup_steps=10,
            
            # Duration: How many times to iterate over the entire dataset.
            num_train_epochs=self.cfg.num_train_epochs,
            
            # Step size for weight updates.
            learning_rate=self.cfg.learning_rate,
            
            # Logging frequency (e.g., print loss every 10 steps).
            logging_steps=self.cfg.logging_steps,
            
            # Optimizer: "paged_adamw_8bit" is a specific QLoRA memory hack.
            # It offloads optimizer states to CPU RAM if GPU VRAM fills up.
            optim="paged_adamw_8bit",
            
            # Checkpoint Strategy: Save model every X steps.
            save_strategy="steps",
            save_steps=self.cfg.save_steps,
            
            # Mixed Precision: Uses 16-bit floats instead of 32-bit.
            # Reduces memory usage by ~50% and speeds up training.
            fp16=True,
            
            # Reporting: Disable external logging tools (like WandB) for simple Colab runs.
            report_to="none",
            
            # CRITICAL FOR VLM: By default, HF Trainer removes columns it doesn't recognize (like 'pixel_values').
            # We set this to False so our image data is passed to the model.
            remove_unused_columns=False
        )

        # Initialize the standard Hugging Face Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator
        )

        print("Starting Training...")
        trainer.train()

        print(f"Saving Trainer State to {self.cfg.output_dir}...")
        trainer.save_state()

        print(f"Saving Adapter to {self.cfg.output_dir}...")
        # Save the LoRA Adapter (Small file, in MBs)
        self.model.save_pretrained(self.cfg.output_dir)
        
        # Save the Processor (Tokenizer config + Image config)
        # Crucial for inference later so we know how to resize images exactly like training.
        self.processor.save_pretrained(self.cfg.output_dir)

        return trainer