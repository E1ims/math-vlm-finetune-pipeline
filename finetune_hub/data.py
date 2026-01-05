import torch
from datasets import load_dataset
from PIL import Image

class DataProcessor:
    """
    Handles data pipeline operations: loading raw datasets and processing batches 
    for the Vision-Language Model.
    
    This class acts as the bridge between the raw Hugging Face dataset and the 
    VLM's expected input format (tensors).
    """
    def __init__(self, processor, cfg):
        """
        Args:
            processor: The Hugging Face AutoProcessor (handles tokenization & image resizing).
            cfg (ModelConfig): Global configuration object containing dataset IDs and prompt text.
        """
        self.processor = processor
        self.cfg = cfg

    def load_data(self, split="train", limit=None):
        """
        Loads the dataset from the Hugging Face Hub based on the config.

        Args:
            split (str): Which split to load (e.g., "train", "test", "validation").
            limit (int, optional): If set, only loads the first N examples. 
                                   Useful for quick debugging or sanity checks.
        """
        print(f"Loading dataset: {self.cfg.dataset_id}...")
        ds = load_dataset(self.cfg.dataset_id, split=split)
        
        # Slicing the dataset for rapid prototyping if a limit is specified
        if limit:
            ds = ds.select(range(limit))
        return ds

    def collate_fn(self, examples):
        """
        The critical batch processing function. This is called by the DataLoader 
        to merge a list of samples into a single batch of tensors.
        
        It handles:
        1. Injecting the text prompt for every image.
        2. Normalizing images (RGB conversion).
        3. Tokenizing text and processing images using the VLM processor.
        """
        # Dynamic prompt injection:
        # We repeat the prompt from config (e.g., "Convert to LaTeX") for every example in the batch
        texts = [f"{self.cfg.prompt_text}" for _ in examples]
        
        # Image normalization:
        # We ensure all images are 3-channel RGB. This prevents errors if the dataset 
        # contains Grayscale (1 channel) or RGBA (4 channels) images.
        images = [ex["image"].convert("RGB") for ex in examples]
        
        # Extract the ground truth labels (the LaTeX code)
        labels = [ex["latex"] for ex in examples]

        # Process the batch:
        # The processor handles tokenization, resizing, pixel normalization, and 
        # creating the attention masks automatically.
        inputs = self.processor(
            text=texts,
            images=images,
            suffix=labels,          # The 'suffix' is what the model learns to generate (the answer)
            return_tensors="pt",    # Return PyTorch tensors
            padding="longest",      # Pad to the length of the longest sequence in this specific batch
            truncation=True,        # Truncate if sequence exceeds max_length
            max_length=self.cfg.max_seq_length
        )

        return inputs