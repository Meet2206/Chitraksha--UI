"""
Utility functions for Chitraksha
Reusable helper functions used across notebooks and modules.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import torch

from src.config import LOG_FILE, LOG_LEVEL

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = "chitraksha") -> logging.Logger:
    """
    Set up logging for the project.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# GPU UTILITIES
# ============================================================================

def get_device() -> torch.device:
    """
    Get the best available device (GPU or CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        logger = setup_logger()
        logger.warning("GPU not available, falling back to CPU")
        return torch.device("cpu")

def print_gpu_info():
    """Print detailed GPU information."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU available")

# ============================================================================
# FILE OPERATIONS
# ============================================================================

def save_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """
    Save data as JSONL (one JSON object per line).
    
    Args:
        data: List of dictionaries to save
        filepath: Path to save file
    """
    with open(filepath, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        filepath: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data

# ============================================================================
# TEXT PROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Remove multiple newlines
    text = '\n'.join(line for line in text.split('\n') if line.strip())
    
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Size of each chunk (characters)
        overlap: Overlap between chunks (characters)
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap
    
    return chunks

# ============================================================================
# CONVERSATION UTILITIES
# ============================================================================

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    """
    Format conversation history for context.
    
    Args:
        history: List of {role: str, content: str} messages
        
    Returns:
        Formatted string
    """
    formatted = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(formatted)

def create_feedback_entry(
    user_message: str,
    bot_response: str,
    user_feedback: str = None,
    metadata: Dict = None
) -> Dict[str, Any]:
    """
    Create a feedback entry for the self-learning dataset.
    
    Args:
        user_message: User's input
        bot_response: Bot's response
        user_feedback: Optional user feedback (thumbs up/down, rating)
        metadata: Additional metadata
        
    Returns:
        Feedback dictionary
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
        "user_feedback": user_feedback,
        "metadata": metadata or {}
    }
    return entry

# ============================================================================
# VALIDATION
# ============================================================================

def validate_model_loaded(model) -> bool:
    """
    Check if a model is properly loaded.
    
    Args:
        model: Model object
        
    Returns:
        True if valid, False otherwise
    """
    return model is not None

def validate_embeddings(embeddings: torch.Tensor, expected_dim: int) -> bool:
    """
    Validate embedding dimensions.
    
    Args:
        embeddings: Embedding tensor
        expected_dim: Expected dimension size
        
    Returns:
        True if valid
    """
    return embeddings.shape[-1] == expected_dim