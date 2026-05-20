"""
Chitraksha Configuration File
All project settings and paths are defined here.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory (chitraksha folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
LLM_DIR = MODELS_DIR / "llm"

# Vector store directory
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "faiss_index"

# Logs and feedback
LOGS_DIR = PROJECT_ROOT / "logs"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASETS_DIR, 
                  EMBEDDINGS_DIR, LLM_DIR, VECTOR_STORE_DIR, 
                  LOGS_DIR, FEEDBACK_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Embedding model (for converting text to vectors)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Why this model? 
# - Small (80MB)
# - Fast on GPU
# - Good quality embeddings
# - 384 dimensions

# LLM model (for generating responses)
LLM_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
# Why this model?
# - Only 3.8B parameters (fits in 6GB GPU)
# - High quality responses
# - Fast inference
# - Good instruction following

# Alternative LLM options (if Phi-3.5 doesn't work):
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Even smaller, 1.1B params
# "meta-llama/Llama-3.2-3B-Instruct"    # Needs HF token

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

# Device settings
DEVICE = "cuda"  # Will auto-fallback to CPU if GPU unavailable
MAX_GPU_MEMORY = "5GB"  # Leave 1GB free for system

# Inference settings
MAX_NEW_TOKENS = 512  # Maximum response length
TEMPERATURE = 0.7     # Randomness (0.0 = deterministic, 1.0 = creative)
TOP_P = 0.9          # Nucleus sampling
TOP_K = 50           # Top-k sampling

# ============================================================================
# RAG CONFIGURATION
# ============================================================================

# Chunking settings
CHUNK_SIZE = 512      # Characters per chunk
CHUNK_OVERLAP = 50    # Overlap between chunks (prevents context loss)

# Retrieval settings
TOP_K_RETRIEVAL = 5   # Number of relevant chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0-1)

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# OCR settings (for Docling)
USE_OCR = True
OCR_ENGINE = "easyocr"  # Options: "easyocr", "tesseract"

# ============================================================================
# HUGGING FACE DATASETS
# ============================================================================

# Mental health datasets to load
HF_DATASETS = [
    "Amod/mental_health_counseling_conversations",
    # We'll add more as needed
]

# ============================================================================
# LOGGING
# ============================================================================

LOG_FILE = LOGS_DIR / "chitraksha.log"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

MAX_CONVERSATION_HISTORY = 10  # Number of previous exchanges to remember
CONVERSATION_SUMMARY_THRESHOLD = 5  # Summarize after this many exchanges

# ============================================================================
# FEEDBACK & SELF-LEARNING
# ============================================================================

FEEDBACK_FILE = FEEDBACK_DIR / "conversations.jsonl"
COLLECT_FEEDBACK = True

# ============================================================================
# SYSTEM PROMPTS (Basic structure - we'll refine later)
# ============================================================================

SYSTEM_PROMPT = """You are Chitraksha, a warm and supportive mental wellness companion.

Guidelines:
- Be empathetic, non-judgmental, and calm
- Ask gentle, open-ended questions
- Help users explore their feelings
- Never diagnose or prescribe
- Offer optional coping suggestions when appropriate
- Recognize crisis situations and provide resources

Remember: You're here to listen and support, not to fix."""

# Crisis keywords (for safety detection)
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "don't want to live",
    "hurt myself", "self harm", "overdose"
]

CRISIS_RESPONSE = """I'm really concerned about what you're sharing. Your safety is the most important thing right now.

Please reach out to a crisis support service immediately:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741
- International: https://findahelpline.com

I'm here, but trained professionals can provide the immediate help you need."""