import os
from dotenv import load_dotenv

load_dotenv()

# Standardized Model Name -> Provider Specific ID
MODEL_MAP = {
    "llama3-70b": {
        "groq": "llama-3.1-70b-versatile",
        "cerebras": "llama3.1-70b",
        "sambanova": "Meta-Llama-3.1-70B-Instruct",
    }
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")