from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
SRC_DIR = BASE_DIR / "src"

# Default PDF name (you can change this anytime)
DEFAULT_PDF_NAME = "pdf-test.pdf"
DEFAULT_PDF_PATH = DATA_DIR / DEFAULT_PDF_NAME

CHROMA_DIR = STORAGE_DIR / "chroma"
CHROMA_COLLECTION = "papers"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

CHROMA_COLLECTION = "papers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"
TOP_K = 5

UPLOADS_DIR = DATA_DIR / "uploads"

API_HOST = "127.0.0.1"
API_PORT = 8000