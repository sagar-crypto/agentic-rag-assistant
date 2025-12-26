import requests

MODEL = "llama3.1"

def ollama_generate(prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]

if __name__ == "__main__":
    out = ollama_generate("Explain RAG in 3 bullet points for a beginner.")
    print(out)