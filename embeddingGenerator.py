from transformers import AutoTokenizer, AutoModel
from typing import List
import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        texts = [str(text) for text in texts]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Pooling embeddings (mean of the last hidden state)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)
