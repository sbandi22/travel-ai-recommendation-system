"""
TravelIQ — Review Intelligence Module
DistilBERT-based sentiment + complaint detection + embedding extraction.

Usage:
    from modules.review import ReviewModel
    rm = ReviewModel(model_path="models/best_model.pt", tokenizer_path="models/traveliq_tokenizer")
    result = rm.predict("The museum was absolutely stunning!")
    emb    = rm.get_attraction_embedding(["Great views!", "Too crowded."])
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


# ── Model Definition ─────────────────────────────────────────────────
class TravelReviewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert          = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout       = nn.Dropout(0.3)
        self.sentiment_head = nn.Linear(768, 2)   # binary: Negative / Positive
        self.complaint_head = nn.Linear(768, 1)   # binary: complaint or not

    def forward(self, input_ids, attention_mask):
        output        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb       = output.last_hidden_state[:, 0, :]   # [CLS] token → 768-dim
        cls_emb       = self.dropout(cls_emb)
        sent_logits   = self.sentiment_head(cls_emb)
        comp_logit    = self.complaint_head(cls_emb).squeeze(-1)
        return sent_logits, comp_logit, cls_emb


# ── Wrapper Class ────────────────────────────────────────────────────
class ReviewModel:
    def __init__(self, model_path: str = "models/best_model.pt",
                 tokenizer_path: str = "models/traveliq_tokenizer"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.model     = TravelReviewModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[ReviewModel] Loaded from {model_path} on {self.device}")

    def _encode(self, text: str):
        return self.tokenizer(
            str(text), max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )

    def predict(self, text: str) -> dict:
        """Predict sentiment + complaint for a single review."""
        enc = self._encode(text)
        with torch.no_grad():
            sent_logits, comp_logit, _ = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device)
            )
        return {
            "sentiment": "Positive" if torch.argmax(sent_logits).item() == 1 else "Negative",
            "complaint": torch.sigmoid(comp_logit).item() > 0.5,
        }

    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        Predict sentiment for a list of reviews in batches.
        Returns list of {"sentiment": str} dicts — same length as texts.
        ~20x faster than calling predict() one-by-one.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc   = self.tokenizer(
                chunk, max_length=128, padding=True,
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                sent_logits, _, _ = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device)
                )
            labels = torch.argmax(sent_logits, dim=1).tolist()
            results.extend(
                {"sentiment": "Positive" if lbl == 1 else "Negative"}
                for lbl in labels
            )
        return results

    def get_attraction_embedding(self, reviews: list) -> np.ndarray:
        """Legacy — kept for compatibility. Returns zero vector (embedding unused by ranker)."""
        return np.zeros(768)


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    rm = ReviewModel()
    test_reviews = [
        "The restaurant was neat and food was yummy",
        "The museum was absolutely stunning, loved every exhibit!",
        "Terrible experience. Staff was rude and it was way too crowded.",
        "It was okay, nothing special but not bad either.",
    ]
    for review in test_reviews:
        result = rm.predict(review)
        print(f'Review : "{review[:60]}"')
        print(f'         Sentiment: {result["sentiment"]} | Complaint: {result["complaint"]}\n')

    # Attraction-level embedding
    emb = rm.get_attraction_embedding(test_reviews)
    print(f"Attraction embedding shape: {emb.shape}")
