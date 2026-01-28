# reformulator.py — QPP-triggered LLM query reformulation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np, torch, nltk
#nltk.download("punkt")

class ReformulatorConfig:
    def __init__(self, llm_model="google/flan-t5-base", n_variants=3, qpp_threshold=0.18):
        self.llm_model = llm_model
        self.n_variants = n_variants
        self.qpp_threshold = qpp_threshold

class Reformulator:
    def __init__(self, config: ReformulatorConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.llm_model)
        self.config = config

    def _clarity_score(self, query: str, docs: list[str]) -> float:
        vec = TfidfVectorizer()
        X = vec.fit_transform(docs)
        doc_avg = np.asarray(X.mean(axis=0)).flatten()
        qv = vec.transform([query]).toarray().flatten()
        mask = qv > 0
        score = 0.0
        for i in np.where(mask)[0]:
            if doc_avg[i] > 0:
                score += qv[i] * np.log2(qv[i] / doc_avg[i])
        return score

    def _prompt_variants(self, query: str) -> list[str]:
        templates = [
            f"Paraphrase this search query: {query}",
            f"Rewrite the query differently: {query}",
            f"Suggest search term variations for: {query}"
        ]
        out = []
        for t in templates[:self.config.n_variants]:
            toks = self.tokenizer(t, return_tensors="pt")
            with torch.no_grad():
                gen = self.model.generate(**toks, max_new_tokens=20)
            out.append(self.tokenizer.decode(gen[0], skip_special_tokens=True).strip().lower())
        return list(set(out))

    def _feedback_refine(self, query: str, docs: list[str]) -> str:
        context = "\n".join([d[:300] for d in docs])
        prompt = f"Based on the following context, suggest a better query for: {query}\n{context}"
        toks = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            gen = self.model.generate(**toks, max_new_tokens=20)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True).strip().lower()

    def reformulate(self, query: str, docs: list[str]) -> list[str]:
        clarity = self._clarity_score(query, docs)
        if clarity >= self.config.qpp_threshold:
            return [query]
        variants = self._prompt_variants(query)
        variants.append(self._feedback_refine(query, docs))
        return list(set([query] + variants))
