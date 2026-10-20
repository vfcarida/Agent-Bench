"""Hybrid BM25 + Dense Vector Retrieval & Indirect RAG Corpus Poisoning Defense.

Implements a deterministic hybrid retriever combining Okapi BM25 lexical scoring
and dense n-gram random-projection vector embeddings via Reciprocal Rank Fusion (RRF),
paired with a RAG Corpus Poisoning Sanitizer & Provenance Verifier that detects
indirect prompt injection, backdoor triggers, and trust-tier spoofing in retrieved chunks.

References:
- Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- Zou et al. (2024), "PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation"
- Greshake et al. (2023), "Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection"
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from agent_bench.core.adapters import RetrievalAdapter, RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9_à-ú]+", re.IGNORECASE)

_POISON_TRIGGERS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
            r"disregard\s+(?:all\s+)?(?:prior|system)\s+rules|"
            r"forget\s+your\s+system\s+prompt)",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "chatml_delimiter_injection",
        re.compile(
            r"(?:<\|im_start\|>|<\|im_end\|>|\[SYSTEM_OVERRIDE\]|\[ADMIN_NOTE\]|"
            r"###\s*(?:SYSTEM|INSTRUCTION)\s*OVERRIDE)",
            re.IGNORECASE,
        ),
        0.90,
    ),
    (
        "regulatory_policy_spoof",
        re.compile(
            r"(?:urgent\s+policy\s+amendment\s*:\s*bypass|"
            r"suitability\s+verification\s+is\s+waived|"
            r"compliance\s+check\s+disabled|"
            r"authorized\s+to\s+skip\s+2fa)",
            re.IGNORECASE,
        ),
        0.80,
    ),
    (
        "exfiltration_directive",
        re.compile(
            r"(?:exfiltrate|send\s+all\s+(?:pii|credentials|tokens)\s+to|"
            r"webhook\.site|attacker\.example|evil\.com|"
            r"unrestricted_transfer)",
            re.IGNORECASE,
        ),
        0.90,
    ),
)

_IMPERATIVE_VERBS = frozenset(
    {
        "ignore",
        "bypass",
        "override",
        "execute",
        "transfer",
        "disable",
        "exfiltrate",
        "disregard",
        "skip",
        "reveal",
        "export",
        "grant",
        "sudo",
    }
)

_TRUST_TIER_RANK: dict[str, int] = {
    "untrusted": 0,
    "external": 1,
    "community": 2,
    "internal": 3,
    "verified": 4,
    "regulatory": 5,
}


@dataclass(frozen=True)
class PoisoningScanVerdict:
    """Audit verdict produced by scanning a retrieved document for RAG poisoning."""

    doc_id: str
    poisoned: bool
    risk_score: float
    imperative_density: float
    matched_triggers: list[str] = field(default_factory=list)
    trust_tier_violation: bool = False
    reason: str = "clean"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorpusPoisoningSanitizer:
    """Scans and sanitizes retrieved RAG corpus chunks against indirect poisoning and backdoors."""

    def __init__(
        self,
        *,
        risk_threshold: float = 0.55,
        imperative_density_threshold: float = 0.18,
        min_trust_tier: str = "internal",
    ) -> None:
        self.risk_threshold = float(risk_threshold)
        self.imperative_density_threshold = float(imperative_density_threshold)
        self.min_trust_tier = min_trust_tier

    def scan_document(
        self,
        doc: dict[str, Any],
        *,
        min_trust_tier: str | None = None,
    ) -> PoisoningScanVerdict:
        """Inspect a single document dictionary for RAG poisoning indicators."""
        doc_id = str(doc.get("doc_id", "unknown_doc"))
        title = str(doc.get("title", ""))
        content = str(doc.get("content", ""))
        full_text = f"{title}\n{content}"

        tokens = [t.lower() for t in _TOKEN_RE.findall(full_text)]
        if tokens:
            imp_count = sum(1 for t in tokens if t in _IMPERATIVE_VERBS)
            imperative_density = imp_count / len(tokens)
        else:
            imperative_density = 0.0

        matched_triggers: list[str] = []
        risk_score = 0.0
        for name, pattern, weight in _POISON_TRIGGERS:
            if pattern.search(full_text):
                matched_triggers.append(name)
                risk_score = max(risk_score, weight)

        if imperative_density >= self.imperative_density_threshold:
            matched_triggers.append("high_imperative_density")
            risk_score = max(
                risk_score,
                min(1.0, 0.50 + imperative_density * 1.5),
            )

        required_tier = min_trust_tier or self.min_trust_tier
        doc_tier = str(doc.get("trust_tier", "verified")).lower()
        doc_rank = _TRUST_TIER_RANK.get(doc_tier, 3)
        req_rank = _TRUST_TIER_RANK.get(required_tier.lower(), 3)
        trust_tier_violation = doc_rank < req_rank

        if trust_tier_violation and (matched_triggers or imperative_density > 0.08):
            risk_score = min(1.0, risk_score + 0.35)
        elif trust_tier_violation:
            risk_score = max(risk_score, 0.60)
            matched_triggers.append(f"untrusted_tier:{doc_tier}")

        poisoned = risk_score >= self.risk_threshold
        reason = (
            f"Detected RAG poisoning indicators: {', '.join(matched_triggers)}"
            if poisoned
            else "clean"
        )
        return PoisoningScanVerdict(
            doc_id=doc_id,
            poisoned=poisoned,
            risk_score=round(risk_score, 4),
            imperative_density=round(imperative_density, 4),
            matched_triggers=matched_triggers,
            trust_tier_violation=trust_tier_violation,
            reason=reason,
        )

    def sanitize_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        quarantine_poisoned: bool = True,
        redact_triggers: bool = True,
        min_trust_tier: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[PoisoningScanVerdict]]:
        """Filter or redact poisoned documents before they enter the agent's context window."""
        sanitized: list[dict[str, Any]] = []
        verdicts: list[PoisoningScanVerdict] = []

        for doc in documents:
            verdict = self.scan_document(doc, min_trust_tier=min_trust_tier)
            verdicts.append(verdict)
            if verdict.poisoned and quarantine_poisoned:
                continue

            doc_copy = dict(doc)
            doc_copy["_poisoning_verdict"] = verdict.to_dict()
            if verdict.poisoned and redact_triggers:
                redacted_content = str(doc_copy.get("content", ""))
                for _, pattern, _ in _POISON_TRIGGERS:
                    redacted_content = pattern.sub("[REDACTED_RAG_INJECTION]", redacted_content)
                doc_copy["content"] = redacted_content
            sanitized.append(doc_copy)

        return sanitized, verdicts


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _compute_dense_embedding(text: str, dim: int = 64) -> list[float]:
    """Compute a deterministic L2-normalized feature-hash embedding over word and char n-grams."""
    tokens = _tokenize(text)
    features: list[str] = list(tokens)
    # Add word bigrams and character 4-grams for semantic/morphological resilience
    for i in range(len(tokens) - 1):
        features.append(f"{tokens[i]}_{tokens[i + 1]}")
    compact = "".join(tokens)
    for i in range(max(0, len(compact) - 3)):
        features.append(f"#{compact[i : i + 4]}")

    vec = [0.0] * dim
    if not features:
        return vec

    for feat in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        vec = [v / norm for v in vec]
    return vec


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


class HybridRRFCorpusRetriever(RetrievalAdapter):
    """Hybrid Okapi BM25 + Dense Vector Retriever with Reciprocal Rank Fusion and RAG Poisoning Defense."""

    def __init__(
        self,
        retriever_id: str = "hybrid_rrf_retriever",
        documents: list[dict[str, Any]] | None = None,
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        bm25_weight: float = 0.55,
        dense_weight: float = 0.45,
        embedding_dim: int = 64,
        latency_ms: float = 15.0,
        sanitizer: CorpusPoisoningSanitizer | None = None,
        quarantine_poisoned: bool = True,
    ) -> None:
        self._retriever_id = retriever_id
        self._bm25_k1 = float(bm25_k1)
        self._bm25_b = float(bm25_b)
        self._rrf_k = max(1, int(rrf_k))
        self._bm25_weight = float(bm25_weight)
        self._dense_weight = float(dense_weight)
        self._embedding_dim = max(8, int(embedding_dim))
        self._latency_ms = float(latency_ms)
        self._sanitizer = sanitizer or CorpusPoisoningSanitizer()
        self._quarantine_poisoned = quarantine_poisoned

        self._documents: list[dict[str, Any]] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len: float = 1.0
        self._doc_embeddings: list[list[float]] = []
        self._last_scan_verdicts: list[PoisoningScanVerdict] = []

        if documents:
            self.load_corpus(documents)

    @property
    def retriever_id(self) -> str:
        return self._retriever_id

    @property
    def last_scan_verdicts(self) -> list[PoisoningScanVerdict]:
        """Return poisoning scan verdicts from the most recent retrieve() call."""
        return list(self._last_scan_verdicts)

    def load_corpus(self, documents: list[dict[str, Any]]) -> None:
        """Index a document corpus for BM25 and dense vector retrieval."""
        self._documents = [dict(d) for d in documents]
        self._doc_tokens = []
        self._doc_freqs = {}
        self._doc_embeddings = []

        total_len = 0
        for doc in self._documents:
            tags_str = " ".join(str(t) for t in doc.get("tags", []))
            text = f"{doc.get('title', '')} {doc.get('content', '')} {tags_str}"
            tokens = _tokenize(text)
            self._doc_tokens.append(tokens)
            total_len += len(tokens)
            for unique_tok in set(tokens):
                self._doc_freqs[unique_tok] = self._doc_freqs.get(unique_tok, 0) + 1
            self._doc_embeddings.append(
                _compute_dense_embedding(text, dim=self._embedding_dim)
            )

        self._avg_doc_len = (
            total_len / len(self._documents) if self._documents else 1.0
        )

    def _score_bm25(self, query_tokens: list[str], candidate_indices: list[int]) -> dict[int, float]:
        n_docs = len(self._documents)
        scores: dict[int, float] = {}
        for idx in candidate_indices:
            doc_toks = self._doc_tokens[idx]
            doc_len = len(doc_toks)
            if doc_len == 0:
                scores[idx] = 0.0
                continue

            tf_map: dict[str, int] = {}
            for tok in doc_toks:
                tf_map[tok] = tf_map.get(tok, 0) + 1

            score = 0.0
            for q_tok in query_tokens:
                tf = tf_map.get(q_tok, 0)
                if tf == 0:
                    continue
                df = self._doc_freqs.get(q_tok, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self._bm25_k1 * (
                    1.0 - self._bm25_b + self._bm25_b * (doc_len / max(1e-9, self._avg_doc_len))
                )
                score += idf * ((tf * (self._bm25_k1 + 1.0)) / denom)
            scores[idx] = score
        return scores

    def _score_dense(self, query_text: str, candidate_indices: list[int]) -> dict[int, float]:
        q_emb = _compute_dense_embedding(query_text, dim=self._embedding_dim)
        return {
            idx: _cosine_similarity(q_emb, self._doc_embeddings[idx])
            for idx in candidate_indices
        }

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve top_k documents via Hybrid BM25 + Dense RRF and sanitize against RAG poisoning."""
        if not self._documents:
            self._last_scan_verdicts = []
            return RetrievalResult(documents=[], query=query, latency_ms=self._latency_ms)

        candidate_indices = list(range(len(self._documents)))
        if filters and "tags" in filters:
            required_tags = set(filters["tags"])
            filtered = [
                idx
                for idx in candidate_indices
                if required_tags.intersection(set(self._documents[idx].get("tags", [])))
            ]
            if filtered:
                candidate_indices = filtered

        query_tokens = _tokenize(query)
        bm25_scores = self._score_bm25(query_tokens, candidate_indices)
        dense_scores = self._score_dense(query, candidate_indices)

        bm25_ranked = sorted(
            candidate_indices,
            key=lambda i: (bm25_scores[i], self._documents[i].get("relevance_score", 0.0)),
            reverse=True,
        )
        dense_ranked = sorted(
            candidate_indices,
            key=lambda i: (dense_scores[i], self._documents[i].get("relevance_score", 0.0)),
            reverse=True,
        )

        bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
        dense_rank_map = {idx: rank + 1 for rank, idx in enumerate(dense_ranked)}

        rrf_scores: dict[int, float] = {}
        for idx in candidate_indices:
            r_bm25 = bm25_rank_map[idx]
            r_dense = dense_rank_map[idx]
            rrf_scores[idx] = (
                self._bm25_weight / (self._rrf_k + r_bm25)
                + self._dense_weight / (self._rrf_k + r_dense)
            )

        fused_indices = sorted(
            candidate_indices,
            key=lambda i: (rrf_scores[i], bm25_scores[i], dense_scores[i]),
            reverse=True,
        )

        raw_candidates: list[dict[str, Any]] = []
        for idx in fused_indices:
            doc_copy = dict(self._documents[idx])
            doc_copy["bm25_score"] = round(bm25_scores[idx], 4)
            doc_copy["dense_score"] = round(dense_scores[idx], 4)
            doc_copy["rrf_score"] = round(rrf_scores[idx], 6)
            raw_candidates.append(doc_copy)

        min_trust_tier = (
            str(filters["min_trust_tier"])
            if filters and "min_trust_tier" in filters
            else None
        )
        sanitized, verdicts = self._sanitizer.sanitize_documents(
            raw_candidates,
            quarantine_poisoned=self._quarantine_poisoned,
            min_trust_tier=min_trust_tier,
        )
        self._last_scan_verdicts = verdicts

        return RetrievalResult(
            documents=sanitized[: max(1, top_k)],
            query=query,
            latency_ms=self._latency_ms,
        )
