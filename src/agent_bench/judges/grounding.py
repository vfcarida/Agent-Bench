"""Grounding judge: verifies response is supported by retrieved documents."""

import re
from typing import Any

from agent_bench.core.adapters import JudgeVerdict
from agent_bench.core.artifacts import TraceEvent, TraceEventType
from agent_bench.core.scenarios import Task


class GroundingJudge:
    """Evaluates whether the response is grounded in retrieved documents.

    Checks:
    - Claims in response have supporting evidence in retrieved docs
    - Citations reference actual retrieved documents
    - No hallucinated facts beyond what docs provide
    """

    @property
    def judge_id(self) -> str:
        return "grounding_verifier"

    @property
    def judge_type(self) -> str:
        return "grounding"

    async def evaluate(
        self,
        task: Task,
        result: dict[str, Any],
        traces: list[TraceEvent],
    ) -> JudgeVerdict:
        response = result.get("response", "")
        retrieved_docs = result.get("retrieved_documents", [])

        # Also check traces for retrieval results
        if not retrieved_docs:
            for trace in traces:
                if trace.event_type == TraceEventType.RETRIEVAL_RESULT:
                    retrieved_docs = trace.data.get("documents", [])
                    break

        if not retrieved_docs:
            return JudgeVerdict(
                score=0.0,
                passed=False,
                reasoning="No retrieved documents found to ground response.",
                judge_id=self.judge_id,
                criteria="grounding",
            )

        # Build corpus text for matching
        corpus_text = " ".join(
            doc.get("content", "") + " " + doc.get("title", "")
            for doc in retrieved_docs
        ).lower()

        # Extract claims from response (sentences with factual content)
        claims = _extract_claims(response)
        if not claims:
            return JudgeVerdict(
                score=1.0,
                passed=True,
                reasoning="No factual claims to verify.",
                judge_id=self.judge_id,
                criteria="grounding",
            )

        # Check grounding of each claim
        grounded_count = 0
        ungrounded_claims = []
        for claim in claims:
            if _is_grounded(claim, corpus_text):
                grounded_count += 1
            else:
                ungrounded_claims.append(claim)

        score = grounded_count / len(claims)
        passed = score >= 0.7  # 70% grounding threshold

        # Check citation accuracy
        citations = _extract_citations(response)
        citation_score = _verify_citations(citations, retrieved_docs)

        # Combine scores
        final_score = score * 0.7 + citation_score * 0.3 if citations else score
        passed = final_score >= 0.7

        return JudgeVerdict(
            score=final_score,
            passed=passed,
            reasoning=(
                f"Grounding: {grounded_count}/{len(claims)} claims supported. "
                f"Citation accuracy: {citation_score:.2f}. "
                f"Ungrounded: {ungrounded_claims[:3]}"
            ),
            judge_id=self.judge_id,
            criteria="grounding",
            metadata={
                "total_claims": len(claims),
                "grounded_claims": grounded_count,
                "citation_score": citation_score,
                "ungrounded_sample": ungrounded_claims[:5],
            },
        )


def _extract_claims(response: str) -> list[str]:
    """Extract factual claims (sentences with numbers, percentages, or specific terms)."""
    sentences = re.split(r'[.!?\n]+', response)
    claims = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 10:
            continue
        # Sentences with numbers, percentages, or specific financial terms are claims
        if re.search(r'\d+|%|R\$|CDI|IPCA|FGC|IR|IOF', s):
            claims.append(s)
    return claims


def _is_grounded(claim: str, corpus_text: str) -> bool:
    """Check if a claim has key terms supported by corpus."""
    claim_lower = claim.lower()
    # Extract key numeric/factual fragments
    fragments = re.findall(r'\d+[\.,]?\d*\s*%?|R\$\s*[\d\.,]+|[A-Z]{2,}', claim)
    if not fragments:
        # Non-numeric claim: check for keyword overlap
        words = set(claim_lower.split()) - _STOP_WORDS
        if not words:
            return True
        overlap = sum(1 for w in words if w in corpus_text)
        return overlap / len(words) >= 0.3

    # Check numeric fragments presence in corpus
    found = sum(1 for f in fragments if f.lower() in corpus_text)
    return found / len(fragments) >= 0.5


def _extract_citations(response: str) -> list[str]:
    """Extract citation references like [source_name] or (Fonte: X)."""
    patterns = [
        r'\[([^\]]+)\]',
        r'\(Fonte:\s*([^)]+)\)',
        r'Fonte:\s*(\S+)',
        r'Ref:\s*(\S+)',
    ]
    citations = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, response))
    return citations


def _verify_citations(citations: list[str], documents: list[dict[str, Any]]) -> float:
    """Verify cited sources exist in retrieved documents."""
    if not citations:
        return 1.0
    doc_sources = set()
    for doc in documents:
        doc_sources.add(doc.get("source", "").lower())
        doc_sources.add(doc.get("doc_id", "").lower())
        doc_sources.add(doc.get("title", "").lower())

    verified = sum(
        1 for c in citations
        if any(c.lower() in s or s in c.lower() for s in doc_sources if s)
    )
    return verified / len(citations)


_STOP_WORDS = {
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "por", "para", "com", "um", "uma", "uns", "umas", "e", "ou",
    "que", "se", "ao", "aos", "é", "ser", "ter", "este", "esta", "esse", "essa",
    "the", "is", "are", "and", "or", "of", "to", "in", "for", "on", "with",
}
