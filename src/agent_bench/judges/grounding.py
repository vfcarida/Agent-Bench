"""Grounding judge: verifies response is supported by retrieved documents.

Enhanced with FinanceBench-inspired (arXiv:2311.11944) evidence strings:
- When evidence_strings are available on the task, each expected claim is
  verified against a specific evidence excerpt from a source document.
- Claims that don't map to any evidence string are flagged as potential
  hallucinations (the dominant failure mode identified by FinanceBench).
- Falls back to corpus keyword matching when no evidence strings are defined.
"""

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
    - (NEW) Evidence strings verify specific claim-to-source mappings
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

        # Use evidence strings if available (FinanceBench approach)
        evidence_strings = getattr(task, "evidence_strings", [])
        if evidence_strings:
            return self._evaluate_with_evidence_strings(
                response, evidence_strings, retrieved_docs
            )

        # Fall back to corpus-based grounding
        return self._evaluate_with_corpus(response, retrieved_docs)

    def _evaluate_with_evidence_strings(
        self,
        response: str,
        evidence_strings: list[dict[str, str]],
        retrieved_docs: list[dict[str, Any]],
    ) -> JudgeVerdict:
        """FinanceBench-style: verify each claim against its specific evidence string."""
        verified_claims = 0
        unverified_claims: list[dict[str, str]] = []
        hallucination_suspects: list[str] = []

        response_lower = response.lower()

        for evidence_entry in evidence_strings:
            claim = evidence_entry.get("claim", "")
            evidence = evidence_entry.get("evidence", "")
            source_doc_id = evidence_entry.get("source_doc_id", "")

            if not claim:
                continue

            # Check 1: Is the claim present in the response?
            claim_present = _fuzzy_contains(response_lower, claim.lower())

            if not claim_present:
                # Claim not in response — might be missing information
                unverified_claims.append({
                    "claim": claim,
                    "reason": "claim_not_in_response",
                })
                continue

            # Check 2: Is the evidence present in the retrieved docs?
            evidence_found = False
            if evidence:
                for doc in retrieved_docs:
                    doc_content = doc.get("content", "").lower()
                    if _fuzzy_contains(doc_content, evidence.lower()):
                        evidence_found = True
                        break

            # Check 3: Does the source_doc_id match a retrieved doc?
            source_found = True
            if source_doc_id:
                source_found = any(
                    doc.get("doc_id", "") == source_doc_id or
                    doc.get("source", "") == source_doc_id
                    for doc in retrieved_docs
                )

            if evidence_found or (not evidence and source_found):
                verified_claims += 1
            else:
                unverified_claims.append({
                    "claim": claim,
                    "reason": "evidence_not_in_retrieved_docs",
                    "expected_source": source_doc_id,
                })

        # Check for extra claims in response not covered by evidence strings
        response_claims = _extract_claims(response)
        evidence_claim_texts = {e.get("claim", "").lower() for e in evidence_strings}
        for rc in response_claims:
            if not any(_fuzzy_contains(rc.lower(), ec) for ec in evidence_claim_texts if ec):
                hallucination_suspects.append(rc)

        total = len(evidence_strings) if evidence_strings else 1
        grounding_score = verified_claims / total if total > 0 else 0.0

        # Compute hallucination rate (FinanceBench key metric)
        total_response_claims = len(response_claims) if response_claims else 1
        hallucination_rate = len(hallucination_suspects) / total_response_claims

        # Combined score: 70% grounding, 30% non-hallucination
        final_score = grounding_score * 0.7 + (1.0 - hallucination_rate) * 0.3
        passed = final_score >= 0.7

        return JudgeVerdict(
            score=final_score,
            passed=passed,
            reasoning=(
                f"Evidence-based grounding: {verified_claims}/{total} claims verified. "
                f"Hallucination rate: {hallucination_rate:.1%} "
                f"({len(hallucination_suspects)} suspect claims). "
                f"Unverified: {[u['claim'][:50] for u in unverified_claims[:3]]}"
            ),
            judge_id=self.judge_id,
            criteria="grounding",
            metadata={
                "mode": "evidence_strings",
                "verified_claims": verified_claims,
                "total_evidence_strings": total,
                "grounding_score": grounding_score,
                "hallucination_rate": hallucination_rate,
                "hallucination_suspects": hallucination_suspects[:5],
                "unverified_claims": unverified_claims[:5],
            },
        )

    def _evaluate_with_corpus(
        self,
        response: str,
        retrieved_docs: list[dict[str, Any]],
    ) -> JudgeVerdict:
        """Original corpus-based grounding (keyword overlap)."""
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

        # Compute hallucination rate
        hallucination_rate = 1.0 - (grounded_count / len(claims)) if claims else 0.0

        return JudgeVerdict(
            score=final_score,
            passed=passed,
            reasoning=(
                f"Grounding: {grounded_count}/{len(claims)} claims supported. "
                f"Citation accuracy: {citation_score:.2f}. "
                f"Hallucination rate: {hallucination_rate:.1%}. "
                f"Ungrounded: {ungrounded_claims[:3]}"
            ),
            judge_id=self.judge_id,
            criteria="grounding",
            metadata={
                "mode": "corpus",
                "total_claims": len(claims),
                "grounded_claims": grounded_count,
                "citation_score": citation_score,
                "hallucination_rate": hallucination_rate,
                "ungrounded_sample": ungrounded_claims[:5],
            },
        )


def _fuzzy_contains(haystack: str, needle: str, min_overlap: float = 0.6) -> bool:
    """Check if needle is approximately contained in haystack."""
    if not needle or not haystack:
        return False
    # Direct substring match
    if needle in haystack:
        return True
    # Word-level overlap
    needle_words = set(needle.split()) - _STOP_WORDS
    haystack_words = set(haystack.split())
    if not needle_words:
        return True
    overlap = len(needle_words & haystack_words) / len(needle_words)
    return overlap >= min_overlap


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
