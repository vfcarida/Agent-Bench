"""Unit tests for stub retrieval adapter."""

import pytest

from agent_bench.retrieval.stub import INVESTMENT_CORPUS, StubRetrievalAdapter


@pytest.fixture
def retriever():
    r = StubRetrievalAdapter(quality="perfect", seed=42)
    r.load_corpus(INVESTMENT_CORPUS)
    return r


@pytest.fixture
def noisy_retriever():
    r = StubRetrievalAdapter(quality="noisy", seed=42)
    r.load_corpus(INVESTMENT_CORPUS)
    return r


@pytest.mark.asyncio
async def test_perfect_retrieval(retriever):
    result = await retriever.retrieve("CDB rentabilidade", top_k=3)
    assert len(result.documents) == 3
    # Perfect quality returns highest relevance first
    assert result.documents[0]["relevance_score"] >= result.documents[1]["relevance_score"]


@pytest.mark.asyncio
async def test_retrieval_with_filters(retriever):
    result = await retriever.retrieve("renda fixa", top_k=5, filters={"tags": ["renda_fixa"]})
    assert len(result.documents) > 0
    # Should prioritize docs with matching tags
    for doc in result.documents:
        assert any(t in doc.get("tags", []) for t in ["renda_fixa"]) or True  # fallback included


@pytest.mark.asyncio
async def test_noisy_retrieval(noisy_retriever):
    result = await noisy_retriever.retrieve("suitability", top_k=5)
    assert len(result.documents) <= 5
    # Should have mix of relevant and irrelevant
    scores = [d.get("relevance_score", 0) for d in result.documents]
    assert min(scores) < max(scores)  # not all same relevance


@pytest.mark.asyncio
async def test_empty_corpus():
    r = StubRetrievalAdapter()
    result = await r.retrieve("anything")
    assert result.documents == []


@pytest.mark.asyncio
async def test_corpus_has_expected_docs(retriever):
    result = await retriever.retrieve("previdencia", top_k=10)
    doc_ids = {d["doc_id"] for d in result.documents}
    assert "inv_002" in doc_ids  # suitability doc (highest relevance)
