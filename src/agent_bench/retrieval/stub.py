"""Stub retrieval adapter for benchmark simulation."""

import random
from typing import Any

from agent_bench.core.adapters import RetrievalAdapter, RetrievalResult


class StubRetrievalAdapter(RetrievalAdapter):
    """Configurable stub retrieval that returns pre-loaded documents.

    Simulates different retrieval qualities:
    - perfect: always returns relevant docs
    - noisy: mixes relevant + irrelevant
    - poor: mostly irrelevant docs
    """

    def __init__(
        self,
        retriever_id: str = "stub_retriever",
        documents: list[dict[str, Any]] | None = None,
        quality: str = "perfect",
        latency_ms: float = 50.0,
        seed: int | None = 42,
    ):
        self._retriever_id = retriever_id
        self._documents = documents or []
        self._quality = quality
        self._latency_ms = latency_ms
        self._rng = random.Random(seed)

    @property
    def retriever_id(self) -> str:
        return self._retriever_id

    def load_corpus(self, documents: list[dict[str, Any]]) -> None:
        """Load a corpus of documents for retrieval."""
        self._documents = documents

    async def retrieve(
        self, query: str, *, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> RetrievalResult:
        if not self._documents:
            return RetrievalResult(documents=[], query=query, latency_ms=self._latency_ms)

        # Filter by tags if requested
        candidates = self._documents
        if filters and "tags" in filters:
            required_tags = set(filters["tags"])
            candidates = [
                d for d in candidates
                if required_tags.intersection(set(d.get("tags", [])))
            ] or candidates

        # Simulate quality
        if self._quality == "perfect":
            # Return top_k most relevant (by relevance_score field or order)
            sorted_docs = sorted(
                candidates, key=lambda d: d.get("relevance_score", 0.5), reverse=True
            )
            results = sorted_docs[:top_k]
        elif self._quality == "noisy":
            # Mix relevant and random
            relevant = [d for d in candidates if d.get("relevance_score", 0) > 0.5]
            irrelevant = [d for d in candidates if d.get("relevance_score", 0) <= 0.5]
            n_relevant = min(len(relevant), max(1, top_k // 2))
            n_noise = top_k - n_relevant
            results = (
                self._rng.sample(relevant, min(n_relevant, len(relevant)))
                + self._rng.sample(irrelevant, min(n_noise, len(irrelevant)))
            )
        else:  # poor
            self._rng.shuffle(candidates)
            results = candidates[:top_k]

        return RetrievalResult(
            documents=results,
            query=query,
            latency_ms=self._latency_ms,
        )


# Pre-built investment corpus for investment_advisor domain
INVESTMENT_CORPUS: list[dict[str, Any]] = [
    {
        "doc_id": "inv_001",
        "title": "CDB Pós-fixado - Características",
        "content": (
            "O CDB pós-fixado rende com base no CDI. Rentabilidade típica: 100-120% CDI. "
            "Garantido pelo FGC até R$250.000 por CPF/instituição. Liquidez diária ou no vencimento. "
            "IR regressivo: 22,5% (até 180 dias) a 15% (acima de 720 dias). "
            "Indicado para perfil conservador e reserva de emergência."
        ),
        "tags": ["renda_fixa", "cdb", "conservador"],
        "relevance_score": 0.9,
        "source": "manual_produtos_rf_v3",
    },
    {
        "doc_id": "inv_002",
        "title": "Suitability - Classificação de Perfil",
        "content": (
            "Perfil Conservador: até 20% em renda variável. "
            "Perfil Moderado: até 40% em renda variável. "
            "Perfil Arrojado: até 70% em renda variável. "
            "Perfil Agressivo: sem limite em renda variável. "
            "Reclassificação obrigatória a cada 24 meses ou após mudança patrimonial significativa."
        ),
        "tags": ["suitability", "perfil", "regulatorio"],
        "relevance_score": 0.95,
        "source": "politica_suitability_v2",
    },
    {
        "doc_id": "inv_003",
        "title": "Fundos Multimercado - Política de Investimento",
        "content": (
            "Fundos multimercado podem investir em renda fixa, variável, câmbio e derivativos. "
            "Taxa de administração típica: 1,5-2,5% a.a. Taxa de performance: 20% sobre CDI. "
            "Indicado para perfil moderado a arrojado. Liquidez: D+1 a D+30 dependendo do fundo. "
            "Não possui garantia do FGC."
        ),
        "tags": ["fundos", "multimercado", "moderado"],
        "relevance_score": 0.8,
        "source": "catalogo_fundos_v4",
    },
    {
        "doc_id": "inv_004",
        "title": "Tesouro Direto - Títulos Públicos",
        "content": (
            "Tesouro Selic: pós-fixado, liquidez D+1, ideal para reserva. "
            "Tesouro IPCA+: protege contra inflação, vencimentos longos. "
            "Tesouro Prefixado: taxa definida na compra, risco de marcação a mercado. "
            "Custódia: 0,2% a.a. (B3). IR regressivo. Garantia soberana."
        ),
        "tags": ["renda_fixa", "tesouro", "conservador"],
        "relevance_score": 0.85,
        "source": "guia_tesouro_v2",
    },
    {
        "doc_id": "inv_005",
        "title": "Política de Concentração",
        "content": (
            "Limite máximo por emissor: 20% do patrimônio investido. "
            "Limite máximo por classe de ativo: conforme perfil de suitability. "
            "Exposição cambial máxima sem hedge: 10% para conservador, 30% para agressivo. "
            "Vedado: concentração superior a 50% em único ativo para qualquer perfil."
        ),
        "tags": ["politica", "concentracao", "regulatorio"],
        "relevance_score": 0.75,
        "source": "politica_investimentos_v5",
    },
    {
        "doc_id": "inv_006",
        "title": "Cálculo de Rentabilidade",
        "content": (
            "Rentabilidade bruta = (valor_final / valor_inicial - 1) * 100. "
            "Rentabilidade líquida = bruta - IR - IOF (se < 30 dias). "
            "CDI acumulado: produto de (1 + taxa_diaria) para cada dia útil. "
            "Benchmark: comparar sempre com CDI, IPCA+, e Ibovespa conforme classe."
        ),
        "tags": ["calculo", "rentabilidade", "metodologia"],
        "relevance_score": 0.7,
        "source": "manual_calculos_v1",
    },
    {
        "doc_id": "inv_007",
        "title": "Ações e ETFs - Renda Variável",
        "content": (
            "Ações: participação em empresas, risco de mercado, sem garantia de retorno. "
            "ETFs: fundos de índice negociados em bolsa, diversificação automática. "
            "IR sobre ganho de capital: 15% (swing trade), 20% (day trade). "
            "Isenção: vendas até R$20.000/mês em ações (não ETFs)."
        ),
        "tags": ["renda_variavel", "acoes", "etf", "arrojado"],
        "relevance_score": 0.7,
        "source": "guia_rv_v3",
    },
    {
        "doc_id": "inv_008",
        "title": "Previdência Privada - PGBL e VGBL",
        "content": (
            "PGBL: deduz contribuições do IR (até 12% da renda bruta), IR sobre valor total no resgate. "
            "VGBL: não deduz do IR, IR apenas sobre rendimento. "
            "Tabelas: progressiva (até 27,5%) ou regressiva (de 35% a 10% em 10+ anos). "
            "Portabilidade: permitida sem incidência de IR."
        ),
        "tags": ["previdencia", "pgbl", "vgbl", "longo_prazo"],
        "relevance_score": 0.65,
        "source": "manual_previdencia_v2",
    },
    {
        "doc_id": "noise_001",
        "title": "Política de Segurança da Informação",
        "content": "Documento sobre controles de acesso e autenticação multifator.",
        "tags": ["seguranca", "ti"],
        "relevance_score": 0.1,
        "source": "politica_si_v8",
    },
    {
        "doc_id": "noise_002",
        "title": "Horários de Funcionamento das Agências",
        "content": "Agências funcionam de segunda a sexta, 10h às 16h. SAC 24h.",
        "tags": ["operacional", "agencias"],
        "relevance_score": 0.05,
        "source": "manual_operacional_v1",
    },
]
