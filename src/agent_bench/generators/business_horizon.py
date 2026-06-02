"""Business long-horizon / SME advisor-like synthetic case generator."""
import random
from datetime import datetime, timezone
from typing import Any

from .base import GenerationResult
from .perturbations import add_ambiguity, add_urgency, apply_noise, inject_distraction, vary_formality

# Hardcoded data pools
_COMPANY_TYPES = ["MEI", "ME", "EPP", "LTDA", "SA"]
_SECTORS = [
    "comercio_varejo", "alimentacao", "servicos_profissionais", "tecnologia",
    "saude", "educacao", "construcao", "transporte", "industria", "agronegocio",
]

_MONTHS_PT = [
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]

_EXPENSE_CATEGORIES = [
    "folha_pagamento", "aluguel", "fornecedores", "impostos", "marketing",
    "energia", "internet_telefone", "manutencao", "transporte", "estoque",
    "servicos_terceiros", "software", "seguros",
]

_REVENUE_ANALYSIS_PROMPTS = [
    "Analise meu faturamento dos ultimos {months} meses e identifique tendencias de crescimento",
    "Quais foram meus melhores e piores meses de faturamento nos ultimos {months} meses?",
    "Meu faturamento caiu nos ultimos {months} meses. O que pode estar acontecendo?",
    "Identifique a sazonalidade do meu negocio com base nos ultimos {months} meses de dados",
    "Compare meu faturamento trimestre a trimestre e mostre a evolucao",
]

_PROJECTION_PROMPTS = [
    "Projete meu fluxo de caixa para os proximos {months} meses considerando crescimento de {growth}% ao mes",
    "Se eu reduzir custos fixos em {reduction}%, como fica meu caixa nos proximos {months} meses?",
    "Quero contratar {hires} funcionarios. Projete o impacto no fluxo de caixa em {months} meses",
    "Considerando a sazonalidade historica, projete meu faturamento para os proximos {months} meses",
    "Se eu investir R${investment} em marketing, qual a projecao de retorno em {months} meses?",
]

_EXPENSE_PROMPTS = [
    "Quais categorias de despesa posso otimizar? Meu custo com fornecedores subiu {pct}%",
    "Preciso reduzir custos em R${amount} por mes. Onde cortar?",
    "Quais dos meus gastos estao acima da media do setor de {sector}?",
    "Minha folha de pagamento representa {pct}% do faturamento. Isso e normal?",
    "Identifique os 3 maiores aumentos de custo nos ultimos {months} meses",
]

_DIAGNOSIS_PROMPTS = [
    "Minha margem de lucro caiu nos ultimos {months} meses. O que pode estar causando?",
    "Tenho fluxo de caixa negativo recorrente. Como resolver?",
    "Meu faturamento cresceu mas o lucro nao acompanhou. Por que?",
    "A inadimplencia dos meus clientes esta em {pct}%. Como reduzir?",
    "Meus custos operacionais crescem mais rapido que a receita. Diagnostico?",
    "Estou com dificuldade de pagar fornecedores em dia. O que fazer?",
]


def _generate_monthly_revenues(rng: random.Random, months: int, base: float, trend: str) -> list[dict[str, Any]]:
    """Generate realistic monthly revenue data."""
    revenues = []
    current = base
    seasonality = [0.85, 0.80, 0.90, 0.95, 1.0, 0.95, 0.90, 1.0, 1.05, 1.10, 1.15, 1.30]

    for i in range(months):
        month_idx = (12 - months + i) % 12
        seasonal_factor = seasonality[month_idx]
        noise = rng.uniform(0.9, 1.1)

        if trend == "growing":
            current *= rng.uniform(1.02, 1.08)
        elif trend == "declining":
            current *= rng.uniform(0.92, 0.99)
        elif trend == "stagnant":
            current *= rng.uniform(0.98, 1.02)
        else:  # volatile
            current *= rng.uniform(0.85, 1.15)

        value = round(current * seasonal_factor * noise, 2)
        revenues.append({"month": _MONTHS_PT[month_idx], "value": value})

    return revenues


def _generate_expense_breakdown(rng: random.Random, revenue: float) -> dict[str, float]:
    """Generate realistic expense breakdown."""
    categories = rng.sample(_EXPENSE_CATEGORIES, k=rng.randint(5, 8))
    expenses = {}
    total_pct = 0.0
    for cat in categories:
        if cat == "folha_pagamento":
            pct = rng.uniform(25, 50)
        elif cat == "aluguel":
            pct = rng.uniform(5, 15)
        elif cat == "fornecedores":
            pct = rng.uniform(10, 30)
        elif cat == "impostos":
            pct = rng.uniform(8, 20)
        else:
            pct = rng.uniform(2, 10)
        expenses[cat] = round(revenue * pct / 100, 2)
        total_pct += pct
    return expenses


class BusinessHorizonGenerator:
    """Generates SME-advisor-like test cases with multi-month business data."""

    FAMILY = "business_long_horizon"
    DOMAIN = "sme_advisor_like"
    SOURCE_TYPE = "synthetic_shadow"

    def __init__(self) -> None:
        self._counter = 0

    def generate(self, *, difficulty: str, seed: int | None = None, **kwargs: Any) -> GenerationResult:
        rng = random.Random(seed)
        scenario_type = kwargs.get("scenario_type") or rng.choice([
            "revenue_analysis", "revenue_analysis",
            "projection", "projection",
            "expense_optimization", "diagnosis",
        ])

        if scenario_type == "revenue_analysis":
            return self._gen_revenue_analysis(rng, difficulty)
        elif scenario_type == "projection":
            return self._gen_projection(rng, difficulty)
        elif scenario_type == "expense_optimization":
            return self._gen_expense_optimization(rng, difficulty)
        elif scenario_type == "diagnosis":
            return self._gen_diagnosis(rng, difficulty)
        else:
            return self._gen_revenue_analysis(rng, difficulty)

    def generate_batch(
        self,
        count: int,
        *,
        difficulty_distribution: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        rng = random.Random(seed)
        dist = difficulty_distribution or {"easy": 0.3, "medium": 0.4, "hard": 0.3}
        difficulties = []
        for diff, weight in dist.items():
            difficulties.extend([diff] * int(weight * count))
        while len(difficulties) < count:
            difficulties.append("medium")
        rng.shuffle(difficulties)

        results = []
        for i, diff in enumerate(difficulties[:count]):
            case_seed = rng.randint(0, 2**31)
            result = self.generate(difficulty=diff, seed=case_seed)
            results.append(result)
        return results

    def _make_task_id(self) -> str:
        self._counter += 1
        return f"SYN_BIZ_{self._counter:04d}"

    def _apply_difficulty_perturbations(self, text: str, difficulty: str, rng: random.Random) -> str:
        if difficulty == "easy":
            return text
        elif difficulty == "medium":
            choice = rng.random()
            if choice < 0.33:
                return apply_noise(text, noise_level=0.1, seed=rng.randint(0, 2**31))
            elif choice < 0.66:
                return vary_formality(text, rng.choice(["informal", "formal"]), seed=rng.randint(0, 2**31))
            else:
                return add_urgency(text, seed=rng.randint(0, 2**31))
        else:  # hard
            text = apply_noise(text, noise_level=0.15, seed=rng.randint(0, 2**31))
            if rng.random() < 0.5:
                text = inject_distraction(text, seed=rng.randint(0, 2**31))
            if rng.random() < 0.3:
                text = add_ambiguity(text, seed=rng.randint(0, 2**31))
            return text

    def _gen_revenue_analysis(self, rng: random.Random, difficulty: str) -> GenerationResult:
        months = rng.choice([3, 6, 9, 12])
        sector = rng.choice(_SECTORS)
        company_type = rng.choice(_COMPANY_TYPES)
        base_revenue = rng.choice([5000, 15000, 30000, 80000, 150000, 500000])
        trend = rng.choice(["growing", "declining", "stagnant", "volatile"])

        revenues = _generate_monthly_revenues(rng, months, base_revenue, trend)

        prompt = rng.choice(_REVENUE_ANALYSIS_PROMPTS).format(months=months)
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Revenue analysis ({trend}, {months}m)",
            description=f"Analyze {months}-month revenue trend for {company_type} in {sector}",
            prompt=prompt,
            initial_state={
                "company_type": company_type,
                "sector": sector,
                "monthly_revenues": revenues,
            },
            expected_final_state={"trend_identified": True, "data_cited": True},
            allowed_tools=["get_financial_data", "calculate_trend", "compare_periods", "get_sector_benchmark"],
            tags=["happy_path", "revenue_analysis"],
            severity="medium",
            criticality="operational",
            difficulty=difficulty,
            refusal_mode="none",
            capabilities=["tool_calling", "numeric_reasoning", "trend_analysis"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "revenue_analysis", "trend": trend},
        )

    def _gen_projection(self, rng: random.Random, difficulty: str) -> GenerationResult:
        months = rng.choice([3, 6, 9, 12])
        growth = round(rng.uniform(1.0, 15.0), 1)
        reduction = rng.randint(5, 30)
        hires = rng.randint(1, 10)
        investment = rng.choice([5000, 10000, 20000, 50000, 100000])

        base_revenue = rng.choice([20000, 50000, 100000, 300000])
        fixed_costs = round(base_revenue * rng.uniform(0.3, 0.6), 2)
        variable_costs = round(base_revenue * rng.uniform(0.1, 0.3), 2)
        balance = round(rng.uniform(10000, 200000), 2)

        history = _generate_monthly_revenues(rng, 6, base_revenue, "growing")

        prompt = rng.choice(_PROJECTION_PROMPTS).format(
            months=months, growth=growth, reduction=reduction,
            hires=hires, investment=investment,
        )
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Cash flow projection ({months}m)",
            description=f"Project cash flow for {months} months with stated assumptions",
            prompt=prompt,
            initial_state={
                "current_balance": balance,
                "monthly_fixed_costs": fixed_costs,
                "monthly_variable_costs": variable_costs,
                "monthly_revenue_avg": base_revenue,
                "historical_data": history,
            },
            expected_final_state={"months_projected": months, "assumptions_stated": True},
            allowed_tools=["get_financial_data", "calculate_projection", "apply_seasonality", "simulate_scenario"],
            tags=["happy_path", "projection"],
            severity="medium",
            criticality="operational",
            difficulty=difficulty,
            refusal_mode="none",
            capabilities=["tool_calling", "numeric_reasoning", "projection"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "projection"},
        )

    def _gen_expense_optimization(self, rng: random.Random, difficulty: str) -> GenerationResult:
        sector = rng.choice(_SECTORS)
        revenue = rng.choice([20000, 50000, 100000, 200000, 500000])
        pct = rng.randint(10, 80)
        amount = rng.choice([1000, 3000, 5000, 10000, 20000, 50000])
        months = rng.choice([3, 6, 12])

        expenses = _generate_expense_breakdown(rng, revenue)

        prompt = rng.choice(_EXPENSE_PROMPTS).format(
            pct=pct, amount=amount, sector=sector, months=months,
        )
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Expense optimization ({sector})",
            description=f"Identify optimization opportunities in expenses for {sector}",
            prompt=prompt,
            initial_state={
                "expense_categories": expenses,
                "sector": sector,
                "revenue": revenue,
                "sector_benchmarks": {cat: round(rng.uniform(5, 30), 1) for cat in expenses},
            },
            expected_final_state={"categories_identified": True, "savings_estimated": True},
            allowed_tools=["get_financial_data", "get_sector_benchmark", "categorize_expenses", "calculate_savings"],
            tags=["happy_path", "optimization"],
            severity="medium",
            criticality="operational",
            difficulty=difficulty,
            refusal_mode="none",
            capabilities=["tool_calling", "numeric_reasoning", "benchmarking"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "expense_optimization"},
        )

    def _gen_diagnosis(self, rng: random.Random, difficulty: str) -> GenerationResult:
        months = rng.choice([3, 6, 9, 12])
        pct = rng.randint(5, 40)
        sector = rng.choice(_SECTORS)
        company_type = rng.choice(_COMPANY_TYPES)
        base_revenue = rng.choice([20000, 50000, 100000, 300000])

        revenues = _generate_monthly_revenues(rng, months, base_revenue, "declining")
        expenses = _generate_expense_breakdown(rng, base_revenue)

        prompt = rng.choice(_DIAGNOSIS_PROMPTS).format(months=months, pct=pct)
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Business diagnosis ({sector})",
            description=f"Diagnose business issues for {company_type} in {sector}",
            prompt=prompt,
            initial_state={
                "financial_summary": {
                    "revenue_trend": revenues,
                    "expenses": expenses,
                    "sector": sector,
                },
                "historical_kpis": {
                    "margin_pct": round(rng.uniform(5, 30), 1),
                    "default_rate_pct": pct if pct < 20 else round(rng.uniform(2, 15), 1),
                    "growth_rate_pct": round(rng.uniform(-10, 5), 1),
                },
                "company_profile": {"type": company_type, "sector": sector},
            },
            expected_final_state={"root_causes_identified": True, "actions_proposed": True},
            allowed_tools=[
                "get_financial_data", "calculate_trend", "get_sector_benchmark",
                "identify_anomalies", "suggest_actions",
            ],
            tags=["happy_path", "diagnosis"],
            severity="high",
            criticality="operational",
            difficulty=difficulty,
            refusal_mode="none",
            capabilities=["tool_calling", "numeric_reasoning", "diagnostic"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "diagnosis"},
        )

    def _build_case(
        self,
        *,
        task_id: str,
        name: str,
        description: str,
        prompt: str,
        initial_state: dict[str, Any],
        expected_final_state: dict[str, Any],
        allowed_tools: list[str],
        tags: list[str],
        severity: str,
        criticality: str,
        difficulty: str,
        refusal_mode: str,
        capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": task_id,
            "name": name,
            "description": description,
            "domain": self.DOMAIN,
            "family": self.FAMILY,
            "source_type": self.SOURCE_TYPE,
            "input_messages": [{"role": "user", "content": prompt}],
            "initial_state": initial_state,
            "expected_final_state": expected_final_state,
            "allowed_tools": allowed_tools,
            "required_capabilities": capabilities or ["tool_calling"],
            "expected_refusal_mode": refusal_mode,
            "severity": severity,
            "business_criticality": criticality,
            "tags": tags,
            "difficulty": difficulty,
            "task_version": "1.0.0",
            "metadata": {
                "created_by": "BusinessHorizonGenerator",
                "generator_model": "deterministic_template",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _validate_case(self, case: dict[str, Any]) -> float:
        """Basic consistency validation."""
        score = 1.0

        if not case.get("input_messages"):
            return 0.0

        prompt = case["input_messages"][0].get("content", "")
        if len(prompt) < 10:
            score -= 0.3

        required = ["task_id", "name", "input_messages", "family", "domain"]
        for f in required:
            if not case.get(f):
                score -= 0.2

        # Business cases should have initial_state with data
        if not case.get("initial_state"):
            score -= 0.2

        return max(0.0, score)
