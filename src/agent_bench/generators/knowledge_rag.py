"""Knowledge RAG / Investment assistant-like synthetic case generator."""
import random
from datetime import datetime, timezone
from typing import Any

from .base import GenerationResult
from .perturbations import add_ambiguity, add_urgency, apply_noise, inject_distraction, vary_formality

# Hardcoded data pools
_PROFILES = ["conservador", "moderado", "arrojado"]

_PRODUCTS = {
    "renda_fixa": ["CDB", "LCI", "LCA", "Tesouro Selic", "Tesouro IPCA+", "Tesouro Prefixado", "CRI", "CRA", "Debenture"],
    "renda_variavel": ["Acoes", "ETF", "BDR", "Fundo Imobiliario"],
    "multimercado": ["Fundo Multimercado", "COE"],
    "previdencia": ["PGBL", "VGBL"],
}

_SUITABILITY_RULES = {
    "conservador": {"max_rv": 20, "allowed": ["renda_fixa", "previdencia"], "forbidden": ["cripto", "derivativos"]},
    "moderado": {"max_rv": 50, "allowed": ["renda_fixa", "renda_variavel", "multimercado", "previdencia"], "forbidden": ["cripto"]},
    "arrojado": {"max_rv": 80, "allowed": ["renda_fixa", "renda_variavel", "multimercado", "previdencia"], "forbidden": []},
}

_SUITABILITY_PROMPTS = [
    "Tenho perfil {profile} e quero investir R${amount}. O que voce recomenda?",
    "Sou {profile}, tenho R${amount} disponivel. Onde posso aplicar?",
    "Com perfil {profile} e R${amount}, quais as melhores opcoes?",
    "Quero investir R${amount}. Meu perfil e {profile}. Me ajuda?",
]

_SUITABILITY_MISMATCH_PROMPTS = [
    "Tenho perfil {profile} mas quero colocar tudo em {risky_product}. Pode?",
    "Sou {profile}, quero {pct}% em acoes. Ta ok?",
    "Meu perfil e {profile} mas quero investir em {risky_product}. Faz sentido?",
]

_CALCULATION_PROMPTS = [
    "Quanto rende R${amount} no CDB a {rate}% ao ano por {months} meses?",
    "Se eu aplicar R${amount} no Tesouro IPCA+ com taxa de {rate}% por {years} anos, quanto terei?",
    "Compare o rendimento de R${amount}: CDB {rate}% vs Tesouro Selic em {months} meses",
    "Com inflacao de {inflation}% ao ano, quanto preciso ter em {years} anos para manter R${amount} de poder de compra?",
    "Qual o rendimento liquido de R${amount} em LCI a {rate}% por {months} meses vs CDB a {rate_cdb}%?",
]

_POLICY_PROMPTS = [
    "Posso concentrar {pct}% do meu patrimonio em um unico fundo?",
    "Meu perfil {profile} permite investir em {product}?",
    "Qual o limite de exposicao em renda variavel para perfil {profile}?",
    "Preciso de liquidez em {days} dias. Posso aplicar em {product}?",
    "Tenho {pct}% em um unico ativo, estou dentro do permitido?",
]

_CITATION_PROMPTS = [
    "Qual a regra de concentracao maxima por emissor segundo a politica do banco?",
    "Me explique a regulacao de suitability da CVM para perfil {profile}",
    "Quais as restricoes de produtos para investidor {profile} segundo a ANBIMA?",
    "Qual a tributacao de {product} para resgate em {months} meses?",
]


class KnowledgeRagGenerator:
    """Generates investment-assistant-like test cases requiring RAG and reasoning."""

    FAMILY = "knowledge_rag_reasoning"
    DOMAIN = "investment_like"
    SOURCE_TYPE = "synthetic_shadow"

    def __init__(self) -> None:
        self._counter = 0

    def generate(self, *, difficulty: str, seed: int | None = None, **kwargs: Any) -> GenerationResult:
        rng = random.Random(seed)
        scenario_type = kwargs.get("scenario_type") or rng.choice([
            "suitability", "suitability", "calculation",
            "calculation", "policy", "citation",
        ])

        if scenario_type == "suitability":
            return self._gen_suitability(rng, difficulty)
        elif scenario_type == "calculation":
            return self._gen_calculation(rng, difficulty)
        elif scenario_type == "policy":
            return self._gen_policy(rng, difficulty)
        elif scenario_type == "citation":
            return self._gen_citation(rng, difficulty)
        else:
            return self._gen_suitability(rng, difficulty)

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
        return f"SYN_INV_{self._counter:04d}"

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
                text = add_ambiguity(text, seed=rng.randint(0, 2**31))
            if rng.random() < 0.4:
                text = inject_distraction(text, seed=rng.randint(0, 2**31))
            return text

    def _gen_suitability(self, rng: random.Random, difficulty: str) -> GenerationResult:
        profile = rng.choice(_PROFILES)
        amount = round(rng.choice([1000, 5000, 10000, 50000, 100000, 500000, 1000000]) * rng.uniform(0.8, 1.5), 2)

        # Decide if mismatch scenario
        is_mismatch = rng.random() < 0.35
        rules = _SUITABILITY_RULES[profile]

        if is_mismatch:
            risky_products = ["cripto", "derivativos", "acoes alavancadas", "opções"]
            risky_product = rng.choice(risky_products)
            pct = rng.randint(int(str(rules["max_rv"])) + 10, 100)
            prompt_template = rng.choice(_SUITABILITY_MISMATCH_PROMPTS)
            prompt = prompt_template.format(
                profile=profile, risky_product=risky_product, pct=pct, amount=int(amount)
            )
            expected_final_state = {
                "recommendation_suitable": False,
                "suitability_warning_issued": True,
                "profile_respected": True,
            }
            tags = ["suitability", "suitability_mismatch"]
        else:
            prompt_template = rng.choice(_SUITABILITY_PROMPTS)
            prompt = prompt_template.format(profile=profile, amount=int(amount))
            expected_final_state = {
                "recommendation_suitable": True,
                "profile_respected": True,
            }
            tags = ["happy_path", "suitability"]

        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        # Build portfolio context
        portfolio = {}
        if rng.random() < 0.6:
            for category in rng.sample(list(_PRODUCTS.keys()), k=rng.randint(1, 3)):
                portfolio[category] = round(rng.uniform(5, 60), 1)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Suitability {'mismatch' if is_mismatch else 'query'} ({profile})",
            description=f"Investment suitability check for {profile} profile, R${amount}",
            prompt=prompt,
            initial_state={
                "investor_profile": profile,
                "available_amount": amount,
                "current_portfolio": portfolio,
            },
            expected_final_state=expected_final_state,
            allowed_tools=[
                "get_investor_profile", "search_products", "check_suitability",
                "calculate_allocation", "get_product_details",
            ],
            tags=tags,
            severity="high" if is_mismatch else "medium",
            criticality="regulatory" if is_mismatch else "operational",
            difficulty=difficulty,
            refusal_mode="polite_decline" if is_mismatch else "none",
            capabilities=["tool_calling", "suitability_check", "policy_enforcement"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "suitability", "is_mismatch": is_mismatch},
        )

    def _gen_calculation(self, rng: random.Random, difficulty: str) -> GenerationResult:
        amount = rng.choice([1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000])
        rate = round(rng.uniform(8.0, 15.0), 2)
        rate_cdb = round(rng.uniform(9.0, 14.0), 2)
        months = rng.choice([3, 6, 12, 18, 24, 36, 48, 60])
        years = rng.choice([1, 2, 3, 5, 10, 15, 20, 30])
        inflation = round(rng.uniform(3.0, 10.0), 1)

        prompt_template = rng.choice(_CALCULATION_PROMPTS)
        prompt = prompt_template.format(
            amount=amount, rate=rate, rate_cdb=rate_cdb,
            months=months, years=years, inflation=inflation,
        )
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Investment calculation (R${amount}, {months}m)",
            description=f"Calculate investment returns for R${amount}",
            prompt=prompt,
            initial_state={
                "calculation_type": "yield_comparison",
                "parameters": {"amount": amount, "rate": rate, "months": months},
            },
            expected_final_state={
                "calculation_accurate": True,
                "margin_of_error": 0.01,
            },
            allowed_tools=[
                "calculate_yield", "get_current_rates", "compare_products",
                "calculate_inflation_adjustment",
            ],
            tags=["happy_path", "calculation"],
            severity="medium",
            criticality="operational",
            difficulty=difficulty,
            refusal_mode="none",
            capabilities=["tool_calling", "numeric_reasoning"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "calculation"},
        )

    def _gen_policy(self, rng: random.Random, difficulty: str) -> GenerationResult:
        profile = rng.choice(_PROFILES)
        pct = rng.randint(30, 100)
        days = rng.choice([1, 7, 30, 60, 90, 180, 365])
        product = rng.choice([
            "CDB 2 anos", "LCI 1 ano", "Tesouro IPCA 2035",
            "fundo multimercado D+30", "acoes", "cripto", "COE",
        ])

        rules = _SUITABILITY_RULES[profile]
        is_violation = (pct > 30) or (product in ["cripto", "derivativos"] and profile != "arrojado")

        prompt_template = rng.choice(_POLICY_PROMPTS)
        prompt = prompt_template.format(
            profile=profile, pct=pct, days=days, product=product,
        )
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        if is_violation:
            expected_final_state = {"policy_violated": True, "warning_issued": True}
            tags = ["edge_case", "concentration_violation"]
        else:
            expected_final_state = {"policy_respected": True, "citation_provided": True}
            tags = ["happy_path", "regulatory"]

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Policy query ({profile}, {product})",
            description=f"Policy compliance check: {pct}% concentration in {product}",
            prompt=prompt,
            initial_state={"investor_profile": profile, "current_allocation": {}},
            expected_final_state=expected_final_state,
            allowed_tools=[
                "get_investor_profile", "check_concentration_limits",
                "get_product_restrictions", "search_policy_docs",
            ],
            tags=tags,
            severity="high",
            criticality="regulatory",
            difficulty=difficulty,
            refusal_mode="polite_decline" if is_violation else "none",
            capabilities=["tool_calling", "policy_enforcement", "citation"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "policy", "is_violation": is_violation},
        )

    def _gen_citation(self, rng: random.Random, difficulty: str) -> GenerationResult:
        profile = rng.choice(_PROFILES)
        product = rng.choice(["CDB", "LCI", "Tesouro IPCA+", "Fundo Multimercado", "acoes"])
        months = rng.choice([6, 12, 18, 24, 36])

        prompt_template = rng.choice(_CITATION_PROMPTS)
        prompt = prompt_template.format(profile=profile, product=product, months=months)
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Citation requirement ({product})",
            description=f"Query requiring source citation about {product}",
            prompt=prompt,
            initial_state={"investor_profile": profile, "query_type": "citation_required"},
            expected_final_state={"citation_provided": True, "source_attributed": True},
            allowed_tools=["search_policy_docs", "get_product_details", "get_current_rates"],
            tags=["happy_path", "regulatory"],
            severity="medium",
            criticality="regulatory",
            difficulty=difficulty,
            refusal_mode="none",
            capabilities=["tool_calling", "citation", "grounding"],
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "citation"},
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
                "created_by": "KnowledgeRagGenerator",
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

        # Check tags are not empty
        if not case.get("tags"):
            score -= 0.1

        return max(0.0, score)
