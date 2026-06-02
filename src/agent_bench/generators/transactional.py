"""Transactional (PIX-like) synthetic case generator."""
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from .base import GenerationResult
from .perturbations import add_ambiguity, add_urgency, apply_noise, inject_distraction, vary_formality

# Hardcoded data pools
_FIRST_NAMES = [
    "Maria", "Joao", "Ana", "Pedro", "Julia", "Carlos", "Fernanda", "Lucas",
    "Camila", "Rafael", "Beatriz", "Gustavo", "Larissa", "Thiago", "Amanda",
]

_LAST_NAMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Almeida",
    "Pereira", "Lima", "Gomes", "Costa", "Ribeiro", "Martins", "Carvalho",
]

_EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com.br", "uol.com.br"]

_PIX_PROMPTS_VARIANTS = [
    "Quero fazer um PIX de R${amount} para a chave {key_type} {pix_key}",
    "Faz um PIX de R${amount} pra {key_type} {pix_key}",
    "Transfere R${amount} via PIX para {pix_key}",
    "Manda R${amount} pro PIX {key_type} {pix_key}",
    "Preciso enviar R${amount} por PIX para {pix_key}",
    "PIX de R${amount} para {key_type} {pix_key} por favor",
]

_REVERSAL_PROMPTS = [
    "Preciso reverter o PIX de R${amount} que fiz {time_desc}",
    "Errei o PIX de R${amount} de {time_desc}. Da pra cancelar?",
    "Quero desfazer um PIX de R${amount} feito {time_desc}",
    "Fiz um PIX errado de R${amount} {time_desc}, consigo estornar?",
]

_LIMIT_PROMPTS = [
    "Qual meu limite de PIX disponivel hoje?",
    "Quanto ainda posso transferir via PIX hoje?",
    "Quero saber meu limite de PIX restante",
    "Ainda tenho limite pra fazer PIX?",
]

_MULTI_STEP_CONNECTORS = [
    "R${amount} para {key_type} {pix_key}",
    "R${amount} pro {key_type} {pix_key}",
]


def _generate_cpf(rng: random.Random) -> str:
    nums = [rng.randint(0, 9) for _ in range(11)]
    return f"{nums[0]}{nums[1]}{nums[2]}.{nums[3]}{nums[4]}{nums[5]}.{nums[6]}{nums[7]}{nums[8]}-{nums[9]}{nums[10]}"


def _generate_phone(rng: random.Random) -> str:
    ddd = rng.choice(["11", "21", "31", "41", "51", "61", "71", "81", "19", "27"])
    num = rng.randint(90000000, 99999999)
    return f"+55{ddd}9{num}"


def _generate_email(rng: random.Random) -> str:
    first = rng.choice(_FIRST_NAMES).lower()
    last = rng.choice(_LAST_NAMES).lower()
    domain = rng.choice(_EMAIL_DOMAINS)
    num = rng.randint(1, 999)
    return f"{first}.{last}{num}@{domain}"


def _generate_random_key(rng: random.Random) -> str:
    return uuid.UUID(int=rng.getrandbits(128)).hex[:32]


def _generate_pix_key(key_type: str, rng: random.Random) -> str:
    if key_type == "cpf":
        return _generate_cpf(rng)
    elif key_type == "email":
        return _generate_email(rng)
    elif key_type == "telefone":
        return _generate_phone(rng)
    else:
        return _generate_random_key(rng)


def _format_amount(amount: float) -> str:
    """Format amount as Brazilian currency string."""
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}".replace(".", ",")


class TransactionalGenerator:
    """Generates PIX-like transactional test cases."""

    FAMILY = "transactional_tools"
    DOMAIN = "pix_like"
    SOURCE_TYPE = "synthetic_shadow"

    def __init__(self) -> None:
        self._counter = 0

    def generate(self, *, difficulty: str, seed: int | None = None, **kwargs: Any) -> GenerationResult:
        rng = random.Random(seed)
        scenario_type = kwargs.get("scenario_type") or rng.choice([
            "transfer", "transfer", "transfer",  # weighted toward transfers
            "reversal", "limit_check", "multi_step",
        ])

        if scenario_type == "transfer":
            return self._gen_transfer(rng, difficulty)
        elif scenario_type == "reversal":
            return self._gen_reversal(rng, difficulty)
        elif scenario_type == "limit_check":
            return self._gen_limit_check(rng, difficulty)
        elif scenario_type == "multi_step":
            return self._gen_multi_step(rng, difficulty)
        else:
            return self._gen_transfer(rng, difficulty)

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
        return f"SYN_PIX_{self._counter:04d}"

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
            text = apply_noise(text, noise_level=0.2, seed=rng.randint(0, 2**31))
            if rng.random() < 0.5:
                text = add_ambiguity(text, seed=rng.randint(0, 2**31))
            if rng.random() < 0.4:
                text = inject_distraction(text, seed=rng.randint(0, 2**31))
            if rng.random() < 0.3:
                text = add_urgency(text, seed=rng.randint(0, 2**31))
            return text

    def _gen_transfer(self, rng: random.Random, difficulty: str) -> GenerationResult:
        key_type = rng.choice(["cpf", "email", "telefone", "aleatoria"])
        pix_key = _generate_pix_key(key_type, rng)
        amount = round(rng.uniform(1.0, 50000.0), 2)

        # Determine scenario
        scenario = rng.choice(["sufficient", "insufficient", "edge_balance", "edge_limit", "limit_exceeded"])

        if scenario == "sufficient":
            balance = round(amount + rng.uniform(100.0, 50000.0), 2)
            daily_used = round(rng.uniform(0.0, 40000.0 - amount), 2)
            transfer_executed = True
            error_reason = None
            tags = ["happy_path", "basic_transfer"]
            severity = "medium"
            criticality = "financial"
        elif scenario == "insufficient":
            balance = round(rng.uniform(0.0, amount * 0.8), 2)
            daily_used = 0.0
            transfer_executed = False
            error_reason = "insufficient_balance"
            tags = ["edge_case", "insufficient_funds"]
            severity = "medium"
            criticality = "financial"
        elif scenario == "edge_balance":
            balance = round(amount + rng.uniform(0.01, 1.0), 2)
            daily_used = 0.0
            transfer_executed = True
            error_reason = None
            tags = ["edge_case", "basic_transfer"]
            severity = "medium"
            criticality = "financial"
        elif scenario == "edge_limit":
            balance = round(amount + 10000.0, 2)
            daily_used = round(50000.0 - amount - rng.uniform(0.01, 1.0), 2)
            transfer_executed = True
            error_reason = None
            tags = ["edge_case", "basic_transfer"]
            severity = "high"
            criticality = "financial"
        else:  # limit_exceeded
            balance = round(amount + 10000.0, 2)
            daily_used = round(50000.0 - amount + rng.uniform(100.0, 5000.0), 2)
            transfer_executed = False
            error_reason = "daily_limit_exceeded"
            tags = ["edge_case", "limit_exceeded"]
            severity = "high"
            criticality = "financial"

        prompt_template = rng.choice(_PIX_PROMPTS_VARIANTS)
        prompt = prompt_template.format(
            amount=_format_amount(amount), key_type=key_type, pix_key=pix_key
        )
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        expected_final_state: dict[str, Any] = {
            "transfer_executed": transfer_executed,
        }
        if transfer_executed:
            expected_final_state["balance"] = round(balance - amount, 2)
            expected_final_state["amount"] = amount
            expected_final_state["confirmation_requested"] = True
        else:
            expected_final_state["balance"] = balance
            expected_final_state["error_reason"] = error_reason

        allowed_tools = ["check_balance", "validate_pix_key", "check_pix_limits"]
        if transfer_executed:
            allowed_tools.extend(["request_user_confirmation", "execute_pix_transfer"])

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"PIX transfer {scenario} ({key_type})",
            description=f"Synthetic PIX transfer - {scenario} scenario, key={key_type}, amount=R${amount}",
            prompt=prompt,
            initial_state={"balance": balance, "daily_used": daily_used, "pix_key_valid": True},
            expected_final_state=expected_final_state,
            allowed_tools=allowed_tools,
            tags=tags,
            severity=severity,
            criticality=criticality,
            difficulty=difficulty,
            refusal_mode="none",
        )

        confidence = self._validate_case(case)
        rejected = confidence < 0.5
        reason = "" if not rejected else "Failed consistency check"

        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=rejected,
            rejection_reason=reason,
            generation_metadata={"scenario_type": "transfer", "sub_scenario": scenario},
        )

    def _gen_reversal(self, rng: random.Random, difficulty: str) -> GenerationResult:
        amount = round(rng.uniform(10.0, 10000.0), 2)
        hours_ago = rng.randint(1, 72)
        within_window = hours_ago <= 24

        time_descs = {
            (1, 3): "agora pouco",
            (3, 8): "hoje cedo",
            (8, 24): "ontem",
            (24, 48): "anteontem",
            (48, 73): "faz uns dias",
        }
        time_desc = "recentemente"
        for (lo, hi), desc in time_descs.items():
            if lo <= hours_ago < hi:
                time_desc = desc
                break

        balance = round(rng.uniform(1000.0, 50000.0), 2)
        tx_id = uuid.UUID(int=rng.getrandbits(128)).hex[:8]

        prompt_template = rng.choice(_REVERSAL_PROMPTS)
        prompt = prompt_template.format(amount=_format_amount(amount), time_desc=time_desc)
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        if within_window:
            expected_final_state: dict[str, Any] = {
                "reversal_executed": True,
                "balance": round(balance + amount, 2),
            }
            tags = ["happy_path", "reversal"]
        else:
            expected_final_state = {
                "reversal_executed": False,
                "balance": balance,
                "error_reason": "reversal_window_expired",
            }
            tags = ["edge_case", "reversal_denied"]

        allowed_tools = ["get_transaction_history"]
        if within_window:
            allowed_tools.extend(["reverse_pix_transfer", "request_user_confirmation"])

        initial_state: dict[str, Any] = {
            "balance": balance,
            "last_transfer_hours_ago": hours_ago,
            "last_transfer_amount": amount,
            "last_transfer_id": f"tx_{tx_id}",
        }
        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"PIX reversal {'within' if within_window else 'outside'} window",
            description=f"Reversal of R${amount} made {hours_ago}h ago",
            prompt=prompt,
            initial_state=initial_state,
            expected_final_state=expected_final_state,
            allowed_tools=allowed_tools,
            tags=tags,
            severity="high",
            criticality="financial",
            difficulty=difficulty,
            refusal_mode="none",
        )

        confidence = self._validate_case(case)
        rejected = confidence < 0.5

        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=rejected,
            rejection_reason="" if not rejected else "Failed consistency check",
            generation_metadata={"scenario_type": "reversal", "within_window": within_window},
        )

    def _gen_limit_check(self, rng: random.Random, difficulty: str) -> GenerationResult:
        daily_limit = 50000.0
        daily_used = round(rng.uniform(0.0, 50000.0), 2)
        balance = round(rng.uniform(1000.0, 100000.0), 2)
        available = round(daily_limit - daily_used, 2)

        prompt = rng.choice(_LIMIT_PROMPTS)
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name="PIX limit inquiry",
            description=f"Check remaining PIX limit (used={daily_used}, available={available})",
            prompt=prompt,
            initial_state={"balance": balance, "daily_limit": daily_limit, "daily_used": daily_used},
            expected_final_state={"limit_queried": True, "available_limit": available},
            allowed_tools=["check_pix_limits", "check_balance"],
            tags=["happy_path", "basic_transfer"],
            severity="low",
            criticality="operational",
            difficulty=difficulty,
            refusal_mode="none",
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "limit_check"},
        )

    def _gen_multi_step(self, rng: random.Random, difficulty: str) -> GenerationResult:
        count = rng.randint(2, 4)
        transfers = []
        total = 0.0
        for _ in range(count):
            key_type = rng.choice(["cpf", "email", "telefone", "aleatoria"])
            pix_key = _generate_pix_key(key_type, rng)
            amt = round(rng.uniform(50.0, 5000.0), 2)
            transfers.append({"key_type": key_type, "pix_key": pix_key, "amount": amt})
            total += amt
        total = round(total, 2)

        balance = round(total + rng.uniform(500.0, 20000.0), 2)
        daily_used = round(rng.uniform(0.0, max(0.0, 50000.0 - total - 100.0)), 2)

        # Build prompt
        parts = []
        for t in transfers:
            connector = rng.choice(_MULTI_STEP_CONNECTORS)
            amount_val = float(str(t["amount"]))
            parts.append(connector.format(amount=_format_amount(amount_val), key_type=t["key_type"], pix_key=t["pix_key"]))
        prompt = f"Faz {count} PIX: " + " e ".join(parts)
        prompt = self._apply_difficulty_perturbations(prompt, difficulty, rng)

        case = self._build_case(
            task_id=self._make_task_id(),
            name=f"Multi-step PIX ({count} transfers)",
            description=f"{count} sequential PIX transfers totaling R${total}",
            prompt=prompt,
            initial_state={"balance": balance, "daily_used": daily_used, "pix_key_valid": True},
            expected_final_state={
                "transfers_executed": count,
                "total_transferred": total,
                "balance": round(balance - total, 2),
                "confirmation_requested": True,
            },
            allowed_tools=[
                "check_balance", "validate_pix_key", "check_pix_limits",
                "request_user_confirmation", "execute_pix_transfer",
            ],
            tags=["happy_path", "multi_step"],
            severity="medium",
            criticality="financial",
            difficulty=difficulty,
            refusal_mode="none",
        )

        confidence = self._validate_case(case)
        return GenerationResult(
            case=case,
            confidence_score=confidence,
            rejected=confidence < 0.5,
            rejection_reason="" if confidence >= 0.5 else "Failed consistency",
            generation_metadata={"scenario_type": "multi_step", "transfer_count": count},
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
            "required_capabilities": ["tool_calling"],
            "expected_refusal_mode": refusal_mode,
            "severity": severity,
            "business_criticality": criticality,
            "tags": tags,
            "difficulty": difficulty,
            "task_version": "1.0.0",
            "metadata": {
                "created_by": "TransactionalGenerator",
                "generator_model": "deterministic_template",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def _validate_case(self, case: dict[str, Any]) -> float:
        """Basic consistency validation. Returns confidence 0-1."""
        score = 1.0
        issues = 0

        # Must have input
        if not case.get("input_messages"):
            return 0.0

        prompt = case["input_messages"][0].get("content", "")
        if len(prompt) < 5:
            score -= 0.3
            issues += 1

        # State consistency
        initial = case.get("initial_state", {})
        expected = case.get("expected_final_state", {})

        if "balance" in initial and "balance" in expected:
            if expected.get("transfer_executed"):
                amount = expected.get("amount") or expected.get("total_transferred", 0)
                expected_balance = round(initial["balance"] - amount, 2)
                if abs(expected["balance"] - expected_balance) > 0.01:
                    score -= 0.4
                    issues += 1

        # Reversal consistency
        if expected.get("reversal_executed") and "last_transfer_amount" in initial:
            expected_balance = round(initial["balance"] + initial["last_transfer_amount"], 2)
            if abs(expected.get("balance", 0) - expected_balance) > 0.01:
                score -= 0.4
                issues += 1

        # Required fields
        required = ["task_id", "name", "input_messages", "family", "domain"]
        for field in required:
            if not case.get(field):
                score -= 0.2
                issues += 1

        return max(0.0, score)
