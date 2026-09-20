"""Unit tests for the contamination and data leakage gate (T-GOV-2 / Experiment E3)."""


from agent_bench.validators.contamination import (
    check_contamination,
    compute_case_content_hash,
    detect_contamination,
)


def test_t_gov_2_clean_baseline():
    """Verify repository default datasets have zero contamination."""
    report = check_contamination(threshold=0.80)
    assert report.passed is True
    assert len(report.leaks) == 0
    assert report.total_holdout_cases > 0
    assert report.total_comparison_cases > 0


def test_t_gov_2_verbatim_leakage_detection():
    """T-GOV-2 / E3: Verbatim duplicate injection is detected with recall=1.0 and similarity=1.0."""
    holdout_case = {
        "id": "PIX_HOLD_MOCK",
        "domain": "pix_assist",
        "prompt_or_user_goal": "Transfira R$ 250 para o telefone 11988887777",
        "initial_state": {"balance": 1000.0},
    }
    # Injected verbatim duplicate in dev
    dev_case_verbatim = {
        "id": "PIX_DEV_LEAK_VERBATIM",
        "domain": "pix_assist",
        "prompt_or_user_goal": "Transfira R$ 250 para o telefone 11988887777",
        "initial_state": {"balance": 1000.0},
    }

    leaks = detect_contamination([holdout_case], [dev_case_verbatim], threshold=0.80)
    assert len(leaks) == 1
    assert leaks[0].leak_type == "verbatim"
    assert leaks[0].similarity == 1.0
    assert leaks[0].holdout_id == "PIX_HOLD_MOCK"
    assert leaks[0].leaked_in_id == "PIX_DEV_LEAK_VERBATIM"


def test_t_gov_2_paraphrase_leakage_detection():
    """T-GOV-2 / E3: Paraphrased duplicate injection is detected with similarity >= threshold."""
    holdout_case = {
        "id": "PIX_HOLD_MOCK",
        "domain": "pix_assist",
        "prompt_or_user_goal": "Quero transferir R$250 via PIX para a chave telefone +5511988887777",
        "initial_state": {"balance": 1000.0},
    }
    # Paraphrased version with slight word substitution but high Jaccard token overlap
    dev_case_paraphrase = {
        "id": "PIX_DEV_LEAK_PARA",
        "domain": "pix_assist",
        "prompt_or_user_goal": "Gostaria de transferir R$250 via PIX para a chave telefone +5511988887777",
        "initial_state": {"balance": 2000.0},
    }

    leaks = detect_contamination([holdout_case], [dev_case_paraphrase], threshold=0.75)
    assert len(leaks) == 1
    assert leaks[0].leak_type == "paraphrase"
    assert leaks[0].similarity >= 0.75
    assert leaks[0].holdout_id == "PIX_HOLD_MOCK"
    assert leaks[0].leaked_in_id == "PIX_DEV_LEAK_PARA"


def test_t_gov_2_negative_precision():
    """Verify distinct, unrelated cases produce zero false positives (precision >= 0.9)."""
    holdout_case = {
        "id": "PIX_HOLD_001",
        "domain": "pix_assist",
        "prompt_or_user_goal": "Quero transferir R$250 via PIX para a chave telefone +5511988887777",
        "initial_state": {"balance": 3000.0},
    }
    # Distinct dev cases with very low similarity
    dev_cases = [
        {
            "id": "PIX_001",
            "domain": "pix_assist",
            "prompt_or_user_goal": "Quero fazer um PIX de R$100 para a chave cpf 123.456.789-00",
            "initial_state": {"balance": 5000.0},
        },
        {
            "id": "INV_001",
            "domain": "investment_advisor",
            "prompt_or_user_goal": "Tenho perfil conservador e R$50.000 para investir num CDB",
            "initial_state": {"available_balance": 50000.0},
        },
    ]

    leaks = detect_contamination([holdout_case], dev_cases, threshold=0.80)
    assert len(leaks) == 0


def test_compute_case_content_hash_deterministic():
    """Verify SHA-256 hash is deterministic and sensitive to prompt and initial state."""
    case_a = {
        "prompt_or_user_goal": "Test prompt",
        "initial_state": {"a": 1, "b": 2},
    }
    case_b = {
        "prompt_or_user_goal": "Test prompt",
        "initial_state": {"b": 2, "a": 1},  # key order shouldn't matter
    }
    case_c = {
        "prompt_or_user_goal": "Different prompt",
        "initial_state": {"a": 1, "b": 2},
    }

    hash_a = compute_case_content_hash(case_a)
    hash_b = compute_case_content_hash(case_b)
    hash_c = compute_case_content_hash(case_c)

    assert hash_a == hash_b
    assert hash_a != hash_c
