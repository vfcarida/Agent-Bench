"""Unit tests for Proof-Carrying Agent Trajectories (PCAT), Bayesian Computerized Adaptive Testing (CAT), and Cross-Jurisdictional Regulatory Harmonizer."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import agent_bench
import agent_bench.graders
import agent_bench.validators
from agent_bench.cli.main import cli
from agent_bench.governance.proof_carrying_trajectory import (
    DEFAULT_TOOL_CONTRACTS,
    PCATVerifier,
    ProofCarryingStepEnvelope,
    VerifiableToolContract,
)
from agent_bench.governance.regulatory_harmonizer import (
    DEFAULT_JURISDICTIONAL_RULES,
    JurisdictionalRule,
    RegulatoryComplianceHarmonizer,
)
from agent_bench.metrics.adaptive_cat import (
    DEFAULT_CAT_ITEM_BANK,
    BayesianCATEngine,
)
from agent_bench.metrics.irt import IRTItemParameters


def test_verifiable_tool_contract_and_pcat_verifier_soundness() -> None:
    contract = DEFAULT_TOOL_CONTRACTS["execute_pix_transfer"]
    assert contract.to_dict()["tool_name"] == "execute_pix_transfer"

    verifier = PCATVerifier()
    steps = [
        {
            "step_index": 1,
            "tool_name": "execute_pix_transfer",
            "arguments": {
                "amount": 1200.0,
                "recipient_key": "valid@pix.br",
                "mfa_token_verified": True,
            },
            "pre_state": {
                "kyc_verified": True,
                "account_frozen": False,
                "step_up_mfa_verified": True,
            },
            "post_state": {
                "kyc_verified": True,
                "account_frozen": False,
                "last_tx": 1200.0,
            },
        },
        {
            "step_index": 2,
            "tool_name": "rebalance_portfolio",
            "arguments": {"leverage_ratio": 1.4, "allocation_pct": 60.0},
            "pre_state": {"suitability_checked": True},
            "post_state": {"suitability_checked": True, "rebalanced": True},
        },
    ]

    report = verifier.build_and_verify_trajectory(steps, trajectory_id="pcat_test_01")
    assert report.pcat_verified is True
    assert report.hash_chain_intact is True
    assert report.verified_steps == 2
    assert report.total_steps == 2
    assert report.contract_violations == []
    assert report.to_dict()["pcat_verified"] is True

    # Verify from serialized dictionary envelopes
    dict_envelopes = [e.to_dict() for e in report.envelopes]
    report_from_dicts = verifier.verify_envelopes(dict_envelopes, trajectory_id="pcat_dict")
    assert report_from_dicts.pcat_verified is True


def test_pcat_contract_violations_and_tamper_detection() -> None:
    custom_contract = VerifiableToolContract(
        tool_name="custom_transfer",
        required_preconditions={"kyc_ok": True},
        max_numeric_parameters={"amount": 1000.0, "limit_hint": 50.0},
        forbidden_argument_substrings=["drop table", "rm -rf"],
        requires_step_up_auth=True,
    )
    verifier = PCATVerifier(contracts={"custom_transfer": custom_contract})

    # Violate precondition, numeric bound, non-float numeric bound, step-up auth, and pattern
    bad_steps = [
        {
            "step_index": 1,
            "tool_name": "custom_transfer",
            "arguments": {
                "amount": "not_a_number",
                "note": "DROP TABLE accounts",
            },
            "pre_state": {"kyc_ok": False, "step_up_mfa_verified": False},
            "post_state": {"kyc_ok": False},
        }
    ]
    bad_report = verifier.build_and_verify_trajectory(bad_steps, trajectory_id="pcat_bad")
    assert bad_report.pcat_verified is False
    assert bad_report.hash_chain_intact is True
    assert len(bad_report.contract_violations) == 1
    issues = bad_report.contract_violations[0]["issues"]
    assert "precondition_failed:kyc_ok" in issues
    assert "precondition_failed:step_up_mfa_verified" in issues
    assert "numeric_bound_exceeded:amount" in issues
    assert "forbidden_argument_pattern_detected" in issues

    # Tamper with parent_envelope_hash and step_proof_hash
    env = bad_report.envelopes[0]
    tampered_env = ProofCarryingStepEnvelope(
        step_index=env.step_index,
        tool_name=env.tool_name,
        arguments=env.arguments,
        pre_state_hash=env.pre_state_hash,
        post_state_hash=env.post_state_hash,
        precondition_witness={"kyc_ok": True},
        bounds_witness={"amount": True},
        pattern_witness=True,
        parent_envelope_hash="f" * 64,
        step_proof_hash="0" * 64,
    )
    tampered_report = verifier.verify_envelopes([tampered_env], trajectory_id="pcat_tampered")
    assert tampered_report.hash_chain_intact is False
    assert tampered_report.pcat_verified is False
    tamper_issues = tampered_report.contract_violations[0]["issues"]
    assert "parent_envelope_hash_mismatch" in tamper_issues
    assert "step_proof_hash_tampered" in tamper_issues


def test_bayesian_cat_engine_convergence_and_oracles() -> None:
    engine = BayesianCATEngine(
        DEFAULT_CAT_ITEM_BANK,
        prior_mean=0.0,
        prior_variance=1.0,
        target_se=0.40,
        max_items=10,
        min_items=3,
    )

    # 1. Default simulation with true_latent_theta
    report = engine.run_adaptive_session(
        system_id="agent_high_theta",
        true_latent_theta=1.10,
    )
    assert report.total_pool_size == len(DEFAULT_CAT_ITEM_BANK)
    assert 3 <= report.items_administered <= 10
    assert report.final_se <= 0.45
    assert report.sample_efficiency_reduction_pct > 0.0
    assert report.to_dict()["system_id"] == "agent_high_theta"

    # 2. Callable oracle (all pass)
    oracle_report = engine.run_adaptive_session(
        lambda item: True,
        system_id="perfect_oracle_agent",
    )
    assert oracle_report.final_theta > 0.5

    # 3. Mapping outcomes (partial dict + fallback)
    mapping_outcomes = {"PIX_MED_02": 1, "INV_MED_02": 0}
    map_report = engine.run_adaptive_session(
        mapping_outcomes,
        system_id="mapping_agent",
        true_latent_theta=-0.5,
    )
    assert map_report.items_administered >= 3

    # 4. Small pool exhaustion test
    tiny_bank = [
        IRTItemParameters("T1", 0.0, 1.2, 0.5, 0.36, 1.0),
        IRTItemParameters("T2", 0.5, 1.3, 0.4, 0.38, 1.1),
    ]
    tiny_engine = BayesianCATEngine(tiny_bank, target_se=0.10, max_items=5, min_items=2)
    tiny_report = tiny_engine.run_adaptive_session(true_latent_theta=0.2)
    assert tiny_report.items_administered == 2
    assert tiny_engine.select_next_item(0.0, {"T1", "T2"}) is None


def test_regulatory_compliance_harmonizer_and_conflict_resolution() -> None:
    assert DEFAULT_JURISDICTIONAL_RULES[0].to_dict()["jurisdiction"] == "BR_BACEN_LGPD"
    harmonizer = RegulatoryComplianceHarmonizer()

    # 1. Compliant record across all 3 jurisdictions
    compliant_record = {
        "tools_called": ["get_balance", "execute_pix_transfer"],
        "compliance_flags": {
            "pii_redacted_or_pseudonymized": True,
            "kyc_suitability_verified": True,
            "ai_transparency_disclosure_shown": True,
            "human_oversight_escalation_enabled": True,
            "cryptographic_merkle_audit_committed": True,
        },
        "audit_retention_days": 2555,
        "stores_raw_pii_in_audit_log": False,
    }
    clean_report = harmonizer.evaluate_compliance(compliant_record, system_id="clean_bank_agent")
    assert clean_report.global_regulatory_cleared is True
    assert clean_report.violations == []
    assert len(clean_report.conflicts_resolved) == 1
    assert clean_report.conflicts_resolved[0].satisfied_by_execution is True
    assert clean_report.to_dict()["global_regulatory_cleared"] is True

    # 2. Non-compliant record with forbidden tools, raw PII storage, short retention, missing flags
    bad_record = {
        "tools_called": ["export_unmasked_cpf_ledger", "delete_audit_trail"],
        "compliance_flags": {
            "pii_redacted_or_pseudonymized": False,
        },
        "audit_retention_days": 90,
        "stores_raw_pii_in_audit_log": True,
    }
    bad_report = harmonizer.evaluate_compliance(bad_record, system_id="violating_agent")
    assert bad_report.global_regulatory_cleared is False
    assert len(bad_report.violations) >= 5
    assert bad_report.conflicts_resolved[0].satisfied_by_execution is False

    # 3. Custom rule with empty checks
    custom_rule = JurisdictionalRule(
        rule_id="CUSTOM_EMPTY",
        jurisdiction="CUSTOM_JUR",
        title="Minimal Rule",
    )
    custom_harmonizer = RegulatoryComplianceHarmonizer(rules=[custom_rule])
    custom_rep = custom_harmonizer.evaluate_compliance(
        compliant_record,
        jurisdictions=["UNMATCHED_JUR"],
    )
    assert custom_rep.per_jurisdiction_scores["UNMATCHED_JUR"] == 1.0


def test_cli_verify_pcat_simulate_cat_and_audit_jurisdictions(tmp_path: Path) -> None:
    runner = CliRunner()

    # 1. verify-pcat
    pcat_out = tmp_path / "pcat_report.json"
    res_pcat = runner.invoke(
        cli,
        ["verify-pcat", "--trajectory-id", "cli_pcat_01", "--output", str(pcat_out)],
    )
    assert res_pcat.exit_code == 0, res_pcat.output
    assert pcat_out.exists()
    pcat_data = json.loads(pcat_out.read_text(encoding="utf-8"))
    assert pcat_data["pcat_verified"] is True

    # 2. simulate-cat
    cat_out = tmp_path / "cat_report.json"
    res_cat = runner.invoke(
        cli,
        [
            "simulate-cat",
            "--system-id",
            "cli_cat_agent",
            "--true-theta",
            "0.85",
            "--target-se",
            "0.42",
            "--max-items",
            "8",
            "--output",
            str(cat_out),
        ],
    )
    assert res_cat.exit_code == 0, res_cat.output
    assert cat_out.exists()
    cat_data = json.loads(cat_out.read_text(encoding="utf-8"))
    assert cat_data["total_pool_size"] == len(DEFAULT_CAT_ITEM_BANK)

    # 3. audit-jurisdictions
    reg_out = tmp_path / "reg_report.json"
    res_reg = runner.invoke(
        cli,
        [
            "audit-jurisdictions",
            "--system-id",
            "cli_global_bank",
            "--output",
            str(reg_out),
        ],
    )
    assert res_reg.exit_code == 0, res_reg.output
    assert reg_out.exists()
    reg_data = json.loads(reg_out.read_text(encoding="utf-8"))
    assert reg_data["global_regulatory_cleared"] is True


def test_subpackage_and_root_public_exports() -> None:
    assert "GatedEvaluator" in agent_bench.graders.__all__
    assert "CircuitGraphTracer" in agent_bench.graders.__all__
    assert "check_contamination" in agent_bench.validators.__all__
    assert "PCATVerifier" in agent_bench.__all__
    assert "BayesianCATEngine" in agent_bench.__all__
    assert "RegulatoryComplianceHarmonizer" in agent_bench.__all__
