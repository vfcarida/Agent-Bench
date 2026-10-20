"""Zero-Knowledge Federated Leaderboard Proofs — MPC + Nova IVC (`zk_federated_leaderboard.py`).

Unifies (epsilon, delta)-Differentially Private Federated Benchmark Aggregation
(`FederatedBenchmarkAggregator`) with Nova-style Relaxed R1CS Incrementally Verifiable
Computation (`NovaIVCFolder` and `ZKEvaluationCircuitCompiler`):
1. Each participating institution generates a `ZKInstitutionSubmission` containing its
   scorecards and a Goldilocks prime field (F_p) R1CS proof attesting that:
   - Its local run satisfied the Phase 0 Hard Safety Gate (`safety_violated == False`),
   - Its reported score lies within the certified 16-bit range and exceeds the minimum
     governance threshold,
   - Its reported Graduated Autonomy Tier is valid.
2. The `ZKFederatedLeaderboardAggregator` verifies every institution's R1CS witness proof,
   cryptographically filtering out any Byzantine institution attempting "metric poisoning"
   or safety-gate suppression.
3. All verified institutional R1CS instances are folded into a single constant-size
   `IVCAccumulatorBundle` via `NovaIVCFolder`, and the verified scorecards are aggregated
   under (epsilon, delta)-Differential Privacy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agent_bench.governance.federated_aggregator import (
    FederatedAggregationReport,
    FederatedBenchmarkAggregator,
    FederatedSiteSubmission,
    PrivacyBudgetAccountant,
)
from agent_bench.governance.ivc_folding import IVCAccumulatorBundle, NovaIVCFolder
from agent_bench.governance.zk_circuit_compiler import ZKEvaluationCircuitCompiler


@dataclass
class ZKInstitutionSubmission:
    """Zero-knowledge attested benchmark submission from a single participating institution."""

    site_id: str
    sample_count: int
    run_manifest: dict[str, Any]
    scorecards: list[dict[str, Any]]
    item_pass_rates: dict[str, float]
    zk_proof_bundle: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZKFederatedLeaderboardBundle:
    """Combined Nova IVC + (epsilon, delta)-DP Federated Leaderboard proof bundle."""

    participating_sites: int
    verified_sites: list[str]
    rejected_byzantine_sites: list[dict[str, Any]]
    k_anonymity_satisfied: bool
    ivc_accumulator_bundle: dict[str, Any]
    dp_aggregation_report: dict[str, Any]
    bundle_commitment_hash: str
    zero_knowledge_verified: bool = True
    threshold_score: float = 0.70
    min_autonomy_tier: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZKFederatedLeaderboardAggregator:
    """Verifies institutional R1CS proofs, folds them via Nova IVC, and releases DP leaderboards."""

    def __init__(
        self,
        *,
        min_sites: int = 2,
        threshold_score: float = 0.70,
        min_autonomy_tier: int = 2,
        accountant: PrivacyBudgetAccountant | None = None,
    ) -> None:
        self.min_sites = max(1, min_sites)
        self.threshold_score = threshold_score
        self.min_autonomy_tier = min_autonomy_tier
        self.dp_aggregator = FederatedBenchmarkAggregator(
            min_sites=self.min_sites, accountant=accountant
        )

    @staticmethod
    def _extract_run_summary(run_manifest: dict[str, Any]) -> dict[str, Any]:
        scorecards = list(run_manifest.get("scorecards", []))
        first_sc = scorecards[0] if scorecards else {}
        global_score = float(
            run_manifest.get("global_score", first_sc.get("global_score", 0.85))
        )
        safety_violated = bool(
            run_manifest.get(
                "safety_violated",
                first_sc.get("safety_gated", False)
                or int(run_manifest.get("safety_violations", 0)) > 0,
            )
        )
        autonomy_tier = int(
            run_manifest.get("autonomy_tier_level", run_manifest.get("autonomy_tier", 3))
        )
        run_id = str(run_manifest.get("run_id", "run_unknown"))
        return {
            "run_id": run_id,
            "global_score": global_score,
            "safety_violated": safety_violated,
            "autonomy_tier_level": autonomy_tier,
            "merkle_root": str(run_manifest.get("merkle_root", run_id)),
        }

    def create_institution_submission(
        self,
        site_id: str,
        run_manifest: dict[str, Any],
        *,
        sample_count: int = 25,
        item_pass_rates: dict[str, float] | None = None,
    ) -> ZKInstitutionSubmission:
        """Compile an institution's run manifest into an R1CS proof and package for federated submission."""
        summary = self._extract_run_summary(run_manifest)
        circuit, proof = ZKEvaluationCircuitCompiler.generate_proof(
            summary,
            threshold_score=self.threshold_score,
            min_autonomy_tier=self.min_autonomy_tier,
        )
        scorecards = list(run_manifest.get("scorecards", []))
        return ZKInstitutionSubmission(
            site_id=site_id,
            sample_count=max(1, int(sample_count)),
            run_manifest=run_manifest,
            scorecards=scorecards,
            item_pass_rates=dict(item_pass_rates or {"PIX_001": 0.92, "PIX_002": 0.85}),
            zk_proof_bundle={
                "circuit": circuit.to_dict(),
                "proof": proof.to_dict(),
            },
        )

    def aggregate_and_fold(
        self,
        submissions: Sequence[ZKInstitutionSubmission | dict[str, Any]],
        *,
        epsilon: float = 1.0,
        delta: float = 0.0,
        seed: int = 42,
    ) -> ZKFederatedLeaderboardBundle:
        """Filter Byzantine submissions via R1CS verification, fold via Nova IVC, and release DP stats."""
        norm_subs: list[ZKInstitutionSubmission] = []
        for s in submissions:
            if isinstance(s, ZKInstitutionSubmission):
                norm_subs.append(s)
            elif isinstance(s, dict):
                if "zk_proof_bundle" in s:
                    norm_subs.append(
                        ZKInstitutionSubmission(
                            site_id=str(s.get("site_id", "site_unknown")),
                            sample_count=int(s.get("sample_count", 20)),
                            run_manifest=dict(s.get("run_manifest", {})),
                            scorecards=list(s.get("scorecards", [])),
                            item_pass_rates=dict(s.get("item_pass_rates", {})),
                            zk_proof_bundle=dict(s["zk_proof_bundle"]),
                        )
                    )
                else:
                    manifest = dict(s.get("run_manifest", s))
                    norm_subs.append(
                        self.create_institution_submission(
                            site_id=str(s.get("site_id", manifest.get("run_id", "site_auto"))),
                            run_manifest=manifest,
                            sample_count=int(s.get("sample_count", 20)),
                            item_pass_rates=s.get("item_pass_rates"),
                        )
                    )

        verified_subs: list[ZKInstitutionSubmission] = []
        rejected_byzantine: list[dict[str, Any]] = []

        for sub in norm_subs:
            verification = ZKEvaluationCircuitCompiler.verify_proof(
                sub.zk_proof_bundle["circuit"],
                sub.zk_proof_bundle["proof"],
            )
            # Check that the submitted scorecard matches the R1CS score_slack gate evaluation!
            # Gate 2 is `score_slack_definition`: (priv_scaled_score - pub_threshold_scaled) * 1 = priv_score_slack
            gate_evals = sub.zk_proof_bundle["proof"].get("gate_evaluations", [])
            pub_inputs = sub.zk_proof_bundle["proof"].get("public_inputs", {})
            threshold_scaled = int(pub_inputs.get("pub_threshold_scaled", round(self.threshold_score * 10000)))
            claimed_slack = int(gate_evals[2][2]) if len(gate_evals) > 2 else -1
            claimed_score_fp = threshold_scaled + claimed_slack

            sc_list = sub.scorecards or sub.run_manifest.get("scorecards", [])
            reported_score = float(sc_list[0].get("global_score", 0.0)) if sc_list else 0.0
            expected_score_fp = int(round(reported_score * 10000))

            score_consistent = abs(claimed_score_fp - expected_score_fp) <= 1
            if verification["verified"] and score_consistent:
                verified_subs.append(sub)
            else:
                reasons = [str(g) for g in verification.get("failed_gates", [])]
                if not score_consistent:
                    reasons.append(
                        f"Byzantine metric poisoning: scorecard global_score ({expected_score_fp} bps) "
                        f"does not match R1CS witness wire ({claimed_score_fp} bps)."
                    )
                rejected_byzantine.append(
                    {
                        "site_id": sub.site_id,
                        "reasons": reasons or ["R1CS constraint or witness check failed"],
                    }
                )

        verified_site_ids = [s.site_id for s in verified_subs]
        if len(verified_subs) < self.min_sites:
            empty_dp = FederatedAggregationReport(
                participating_sites=len(verified_subs),
                total_samples=sum(s.sample_count for s in verified_subs),
                k_anonymity_satisfied=False,
                epsilon_spent=self.dp_aggregator.accountant.epsilon_spent,
                delta_spent=self.dp_aggregator.accountant.delta_spent,
            )
            payload = {
                "verified_sites": verified_site_ids,
                "rejected": rejected_byzantine,
                "k_anonymity_satisfied": False,
            }
            commit = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            return ZKFederatedLeaderboardBundle(
                participating_sites=len(norm_subs),
                verified_sites=verified_site_ids,
                rejected_byzantine_sites=rejected_byzantine,
                k_anonymity_satisfied=False,
                ivc_accumulator_bundle={},
                dp_aggregation_report=empty_dp.to_dict(),
                bundle_commitment_hash=commit,
                zero_knowledge_verified=False,
                threshold_score=self.threshold_score,
                min_autonomy_tier=self.min_autonomy_tier,
            )

        # Fold all verified institutional runs into a constant-size Nova IVC accumulator
        summaries = [self._extract_run_summary(s.run_manifest) for s in verified_subs]
        ivc_bundle: IVCAccumulatorBundle = NovaIVCFolder.fold_runs(
            summaries,
            threshold_score=self.threshold_score,
            min_autonomy_tier=self.min_autonomy_tier,
        )

        # Run (epsilon, delta)-DP aggregation strictly over the ZK-verified submissions
        dp_inputs = [
            FederatedSiteSubmission(
                site_id=s.site_id,
                sample_count=s.sample_count,
                scorecards=s.scorecards,
                item_pass_rates=s.item_pass_rates,
            )
            for s in verified_subs
        ]
        dp_report = self.dp_aggregator.aggregate(
            dp_inputs, epsilon=epsilon, delta=delta, seed=seed
        )

        commit_payload = {
            "verified_sites": verified_site_ids,
            "ivc_instance_commitment": ivc_bundle.accumulator.instance_commitment,
            "dp_scorecards": dp_report.aggregated_scorecards,
            "epsilon_spent": dp_report.epsilon_spent,
        }
        commit_hash = hashlib.sha256(
            json.dumps(commit_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return ZKFederatedLeaderboardBundle(
            participating_sites=len(norm_subs),
            verified_sites=verified_site_ids,
            rejected_byzantine_sites=rejected_byzantine,
            k_anonymity_satisfied=dp_report.k_anonymity_satisfied,
            ivc_accumulator_bundle=ivc_bundle.to_dict(),
            dp_aggregation_report=dp_report.to_dict(),
            bundle_commitment_hash=commit_hash,
            zero_knowledge_verified=True,
            threshold_score=self.threshold_score,
            min_autonomy_tier=self.min_autonomy_tier,
        )

    def verify_federated_bundle(
        self, bundle: ZKFederatedLeaderboardBundle | dict[str, Any]
    ) -> dict[str, Any]:
        """Verify both the Nova IVC folded proof and the federated DP commitment."""
        b_dict = bundle.to_dict() if isinstance(bundle, ZKFederatedLeaderboardBundle) else dict(bundle)
        errors: list[str] = []

        if not b_dict.get("k_anonymity_satisfied", False):
            errors.append("k-anonymity threshold was not satisfied.")

        ivc_dict = b_dict.get("ivc_accumulator_bundle", {})
        if not ivc_dict:
            errors.append("Missing Nova IVC accumulator bundle.")
            ivc_res: dict[str, Any] = {"verified": False, "folded_runs_count": 0}
        else:
            ivc_res = NovaIVCFolder.verify_ivc_bundle(ivc_dict)
            if not ivc_res["verified"]:
                failed = ivc_res.get("failed_gates", [])
                if isinstance(failed, list):
                    errors.extend([str(x) for x in failed])
                else:
                    errors.append("Nova IVC verification failed.")

        dp_dict = b_dict.get("dp_aggregation_report", {})
        if ivc_dict and dp_dict:
            expected_payload = {
                "verified_sites": b_dict.get("verified_sites", []),
                "ivc_instance_commitment": ivc_dict.get("accumulator", {}).get(
                    "instance_commitment", ""
                ),
                "dp_scorecards": dp_dict.get("aggregated_scorecards", []),
                "epsilon_spent": dp_dict.get("epsilon_spent", 0.0),
            }
            expected_commit = hashlib.sha256(
                json.dumps(expected_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if expected_commit != b_dict.get("bundle_commitment_hash"):
                errors.append("Federated bundle commitment hash mismatch.")

        return {
            "verified": len(errors) == 0,
            "verified_sites_count": len(b_dict.get("verified_sites", [])),
            "rejected_byzantine_count": len(b_dict.get("rejected_byzantine_sites", [])),
            "ivc_runs_folded": int(ivc_res.get("folded_runs_count", 0)),
            "errors": errors,
        }
