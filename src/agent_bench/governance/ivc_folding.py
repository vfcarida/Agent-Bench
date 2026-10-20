"""Incrementally Verifiable Computation (IVC) & Nova-Style Relaxed R1CS Folding Scheme.

Folds N benchmark evaluation R1CS instances over the Goldilocks prime field F_p
into a single constant-size Relaxed R1CS accumulator (Z, u, E) satisfying:
    (A . Z) o (B . Z) == u * (C . Z) + E  (mod p)
via the exact bilinear cross-term identity:
    T  = (A Z_1) o (B Z_2) + (A Z_2) o (B Z_1) - u_1 (C Z_2) - u_2 (C Z_1)  (mod p)
    Z' = Z_1 + r * Z_2  (mod p)
    u' = u_1 + r * u_2  (mod p)
    E' = E_1 + r * T + r^2 * E_2  (mod p)

References:
- Kothapalli, Setty & Tzialla (CRYPTO 2022), "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from agent_bench.governance.zk_circuit_compiler import (
    GOLDILOCKS_PRIME,
    ZKArithmeticCircuit,
    ZKEvaluationCircuitCompiler,
)

_SCORE_SCALE: int = 10000


def _hash_to_field(text: str, p: int = GOLDILOCKS_PRIME) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % p


def _commit_vector(label: str, vec: Sequence[int], p: int = GOLDILOCKS_PRIME) -> str:
    payload = json.dumps({"label": label, "p": p, "vec": list(vec)}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RelaxedR1CSInstance:
    """Nova-style Relaxed R1CS instance (Z, u, E) over F_p."""

    step_count: int
    u_scalar: int
    witness_vector: list[int]
    error_vector: list[int]
    instance_commitment: str
    folded_run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RelaxedR1CSInstance:
        return cls(
            step_count=int(data["step_count"]),
            u_scalar=int(data["u_scalar"]),
            witness_vector=[int(x) for x in data.get("witness_vector", [])],
            error_vector=[int(x) for x in data.get("error_vector", [])],
            instance_commitment=str(data["instance_commitment"]),
            folded_run_ids=[str(x) for x in data.get("folded_run_ids", [])],
        )


@dataclass(frozen=True)
class IVCFoldingStepProof:
    """Transcript record of a single Nova folding step."""

    step_index: int
    run_id: str
    cross_term_commitment: str
    fiat_shamir_r: int
    accumulated_commitment: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IVCFoldingStepProof:
        return cls(
            step_index=int(data["step_index"]),
            run_id=str(data.get("run_id", "")),
            cross_term_commitment=str(data["cross_term_commitment"]),
            fiat_shamir_r=int(data["fiat_shamir_r"]),
            accumulated_commitment=str(data["accumulated_commitment"]),
        )


@dataclass(frozen=True)
class IVCAccumulatorBundle:
    """Constant-size folded Relaxed R1CS accumulator and IVC transcript."""

    circuit_id: str
    circuit_hash: str
    prime_modulus: int
    accumulator: RelaxedR1CSInstance
    step_proofs: list[IVCFoldingStepProof] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "circuit_hash": self.circuit_hash,
            "prime_modulus": self.prime_modulus,
            "accumulator": self.accumulator.to_dict(),
            "step_proofs": [s.to_dict() for s in self.step_proofs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IVCAccumulatorBundle:
        return cls(
            circuit_id=str(data["circuit_id"]),
            circuit_hash=str(data["circuit_hash"]),
            prime_modulus=int(data.get("prime_modulus", GOLDILOCKS_PRIME)),
            accumulator=RelaxedR1CSInstance.from_dict(data["accumulator"]),
            step_proofs=[
                IVCFoldingStepProof.from_dict(s) for s in data.get("step_proofs", [])
            ],
        )


class NovaIVCFolder:
    """Folds multiple per-run R1CS proofs into a single Relaxed R1CS accumulator."""

    @staticmethod
    def build_fresh_instance(
        run_summary: Mapping[str, Any],
        circuit: ZKArithmeticCircuit | None = None,
        *,
        threshold_score: float = 0.70,
        min_autonomy_tier: int = 2,
    ) -> tuple[ZKArithmeticCircuit, RelaxedR1CSInstance]:
        """Construct an unrelaxed R1CS instance (u=1, E=0) for a single evaluation run."""
        circ = circuit or ZKEvaluationCircuitCompiler.compile_evaluation_circuit()
        p = circ.prime_modulus

        run_id = str(run_summary.get("run_id", "run_unknown"))
        safety_violated = 1 if bool(run_summary.get("safety_violated", False)) else 0
        raw_score = float(
            run_summary.get("global_score", run_summary.get("mean_score", 0.85))
        )
        scaled_score = int(round(max(0.0, min(1.0, raw_score)) * _SCORE_SCALE))
        threshold_scaled = int(
            round(max(0.0, min(1.0, threshold_score)) * _SCORE_SCALE)
        )

        autonomy_tier = int(run_summary.get("autonomy_tier_level", 3))
        merkle_str = str(run_summary.get("merkle_root", run_id))
        merkle_elem = _hash_to_field(merkle_str, p)
        expected_merkle_elem = _hash_to_field(
            str(run_summary.get("expected_merkle_root", merkle_str)),
            p,
        )

        score_diff = scaled_score - threshold_scaled
        tier_diff = autonomy_tier - min_autonomy_tier

        witness = [0] * len(circ.wire_names)
        witness[0] = 1  # u = 1 for fresh unrelaxed R1CS instance
        witness[1] = threshold_scaled
        witness[2] = min_autonomy_tier
        witness[3] = expected_merkle_elem
        witness[4] = safety_violated
        witness[5] = scaled_score
        witness[6] = autonomy_tier
        witness[7] = merkle_elem
        witness[8] = score_diff % p
        witness[9] = tier_diff % p

        non_neg_score = max(0, score_diff)
        for k in range(16):
            witness[10 + k] = (non_neg_score >> k) & 1

        non_neg_tier = max(0, tier_diff)
        for k in range(3):
            witness[26 + k] = (non_neg_tier >> k) & 1

        error_vec = [0] * len(circ.constraints)
        commitment = _commit_vector(f"inst:{run_id}:u=1", witness + error_vec, p)

        return circ, RelaxedR1CSInstance(
            step_count=1,
            u_scalar=1,
            witness_vector=witness,
            error_vector=error_vec,
            instance_commitment=commitment,
            folded_run_ids=[run_id],
        )

    @classmethod
    def fold_instances(
        cls,
        circuit: ZKArithmeticCircuit,
        acc: RelaxedR1CSInstance,
        incoming: RelaxedR1CSInstance,
        *,
        step_index: int = 1,
    ) -> tuple[RelaxedR1CSInstance, IVCFoldingStepProof]:
        """Fold two Relaxed R1CS instances (acc, incoming) into a single accumulator."""
        p = circuit.prime_modulus
        z1 = acc.witness_vector
        z2 = incoming.witness_vector
        u1 = acc.u_scalar % p
        u2 = incoming.u_scalar % p
        e1 = acc.error_vector
        e2 = incoming.error_vector

        # Compute cross-term vector T in F_p^M:
        # T_m = (A_m . Z1)(B_m . Z2) + (A_m . Z2)(B_m . Z1) - u1*(C_m . Z2) - u2*(C_m . Z1) mod p
        cross_term: list[int] = []
        for constraint in circuit.constraints:
            az1, bz1, cz1 = constraint.evaluate_sides(z1, p)
            az2, bz2, cz2 = constraint.evaluate_sides(z2, p)
            t_m = (
                (az1 * bz2)
                + (az2 * bz1)
                - (u1 * cz2)
                - (u2 * cz1)
            ) % p
            cross_term.append(t_m)

        cross_comm = _commit_vector(f"cross_step_{step_index}", cross_term, p)
        challenge_r = _hash_to_field(
            f"{circuit.circuit_hash}:{acc.instance_commitment}:{incoming.instance_commitment}:{cross_comm}",
            p,
        )
        r_sq = (challenge_r * challenge_r) % p

        z_folded = [(z1[i] + challenge_r * z2[i]) % p for i in range(len(z1))]
        u_folded = (u1 + challenge_r * u2) % p
        e_folded = [
            (e1[m] + challenge_r * cross_term[m] + r_sq * e2[m]) % p
            for m in range(len(circuit.constraints))
        ]

        folded_runs = [*acc.folded_run_ids, *incoming.folded_run_ids]
        new_comm = _commit_vector(
            f"acc_step_{acc.step_count + incoming.step_count}",
            z_folded + e_folded,
            p,
        )

        new_acc = RelaxedR1CSInstance(
            step_count=acc.step_count + incoming.step_count,
            u_scalar=u_folded,
            witness_vector=z_folded,
            error_vector=e_folded,
            instance_commitment=new_comm,
            folded_run_ids=folded_runs,
        )
        step_proof = IVCFoldingStepProof(
            step_index=step_index,
            run_id=incoming.folded_run_ids[-1] if incoming.folded_run_ids else "",
            cross_term_commitment=cross_comm,
            fiat_shamir_r=challenge_r,
            accumulated_commitment=new_comm,
        )
        return new_acc, step_proof

    @classmethod
    def fold_runs(
        cls,
        run_summaries: Sequence[Mapping[str, Any]],
        *,
        threshold_score: float = 0.70,
        min_autonomy_tier: int = 2,
    ) -> IVCAccumulatorBundle:
        """Fold a sequence of N run summaries into a constant-size IVCAccumulatorBundle."""
        if not run_summaries:
            raise ValueError("At least one run summary is required for IVC folding.")

        circuit = ZKEvaluationCircuitCompiler.compile_evaluation_circuit()
        _, accumulator = cls.build_fresh_instance(
            run_summaries[0],
            circuit=circuit,
            threshold_score=threshold_score,
            min_autonomy_tier=min_autonomy_tier,
        )
        step_proofs: list[IVCFoldingStepProof] = []

        for idx, run_sum in enumerate(run_summaries[1:], start=1):
            _, incoming = cls.build_fresh_instance(
                run_sum,
                circuit=circuit,
                threshold_score=threshold_score,
                min_autonomy_tier=min_autonomy_tier,
            )
            accumulator, proof = cls.fold_instances(
                circuit, accumulator, incoming, step_index=idx
            )
            step_proofs.append(proof)

        return IVCAccumulatorBundle(
            circuit_id=circuit.circuit_id,
            circuit_hash=circuit.circuit_hash,
            prime_modulus=circuit.prime_modulus,
            accumulator=accumulator,
            step_proofs=step_proofs,
        )

    @classmethod
    def verify_ivc_bundle(
        cls,
        bundle: IVCAccumulatorBundle | Mapping[str, Any],
        circuit: ZKArithmeticCircuit | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify the final folded Relaxed R1CS equation (A Z) o (B Z) == u (C Z) + E (mod p)."""
        b_obj = (
            bundle
            if isinstance(bundle, IVCAccumulatorBundle)
            else IVCAccumulatorBundle.from_dict(bundle)
        )
        if circuit is None:
            circ_obj = ZKEvaluationCircuitCompiler.compile_evaluation_circuit(
                circuit_id=b_obj.circuit_id
            )
        elif isinstance(circuit, ZKArithmeticCircuit):
            circ_obj = circuit
        else:
            circ_obj = ZKArithmeticCircuit.from_dict(circuit)

        p = circ_obj.prime_modulus
        acc = b_obj.accumulator
        z = acc.witness_vector
        u = acc.u_scalar % p
        e = acc.error_vector

        circuit_hash_match = b_obj.circuit_hash == circ_obj.circuit_hash
        u_wire_consistent = len(z) > 0 and (z[0] % p) == u

        failed_gates: list[str] = []
        if len(z) != len(circ_obj.wire_names) or len(e) != len(circ_obj.constraints):
            failed_gates.append("dimension_mismatch")
        else:
            for m, constraint in enumerate(circ_obj.constraints):
                az, bz, cz = constraint.evaluate_sides(z, p)
                lhs = (az * bz) % p
                rhs = (u * cz + e[m]) % p
                if lhs != rhs:
                    failed_gates.append(constraint.gate_id)

        verified = (
            circuit_hash_match
            and u_wire_consistent
            and len(failed_gates) == 0
        )

        return {
            "verified": verified,
            "circuit_id": circ_obj.circuit_id,
            "folded_runs_count": acc.step_count,
            "folded_run_ids": acc.folded_run_ids,
            "circuit_hash_match": circuit_hash_match,
            "u_wire_consistent": u_wire_consistent,
            "failed_gates": failed_gates,
        }
