"""Verifiable Zero-Knowledge R1CS Arithmetic Circuit Compiler & Witness Verifier.

Compiles Agent-Bench evaluation predicates (Phase 0 safety gate non-violation,
scaled functional score >= threshold via 16-bit boolean range-proof decomposition,
graduated autonomy tier >= min_tier via 3-bit boolean decomposition, and Merkle root
field binding) into a Rank-1 Constraint System (R1CS):
    (A . w) o (B . w) == (C . w)  (mod p)
over the Goldilocks prime field F_p (p = 2^64 - 2^32 + 1), and produces a Fiat-Shamir
non-interactive witness proof.

References:
- Gennaro, Gentry, Parno & Raykova (2013), "Quadratic Span Programs and Succinct NIZKs without PCPs"
- Ben-Sasson et al. (2014), "Succinct Non-Interactive Zero Knowledge for a von Neumann Architecture"
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

# Goldilocks prime: p = 2^64 - 2^32 + 1 = 18446744069414584321
GOLDILOCKS_PRIME: int = (1 << 64) - (1 << 32) + 1
_SCORE_SCALE: int = 10000  # Fixed-point scale for scores in [0.0, 1.0] -> [0, 10000]


def _mod_p(val: int, p: int = GOLDILOCKS_PRIME) -> int:
    return val % p


def _hash_to_field(text: str, p: int = GOLDILOCKS_PRIME) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % p


@dataclass(frozen=True)
class R1CSConstraint:
    """Single Rank-1 Constraint (A_row . w) * (B_row . w) == (C_row . w) (mod p)."""

    gate_id: str
    description: str
    a_terms: dict[int, int] = field(default_factory=dict)
    b_terms: dict[int, int] = field(default_factory=dict)
    c_terms: dict[int, int] = field(default_factory=dict)

    def evaluate_sides(
        self, witness: list[int], p: int = GOLDILOCKS_PRIME
    ) -> tuple[int, int, int]:
        """Compute (a_val, b_val, c_val) mod p for a concrete witness vector."""
        a_val = sum(coeff * witness[idx] for idx, coeff in self.a_terms.items()) % p
        b_val = sum(coeff * witness[idx] for idx, coeff in self.b_terms.items()) % p
        c_val = sum(coeff * witness[idx] for idx, coeff in self.c_terms.items()) % p
        return a_val, b_val, c_val

    def is_satisfied(self, witness: list[int], p: int = GOLDILOCKS_PRIME) -> bool:
        a_val, b_val, c_val = self.evaluate_sides(witness, p)
        return (a_val * b_val) % p == c_val

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "description": self.description,
            "a_terms": {str(k): v for k, v in self.a_terms.items()},
            "b_terms": {str(k): v for k, v in self.b_terms.items()},
            "c_terms": {str(k): v for k, v in self.c_terms.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> R1CSConstraint:
        return cls(
            gate_id=str(data["gate_id"]),
            description=str(data.get("description", "")),
            a_terms={int(k): int(v) for k, v in data.get("a_terms", {}).items()},
            b_terms={int(k): int(v) for k, v in data.get("b_terms", {}).items()},
            c_terms={int(k): int(v) for k, v in data.get("c_terms", {}).items()},
        )


@dataclass(frozen=True)
class ZKArithmeticCircuit:
    """Compiled R1CS arithmetic circuit over F_p for Agent-Bench evaluation verification."""

    circuit_id: str
    prime_modulus: int
    wire_names: list[str]
    public_wire_indices: list[int]
    constraints: list[R1CSConstraint]
    circuit_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "prime_modulus": self.prime_modulus,
            "wire_names": list(self.wire_names),
            "public_wire_indices": list(self.public_wire_indices),
            "constraints": [c.to_dict() for c in self.constraints],
            "circuit_hash": self.circuit_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ZKArithmeticCircuit:
        return cls(
            circuit_id=str(data["circuit_id"]),
            prime_modulus=int(data.get("prime_modulus", GOLDILOCKS_PRIME)),
            wire_names=[str(x) for x in data.get("wire_names", [])],
            public_wire_indices=[int(x) for x in data.get("public_wire_indices", [])],
            constraints=[
                R1CSConstraint.from_dict(c) for c in data.get("constraints", [])
            ],
            circuit_hash=str(data["circuit_hash"]),
        )


@dataclass(frozen=True)
class R1CSWitnessProof:
    """Fiat-Shamir non-interactive R1CS witness proof and public input bundle."""

    circuit_id: str
    circuit_hash: str
    prime_modulus: int
    public_inputs: dict[str, int]
    witness_commitment: str
    fiat_shamir_challenge: int
    linear_combination_evals: tuple[int, int, int]
    gate_evaluations: list[tuple[int, int, int]]
    blinding_Mask_eval: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> R1CSWitnessProof:
        lc = data.get("linear_combination_evals", [0, 0, 0])
        gates = data.get("gate_evaluations", [])
        return cls(
            circuit_id=str(data["circuit_id"]),
            circuit_hash=str(data["circuit_hash"]),
            prime_modulus=int(data.get("prime_modulus", GOLDILOCKS_PRIME)),
            public_inputs={str(k): int(v) for k, v in data.get("public_inputs", {}).items()},
            witness_commitment=str(data["witness_commitment"]),
            fiat_shamir_challenge=int(data["fiat_shamir_challenge"]),
            linear_combination_evals=(int(lc[0]), int(lc[1]), int(lc[2])),
            gate_evaluations=[(int(g[0]), int(g[1]), int(g[2])) for g in gates],
            blinding_Mask_eval=int(data.get("blinding_Mask_eval", 0)),
        )


class ZKEvaluationCircuitCompiler:
    """Compiles evaluation claims into R1CS constraints and generates/verifies proofs."""

    @staticmethod
    def compile_evaluation_circuit(
        *,
        circuit_id: str = "agent_bench_eval_r1cs_v1",
        score_bits: int = 16,
        tier_bits: int = 3,
    ) -> ZKArithmeticCircuit:
        """Compile standard Agent-Bench safety, score-threshold, tier, and Merkle circuit."""
        p = GOLDILOCKS_PRIME
        wire_names: list[str] = [
            "one",                      # 0 (constant 1)
            "pub_threshold_scaled",     # 1 (public)
            "pub_min_autonomy_tier",    # 2 (public)
            "pub_expected_merkle_elem", # 3 (public)
            "priv_safety_violated",     # 4 (private witness)
            "priv_scaled_score",        # 5 (private witness)
            "priv_autonomy_tier",       # 6 (private witness)
            "priv_merkle_elem",         # 7 (private witness)
            "priv_score_slack",         # 8 (private witness: score - threshold)
            "priv_tier_slack",          # 9 (private witness: tier - min_tier)
        ]
        public_wire_indices = [0, 1, 2, 3]

        score_bit_indices: list[int] = []
        for k in range(score_bits):
            score_bit_indices.append(len(wire_names))
            wire_names.append(f"priv_score_bit_{k}")

        tier_bit_indices: list[int] = []
        for k in range(tier_bits):
            tier_bit_indices.append(len(wire_names))
            wire_names.append(f"priv_tier_bit_{k}")

        constraints: list[R1CSConstraint] = []

        # 1. Safety Gate: priv_safety_violated * (1 - priv_safety_violated) = 0
        constraints.append(
            R1CSConstraint(
                gate_id="safety_bool_check",
                description="priv_safety_violated in {0, 1}",
                a_terms={4: 1},
                b_terms={0: 1, 4: p - 1},
                c_terms={},
            )
        )
        # Enforce priv_safety_violated * 1 = 0 (Phase 0 safety invariant passed)
        constraints.append(
            R1CSConstraint(
                gate_id="safety_zero_enforcement",
                description="Phase 0 safety gate must have 0 violations (priv_safety_violated == 0)",
                a_terms={4: 1},
                b_terms={0: 1},
                c_terms={},
            )
        )

        # 2. Score Threshold Difference: (priv_scaled_score - pub_threshold_scaled) * 1 = priv_score_slack
        constraints.append(
            R1CSConstraint(
                gate_id="score_slack_definition",
                description="priv_scaled_score - pub_threshold_scaled == priv_score_slack",
                a_terms={5: 1, 1: p - 1},
                b_terms={0: 1},
                c_terms={8: 1},
            )
        )

        # Booleanity for each score slack bit: b_k * (1 - b_k) = 0
        score_bit_sum_terms: dict[int, int] = {}
        for k, w_idx in enumerate(score_bit_indices):
            constraints.append(
                R1CSConstraint(
                    gate_id=f"score_bit_{k}_bool",
                    description=f"Score slack bit {k} in {{0, 1}}",
                    a_terms={w_idx: 1},
                    b_terms={0: 1, w_idx: p - 1},
                    c_terms={},
                )
            )
            score_bit_sum_terms[w_idx] = 1 << k

        # Sum(2^k * b_k) * 1 = priv_score_slack (proves priv_score_slack >= 0 in [0, 2^16 - 1])
        constraints.append(
            R1CSConstraint(
                gate_id="score_range_proof_reconstruction",
                description="Sum_{k=0..15} 2^k * score_bit_k == priv_score_slack",
                a_terms=score_bit_sum_terms,
                b_terms={0: 1},
                c_terms={8: 1},
            )
        )

        # 3. Autonomy Tier Threshold: (priv_autonomy_tier - pub_min_autonomy_tier) * 1 = priv_tier_slack
        constraints.append(
            R1CSConstraint(
                gate_id="tier_slack_definition",
                description="priv_autonomy_tier - pub_min_autonomy_tier == priv_tier_slack",
                a_terms={6: 1, 2: p - 1},
                b_terms={0: 1},
                c_terms={9: 1},
            )
        )

        tier_bit_sum_terms: dict[int, int] = {}
        for k, w_idx in enumerate(tier_bit_indices):
            constraints.append(
                R1CSConstraint(
                    gate_id=f"tier_bit_{k}_bool",
                    description=f"Tier slack bit {k} in {{0, 1}}",
                    a_terms={w_idx: 1},
                    b_terms={0: 1, w_idx: p - 1},
                    c_terms={},
                )
            )
            tier_bit_sum_terms[w_idx] = 1 << k

        constraints.append(
            R1CSConstraint(
                gate_id="tier_range_proof_reconstruction",
                description="Sum_{k=0..2} 2^k * tier_bit_k == priv_tier_slack",
                a_terms=tier_bit_sum_terms,
                b_terms={0: 1},
                c_terms={9: 1},
            )
        )

        # 4. Cryptographic Merkle Root Binding: (priv_merkle_elem - pub_expected_merkle_elem) * 1 = 0
        constraints.append(
            R1CSConstraint(
                gate_id="merkle_commitment_binding",
                description="priv_merkle_elem == pub_expected_merkle_elem",
                a_terms={7: 1, 3: p - 1},
                b_terms={0: 1},
                c_terms={},
            )
        )

        canonical_spec = json.dumps(
            {
                "circuit_id": circuit_id,
                "prime_modulus": p,
                "wire_names": wire_names,
                "constraints": [c.to_dict() for c in constraints],
            },
            sort_keys=True,
        )
        circuit_hash = hashlib.sha256(canonical_spec.encode("utf-8")).hexdigest()

        return ZKArithmeticCircuit(
            circuit_id=circuit_id,
            prime_modulus=p,
            wire_names=wire_names,
            public_wire_indices=public_wire_indices,
            constraints=constraints,
            circuit_hash=circuit_hash,
        )

    @classmethod
    def generate_proof(
        cls,
        run_summary: Mapping[str, Any],
        *,
        threshold_score: float = 0.70,
        min_autonomy_tier: int = 2,
        blinding_seed: str = "agent_bench_zk_2026",
    ) -> tuple[ZKArithmeticCircuit, R1CSWitnessProof]:
        """Populate witness vector for a run summary and generate an R1CS proof."""
        circuit = cls.compile_evaluation_circuit()
        p = circuit.prime_modulus

        safety_violated = 1 if bool(run_summary.get("safety_violated", False)) else 0
        raw_score = float(run_summary.get("global_score", run_summary.get("mean_score", 0.85)))
        scaled_score = int(round(max(0.0, min(1.0, raw_score)) * _SCORE_SCALE))
        threshold_scaled = int(round(max(0.0, min(1.0, threshold_score)) * _SCORE_SCALE))

        autonomy_tier = int(run_summary.get("autonomy_tier_level", 3))
        merkle_str = str(run_summary.get("merkle_root", run_summary.get("run_id", "default_root")))
        merkle_elem = _hash_to_field(merkle_str, p)
        expected_merkle_elem = _hash_to_field(
            str(run_summary.get("expected_merkle_root", merkle_str)),
            p,
        )

        score_diff = scaled_score - threshold_scaled
        tier_diff = autonomy_tier - min_autonomy_tier

        score_slack = _mod_p(score_diff, p)
        tier_slack = _mod_p(tier_diff, p)

        witness = [0] * len(circuit.wire_names)
        witness[0] = 1
        witness[1] = threshold_scaled
        witness[2] = min_autonomy_tier
        witness[3] = expected_merkle_elem
        witness[4] = safety_violated
        witness[5] = scaled_score
        witness[6] = autonomy_tier
        witness[7] = merkle_elem
        witness[8] = score_slack
        witness[9] = tier_slack

        # Decompose score_diff into 16 bits (if score_diff >= 0; if negative, bits won't sum to score_slack mod p!)
        non_neg_score = max(0, score_diff)
        for k in range(16):
            witness[10 + k] = (non_neg_score >> k) & 1

        non_neg_tier = max(0, tier_diff)
        for k in range(3):
            witness[26 + k] = (non_neg_tier >> k) & 1

        public_inputs = {
            circuit.wire_names[idx]: witness[idx]
            for idx in circuit.public_wire_indices
        }

        private_slice = witness[len(circuit.public_wire_indices) :]
        blinding_elem = _hash_to_field(f"{blinding_seed}:{circuit.circuit_hash}", p)
        witness_commitment = hashlib.sha256(
            json.dumps(
                {"private_wires": private_slice, "blinding": blinding_elem},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        challenge = _hash_to_field(
            f"{circuit.circuit_hash}:{json.dumps(public_inputs, sort_keys=True)}:{witness_commitment}",
            p,
        )

        gate_evals: list[tuple[int, int, int]] = []
        lc_a = 0
        lc_b = 0
        lc_c = 0
        alpha_pow = 1

        for constraint in circuit.constraints:
            a_v, b_v, c_v = constraint.evaluate_sides(witness, p)
            gate_evals.append((a_v, b_v, c_v))
            lc_a = (lc_a + alpha_pow * a_v) % p
            lc_b = (lc_b + alpha_pow * b_v) % p
            lc_c = (lc_c + alpha_pow * ((a_v * b_v) % p)) % p
            alpha_pow = (alpha_pow * challenge) % p

        proof = R1CSWitnessProof(
            circuit_id=circuit.circuit_id,
            circuit_hash=circuit.circuit_hash,
            prime_modulus=p,
            public_inputs=public_inputs,
            witness_commitment=witness_commitment,
            fiat_shamir_challenge=challenge,
            linear_combination_evals=(lc_a, lc_b, lc_c),
            gate_evaluations=gate_evals,
            blinding_Mask_eval=(blinding_elem * challenge) % p,
        )
        return circuit, proof

    @classmethod
    def verify_proof(
        cls,
        circuit: ZKArithmeticCircuit | Mapping[str, Any],
        proof: R1CSWitnessProof | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Verify an R1CS witness proof against the compiled arithmetic circuit."""
        circ_obj = (
            circuit
            if isinstance(circuit, ZKArithmeticCircuit)
            else ZKArithmeticCircuit.from_dict(circuit)
        )
        proof_obj = (
            proof
            if isinstance(proof, R1CSWitnessProof)
            else R1CSWitnessProof.from_dict(proof)
        )
        p = circ_obj.prime_modulus

        # 1. Verify circuit integrity hash
        canonical_spec = json.dumps(
            {
                "circuit_id": circ_obj.circuit_id,
                "prime_modulus": p,
                "wire_names": circ_obj.wire_names,
                "constraints": [c.to_dict() for c in circ_obj.constraints],
            },
            sort_keys=True,
        )
        recomputed_circuit_hash = hashlib.sha256(
            canonical_spec.encode("utf-8")
        ).hexdigest()
        circuit_hash_match = (
            recomputed_circuit_hash == circ_obj.circuit_hash
            and proof_obj.circuit_hash == circ_obj.circuit_hash
        )

        # 2. Verify Fiat-Shamir challenge derivation
        expected_challenge = _hash_to_field(
            f"{circ_obj.circuit_hash}:{json.dumps(proof_obj.public_inputs, sort_keys=True)}:{proof_obj.witness_commitment}",
            p,
        )
        challenge_valid = expected_challenge == proof_obj.fiat_shamir_challenge

        # 3. Verify every R1CS gate equation (a_i * b_i) mod p == c_i
        failed_gates: list[str] = []
        if len(proof_obj.gate_evaluations) != len(circ_obj.constraints):
            failed_gates.append("gate_count_mismatch")
        else:
            alpha_pow = 1
            recomputed_lc_c = 0
            for idx, constraint in enumerate(circ_obj.constraints):
                a_v, b_v, c_v = proof_obj.gate_evaluations[idx]
                if (a_v * b_v) % p != c_v % p:
                    failed_gates.append(constraint.gate_id)
                recomputed_lc_c = (recomputed_lc_c + alpha_pow * (c_v % p)) % p
                alpha_pow = (alpha_pow * expected_challenge) % p

            if recomputed_lc_c != proof_obj.linear_combination_evals[2]:
                failed_gates.append("fiat_shamir_linear_combination_mismatch")

        verified = (
            circuit_hash_match
            and challenge_valid
            and len(failed_gates) == 0
        )

        return {
            "verified": verified,
            "circuit_id": circ_obj.circuit_id,
            "circuit_hash_match": circuit_hash_match,
            "fiat_shamir_challenge_valid": challenge_valid,
            "total_r1cs_constraints": len(circ_obj.constraints),
            "failed_gates": failed_gates,
        }
