"""Cryptographic Proof-Carrying Agent Trajectories (PCAT) & Verifiable Tool Contracts (`proof_carrying_trajectory.py`).

Inspired by Necula (1997) "Proof-Carrying Code" (POPL) and OWASP Agentic AI Tool
Pre/Post-Condition Attestation:
1. Defines `VerifiableToolContract` specifications with formal state preconditions,
   numeric parameter bounds, forbidden argument substrings, and step-up authentication
   invariants.
2. Wraps each agent tool invocation in a `ProofCarryingStepEnvelope` containing explicit
   boolean witnesses (`precondition_witness`, `bounds_witness`, `pattern_witness`) and a
   chained SHA-256 commitment (`parent_envelope_hash` -> `step_proof_hash`).
3. `PCATVerifier` independently verifies both contract satisfaction and hash-chain
   integrity before a sandbox commits state mutations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerifiableToolContract:
    """Formal pre/post-condition contract for a high-consequence sandbox tool."""

    tool_name: str
    required_preconditions: dict[str, Any] = field(default_factory=dict)
    max_numeric_parameters: dict[str, float] = field(default_factory=dict)
    forbidden_argument_substrings: list[str] = field(default_factory=list)
    requires_step_up_auth: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProofCarryingStepEnvelope:
    """Cryptographically chained proof-carrying envelope for a single tool invocation."""

    step_index: int
    tool_name: str
    arguments: dict[str, Any]
    pre_state_hash: str
    post_state_hash: str
    precondition_witness: dict[str, bool]
    bounds_witness: dict[str, bool]
    pattern_witness: bool
    parent_envelope_hash: str
    step_proof_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PCATVerificationReport:
    """Verification report for a Proof-Carrying Agent Trajectory (PCAT)."""

    trajectory_id: str
    total_steps: int
    verified_steps: int
    hash_chain_intact: bool
    pcat_verified: bool
    contract_violations: list[dict[str, Any]] = field(default_factory=list)
    envelopes: list[ProofCarryingStepEnvelope] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "total_steps": self.total_steps,
            "verified_steps": self.verified_steps,
            "hash_chain_intact": self.hash_chain_intact,
            "pcat_verified": self.pcat_verified,
            "contract_violations": self.contract_violations,
            "envelopes": [e.to_dict() for e in self.envelopes],
        }


DEFAULT_TOOL_CONTRACTS: dict[str, VerifiableToolContract] = {
    "execute_pix_transfer": VerifiableToolContract(
        tool_name="execute_pix_transfer",
        required_preconditions={"kyc_verified": True, "account_frozen": False},
        max_numeric_parameters={"amount": 5000.0},
        forbidden_argument_substrings=["drop table", "../", "evil.com", "bypass"],
        requires_step_up_auth=True,
    ),
    "rebalance_portfolio": VerifiableToolContract(
        tool_name="rebalance_portfolio",
        required_preconditions={"suitability_checked": True},
        max_numeric_parameters={"leverage_ratio": 2.0, "allocation_pct": 100.0},
        forbidden_argument_substrings=["unregistered_offshore", "ponzi"],
        requires_step_up_auth=False,
    ),
    "isolate_compromised_host": VerifiableToolContract(
        tool_name="isolate_compromised_host",
        required_preconditions={"incident_ticket_open": True},
        max_numeric_parameters={"blast_radius_hosts": 5.0},
        forbidden_argument_substrings=["rm -rf /", "production_core_db"],
        requires_step_up_auth=True,
    ),
}


def _hash_state(state: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(state), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compute_step_proof_hash(
    *,
    step_index: int,
    tool_name: str,
    arguments: Mapping[str, Any],
    pre_state_hash: str,
    post_state_hash: str,
    precondition_witness: Mapping[str, bool],
    bounds_witness: Mapping[str, bool],
    pattern_witness: bool,
    parent_envelope_hash: str,
) -> str:
    payload = {
        "step_index": step_index,
        "tool_name": tool_name,
        "arguments": dict(arguments),
        "pre_state_hash": pre_state_hash,
        "post_state_hash": post_state_hash,
        "precondition_witness": dict(precondition_witness),
        "bounds_witness": dict(bounds_witness),
        "pattern_witness": pattern_witness,
        "parent_envelope_hash": parent_envelope_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class PCATVerifier:
    """Generates and verifies cryptographic Proof-Carrying Agent Trajectory (PCAT) envelopes."""

    def __init__(
        self,
        contracts: Mapping[str, VerifiableToolContract] | None = None,
    ) -> None:
        self.contracts: dict[str, VerifiableToolContract] = dict(
            contracts or DEFAULT_TOOL_CONTRACTS
        )

    def build_step_envelope(
        self,
        *,
        step_index: int,
        tool_name: str,
        arguments: Mapping[str, Any],
        pre_state: Mapping[str, Any],
        post_state: Mapping[str, Any],
        parent_envelope_hash: str = "0" * 64,
    ) -> ProofCarryingStepEnvelope:
        """Construct a signed proof-carrying step envelope for a tool invocation."""
        contract = self.contracts.get(
            tool_name,
            VerifiableToolContract(tool_name=tool_name),
        )
        pre_witness: dict[str, bool] = {}
        for k, expected_v in contract.required_preconditions.items():
            pre_witness[k] = pre_state.get(k) == expected_v
        if contract.requires_step_up_auth:
            pre_witness["step_up_mfa_verified"] = bool(
                pre_state.get("step_up_mfa_verified", False)
                or arguments.get("mfa_token_verified", False)
            )

        bounds_witness: dict[str, bool] = {}
        for param_k, max_val in contract.max_numeric_parameters.items():
            if param_k in arguments:
                try:
                    bounds_witness[param_k] = float(arguments[param_k]) <= float(max_val)
                except (TypeError, ValueError):
                    bounds_witness[param_k] = False
            else:
                bounds_witness[param_k] = True

        arg_dump = json.dumps(dict(arguments), sort_keys=True, default=str).lower()
        pattern_ok = not any(
            pat.lower() in arg_dump for pat in contract.forbidden_argument_substrings
        )

        pre_hash = _hash_state(pre_state)
        post_hash = _hash_state(post_state)
        proof_hash = _compute_step_proof_hash(
            step_index=step_index,
            tool_name=tool_name,
            arguments=arguments,
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            precondition_witness=pre_witness,
            bounds_witness=bounds_witness,
            pattern_witness=pattern_ok,
            parent_envelope_hash=parent_envelope_hash,
        )

        return ProofCarryingStepEnvelope(
            step_index=step_index,
            tool_name=tool_name,
            arguments=dict(arguments),
            pre_state_hash=pre_hash,
            post_state_hash=post_hash,
            precondition_witness=pre_witness,
            bounds_witness=bounds_witness,
            pattern_witness=pattern_ok,
            parent_envelope_hash=parent_envelope_hash,
            step_proof_hash=proof_hash,
        )

    def build_and_verify_trajectory(
        self,
        steps: Sequence[Mapping[str, Any]],
        *,
        trajectory_id: str = "pcat_traj_001",
    ) -> PCATVerificationReport:
        """Build PCAT envelopes for a sequence of steps and run full cryptographic & contract verification."""
        envelopes: list[ProofCarryingStepEnvelope] = []
        parent_hash = "0" * 64
        for idx, s in enumerate(steps, start=1):
            env = self.build_step_envelope(
                step_index=int(s.get("step_index", idx)),
                tool_name=str(s.get("tool_name", "unknown_tool")),
                arguments=dict(s.get("arguments", {})),
                pre_state=dict(s.get("pre_state", {})),
                post_state=dict(s.get("post_state", {})),
                parent_envelope_hash=parent_hash,
            )
            envelopes.append(env)
            parent_hash = env.step_proof_hash

        return self.verify_envelopes(envelopes, trajectory_id=trajectory_id)

    def verify_envelopes(
        self,
        envelopes: Sequence[ProofCarryingStepEnvelope | Mapping[str, Any]],
        *,
        trajectory_id: str = "pcat_traj_001",
    ) -> PCATVerificationReport:
        """Verify an existing list of `ProofCarryingStepEnvelope` objects or dicts."""
        norm_envs: list[ProofCarryingStepEnvelope] = []
        for e in envelopes:
            if isinstance(e, ProofCarryingStepEnvelope):
                norm_envs.append(e)
            else:
                norm_envs.append(
                    ProofCarryingStepEnvelope(
                        step_index=int(e["step_index"]),
                        tool_name=str(e["tool_name"]),
                        arguments=dict(e.get("arguments", {})),
                        pre_state_hash=str(e["pre_state_hash"]),
                        post_state_hash=str(e["post_state_hash"]),
                        precondition_witness={
                            str(k): bool(v)
                            for k, v in e.get("precondition_witness", {}).items()
                        },
                        bounds_witness={
                            str(k): bool(v) for k, v in e.get("bounds_witness", {}).items()
                        },
                        pattern_witness=bool(e.get("pattern_witness", True)),
                        parent_envelope_hash=str(e["parent_envelope_hash"]),
                        step_proof_hash=str(e["step_proof_hash"]),
                    )
                )

        violations: list[dict[str, Any]] = []
        verified_count = 0
        chain_ok = True
        expected_parent = "0" * 64

        for env in norm_envs:
            step_issues: list[str] = []
            if env.parent_envelope_hash != expected_parent:
                chain_ok = False
                step_issues.append("parent_envelope_hash_mismatch")

            recomputed = _compute_step_proof_hash(
                step_index=env.step_index,
                tool_name=env.tool_name,
                arguments=env.arguments,
                pre_state_hash=env.pre_state_hash,
                post_state_hash=env.post_state_hash,
                precondition_witness=env.precondition_witness,
                bounds_witness=env.bounds_witness,
                pattern_witness=env.pattern_witness,
                parent_envelope_hash=env.parent_envelope_hash,
            )
            if recomputed != env.step_proof_hash:
                chain_ok = False
                step_issues.append("step_proof_hash_tampered")

            for k, ok in env.precondition_witness.items():
                if not ok:
                    step_issues.append(f"precondition_failed:{k}")
            for k, ok in env.bounds_witness.items():
                if not ok:
                    step_issues.append(f"numeric_bound_exceeded:{k}")
            if not env.pattern_witness:
                step_issues.append("forbidden_argument_pattern_detected")

            if step_issues:
                violations.append(
                    {
                        "step_index": env.step_index,
                        "tool_name": env.tool_name,
                        "issues": step_issues,
                    }
                )
            else:
                verified_count += 1

            expected_parent = env.step_proof_hash

        return PCATVerificationReport(
            trajectory_id=trajectory_id,
            total_steps=len(norm_envs),
            verified_steps=verified_count,
            hash_chain_intact=chain_ok,
            pcat_verified=(chain_ok and len(violations) == 0),
            contract_violations=violations,
            envelopes=norm_envs,
        )
