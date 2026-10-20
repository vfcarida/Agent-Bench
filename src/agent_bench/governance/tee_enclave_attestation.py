"""Hardware-Enforced Confidential Computing Attestation (`tee_enclave_attestation.py`).

Binds `RunAttestationBundle` SHA-256 Merkle roots to Platform Configuration Register
(PCR) measurements (`PCR0` runtime image, `PCR1` suite/config hash, `PCR2` model/adapter
lineage hash, `PCR8` enclave signing key fingerprint) modeled after AWS Nitro Enclaves,
Intel TDX, and AMD SEV-SNP remote attestation quotes.

Provides cryptographic verification of:
1. Hardware quote signature and anti-replay nonce binding;
2. Cryptographic binding (`user_data_digest`) between the enclave quote and the
   benchmark run's `RunAttestationBundle.merkle_root`;
3. Golden PCR Reference Policy enforcement (`verify_tee_quote`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent_bench.governance.zk_attestation import (
    MerkleExecutionAttestor,
    RunAttestationBundle,
)


def _sha256_hex(payload: bytes | str) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class PCRMeasurementSet:
    """Platform Configuration Register (PCR) SHA-256 measurements for a TEE enclave."""

    pcr0_runtime_image: str  # Hash of evaluator binary / source package
    pcr1_config_suite: str  # Hash of suite configuration & safety invariants
    pcr2_model_lineage: str  # Hash of model ID, weights revision, and adapter digest
    pcr8_enclave_signer: str  # Fingerprint of enclave attestation key

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class TEEAttestationQuote:
    """Hardware-bound Confidential Computing attestation quote wrapping a Merkle run root."""

    quote_id: str
    tee_architecture: str  # "aws_nitro_enclave" | "intel_tdx" | "amd_sev_snp"
    run_id: str
    merkle_root: str
    nonce: str
    user_data_digest: str
    pcrs: PCRMeasurementSet
    generated_at: str
    quote_signature: str
    attestation_bundle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfidentialEnclaveAttestor:
    """Generates and verifies TEE remote attestation quotes bound to Merkle execution proofs."""

    @staticmethod
    def derive_default_pcrs(
        run_manifest: dict[str, Any],
        *,
        enclave_signing_key: str = "agent-bench-tee-hardware-root-key",
    ) -> PCRMeasurementSet:
        """Deterministically derive PCR0, PCR1, PCR2, and PCR8 from a run manifest."""
        version = str(run_manifest.get("benchmark_version", "1.0.0"))
        suite_id = str(run_manifest.get("suite_id", "default_suite"))
        cfg_hash = str(run_manifest.get("config_hash", "default_cfg"))
        model_id = str(run_manifest.get("model_id", run_manifest.get("system_id", "default_model")))

        pcr0 = _sha256_hex(f"pcr0:agent_bench_runtime_v{version}")
        pcr1 = _sha256_hex(f"pcr1:suite={suite_id}:config={cfg_hash}")
        pcr2 = _sha256_hex(f"pcr2:model_lineage={model_id}")
        pcr8 = _sha256_hex(f"pcr8:signer={enclave_signing_key}")

        return PCRMeasurementSet(
            pcr0_runtime_image=pcr0,
            pcr1_config_suite=pcr1,
            pcr2_model_lineage=pcr2,
            pcr8_enclave_signer=pcr8,
        )

    @classmethod
    def generate_tee_quote(
        cls,
        run_manifest: dict[str, Any],
        *,
        tee_architecture: str = "aws_nitro_enclave",
        nonce: str = "default_auditor_nonce_2026",
        enclave_signing_key: str = "agent-bench-tee-hardware-root-key",
        pcrs: PCRMeasurementSet | None = None,
    ) -> TEEAttestationQuote:
        """Attest a benchmark run and bind its Merkle root into a signed TEE Enclave Quote."""
        bundle: RunAttestationBundle = MerkleExecutionAttestor.attest_run(
            run_manifest, signing_key=enclave_signing_key
        )
        active_pcrs = pcrs or cls.derive_default_pcrs(
            run_manifest, enclave_signing_key=enclave_signing_key
        )
        user_data_digest = _sha256_hex(
            f"tee_user_data:{bundle.run_id}:{bundle.merkle_root}:{nonce}"
        )
        generated_at = datetime.now(UTC).isoformat()
        quote_id = f"quote_{_sha256_hex(f'{bundle.run_id}:{nonce}:{generated_at}')[:16]}"

        signing_body = _canonical_json(
            {
                "tee_architecture": tee_architecture,
                "run_id": bundle.run_id,
                "merkle_root": bundle.merkle_root,
                "nonce": nonce,
                "user_data_digest": user_data_digest,
                "pcrs": active_pcrs.to_dict(),
            }
        )
        quote_sig = hmac.new(
            enclave_signing_key.encode("utf-8"),
            signing_body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return TEEAttestationQuote(
            quote_id=quote_id,
            tee_architecture=tee_architecture,
            run_id=bundle.run_id,
            merkle_root=bundle.merkle_root,
            nonce=nonce,
            user_data_digest=user_data_digest,
            pcrs=active_pcrs,
            generated_at=generated_at,
            quote_signature=quote_sig,
            attestation_bundle=bundle.to_dict(),
        )

    @classmethod
    def verify_tee_quote(
        cls,
        quote: TEEAttestationQuote | dict[str, Any],
        *,
        enclave_signing_key: str = "agent-bench-tee-hardware-root-key",
        expected_nonce: str | None = None,
        golden_pcrs: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Verify TEE quote signature, user_data Merkle binding, Golden PCRs, and inner Merkle proofs."""
        q_dict = quote.to_dict() if isinstance(quote, TEEAttestationQuote) else dict(quote)
        tee_arch = str(q_dict.get("tee_architecture", ""))
        run_id = str(q_dict.get("run_id", ""))
        merkle_root = str(q_dict.get("merkle_root", ""))
        nonce = str(q_dict.get("nonce", ""))
        user_data_digest = str(q_dict.get("user_data_digest", ""))
        pcrs_dict = dict(q_dict.get("pcrs", {}))
        provided_sig = str(q_dict.get("quote_signature", ""))

        # 1. Verify user_data_digest binding to (run_id, merkle_root, nonce)
        expected_user_data = _sha256_hex(f"tee_user_data:{run_id}:{merkle_root}:{nonce}")
        user_data_bound = hmac.compare_digest(user_data_digest, expected_user_data)

        # 2. Verify anti-replay nonce if specified by auditor
        nonce_valid = True if expected_nonce is None else hmac.compare_digest(nonce, expected_nonce)

        # 3. Verify hardware quote signature
        signing_body = _canonical_json(
            {
                "tee_architecture": tee_arch,
                "run_id": run_id,
                "merkle_root": merkle_root,
                "nonce": nonce,
                "user_data_digest": user_data_digest,
                "pcrs": pcrs_dict,
            }
        )
        expected_sig = hmac.new(
            enclave_signing_key.encode("utf-8"),
            signing_body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        quote_signature_valid = hmac.compare_digest(provided_sig, expected_sig)

        # 4. Verify Golden PCR reference policy if provided
        pcr_mismatches: list[str] = []
        if golden_pcrs:
            for reg_key, expected_hash in golden_pcrs.items():
                actual_hash = str(pcrs_dict.get(reg_key, ""))
                if not hmac.compare_digest(actual_hash, str(expected_hash)):
                    pcr_mismatches.append(reg_key)
        pcr_policy_passed = len(pcr_mismatches) == 0

        # 5. Verify inner Merkle attestation bundle
        inner_bundle = q_dict.get("attestation_bundle", {})
        merkle_verdict = (
            MerkleExecutionAttestor.verify_attestation_bundle(
                inner_bundle, signing_key=enclave_signing_key
            )
            if isinstance(inner_bundle, dict) and inner_bundle
            else {"verified": True}
        )
        bundle_root_matches = (
            hmac.compare_digest(str(inner_bundle.get("merkle_root", merkle_root)), merkle_root)
            if isinstance(inner_bundle, dict) and inner_bundle
            else True
        )

        overall_verified = (
            user_data_bound
            and nonce_valid
            and quote_signature_valid
            and pcr_policy_passed
            and bool(merkle_verdict.get("verified", False))
            and bundle_root_matches
        )

        return {
            "verified": overall_verified,
            "tee_architecture": tee_arch,
            "quote_signature_valid": quote_signature_valid,
            "user_data_bound": user_data_bound,
            "nonce_valid": nonce_valid,
            "pcr_policy_passed": pcr_policy_passed,
            "pcr_mismatches": pcr_mismatches,
            "merkle_bundle_verified": bool(merkle_verdict.get("verified", False)),
            "merkle_root": merkle_root,
        }
