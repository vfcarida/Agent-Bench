"""Agent-Bench: Enterprise benchmark framework for evaluating LLM agents as complete systems."""

from agent_bench.core.adapters import (
    JudgeAdapter,
    JudgeVerdict,
    ModelAdapter,
    SafetyVerdict,
    ToolAdapter,
)
from agent_bench.core.artifacts import RunArtifact, TraceEvent, TraceEventType
from agent_bench.core.config import BenchConfig, SuiteConfig, SystemConfig, load_config
from agent_bench.core.metrics import MetricCategory, MetricResult, Scorecard
from agent_bench.core.protocols import AgentRunner, Evaluator, TaskEnvironment
from agent_bench.core.scenarios import RefusalMode, Severity, Task
from agent_bench.core.schema_v2 import Difficulty, EvalCase, Family, GradingStrategy, Split
from agent_bench.core.user_simulator import (
    DualControlAction,
    DualControlUserSimulator,
    HumanUserSimulator,
    RuleBasedUserSimulator,
    ScriptedUserSimulator,
    UserSimulator,
)
from agent_bench.environments.cdp_browser_bridge import (
    CDPBrowserSandbox,
    EgressViolationEvent,
    NetworkEgressFirewall,
)
from agent_bench.environments.gui_navigation_runtime import (
    GUINavigationRuntime,
    GUIScreenNode,
)
from agent_bench.environments.market_contagion_sandbox import (
    OrderBookSnapshot,
    SystemicContagionReport,
    SystemicContagionSandbox,
)
from agent_bench.environments.multimodal_sandbox import (
    DocumentArtifact,
    MultiModalFinancialSandbox,
    UIElementState,
)
from agent_bench.environments.regime_shift_simulator import (
    NonStationaryWorldSimulator,
    RegimeAdaptationReport,
    RegimeShiftShock,
)
from agent_bench.generators.evolutionary_fuzzer import (
    EvolutionaryRedTeamFuzzer,
    MAPElitesArchive,
    MutationStrategy,
    RedTeamCandidate,
)
from agent_bench.governance.collusion_detector import (
    CollusionAuditReport,
    MultiAgentCollusionDetector,
    PairwiseCollusionSignal,
    ShapleyContributionAuditor,
)
from agent_bench.governance.federated_aggregator import (
    FederatedAggregationReport,
    FederatedBenchmarkAggregator,
    FederatedSiteSubmission,
    PrivacyBudgetAccountant,
    PrivacyBudgetExceededError,
)
from agent_bench.governance.ivc_folding import (
    IVCAccumulatorBundle,
    IVCFoldingStepProof,
    NovaIVCFolder,
    RelaxedR1CSInstance,
)
from agent_bench.governance.policy_synthesizer import (
    SelfHealingPolicySynthesizer,
    SynthesizedPolicyPatch,
)
from agent_bench.governance.proof_carrying_trajectory import (
    PCATVerificationReport,
    PCATVerifier,
    ProofCarryingStepEnvelope,
    VerifiableToolContract,
)
from agent_bench.governance.regulatory_harmonizer import (
    ConflictOfLawsResolution,
    JurisdictionalRule,
    MultiJurisdictionComplianceReport,
    RegulatoryComplianceHarmonizer,
)
from agent_bench.governance.tee_enclave_attestation import (
    ConfidentialEnclaveAttestor,
    PCRMeasurementSet,
    TEEAttestationQuote,
)
from agent_bench.governance.zk_attestation import (
    MerkleExecutionAttestor,
    MerkleInclusionProof,
    RunAttestationBundle,
    SelectiveDisclosureClaim,
)
from agent_bench.governance.zk_circuit_compiler import (
    GOLDILOCKS_PRIME,
    R1CSConstraint,
    R1CSWitnessProof,
    ZKArithmeticCircuit,
    ZKEvaluationCircuitCompiler,
)
from agent_bench.governance.zk_federated_leaderboard import (
    ZKFederatedLeaderboardAggregator,
    ZKFederatedLeaderboardBundle,
    ZKInstitutionSubmission,
)
from agent_bench.graders.activation_probe import (
    ActivationProbeVerdict,
    ActivationSafetyMonitor,
    CoTFaithfulnessMonitor,
    LinearConceptProbe,
)
from agent_bench.graders.activation_steering import (
    FeatureClampIntervention,
    MechanisticActivationSteerer,
    SteeringEvaluationReport,
)
from agent_bench.graders.circuit_tracer import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
    CircuitGraphTracer,
    CrossLayerTranscoder,
)
from agent_bench.graders.formal_verifier import (
    FormalVerificationReport,
    SymbolicInvariant,
    SymbolicTrajectoryVerifier,
    TemporalOperator,
)
from agent_bench.graders.gated_evaluator import GatedEvaluator
from agent_bench.graders.sae_attribution import (
    SAEAttributionReport,
    SAECircuitAttributor,
    SAEFeatureActivation,
    SAEFeatureSpec,
    SparseAutoencoderDictionary,
)
from agent_bench.graders.safety_gate import evaluate_safety
from agent_bench.metrics.adaptive_cat import (
    BayesianCATEngine,
    CATSessionReport,
    CATStepTelemetry,
)
from agent_bench.metrics.compute import compute_pass_hat_k, compute_pass_k, compute_task_metrics
from agent_bench.metrics.expanded import (
    SPRTDecision,
    compute_bootstrap_ci,
    compute_wilson_score_interval,
    evaluate_wald_sprt,
)
from agent_bench.metrics.irt import (
    IRTItemParameters,
    IRTModelFit,
    estimate_full_score_from_anchor,
    fit_2pl_irt,
    select_irt_anchor_tasks,
)
from agent_bench.metrics.pareto import (
    AutonomyTier,
    classify_autonomy_tier,
    compute_hypervolume_2d,
    compute_pareto_frontier,
)
from agent_bench.metrics.scorecard import compute_balanced_profile, compute_scorecard
from agent_bench.retrieval.hybrid_retriever import (
    CorpusPoisoningSanitizer,
    HybridRRFCorpusRetriever,
    PoisoningScanVerdict,
)
from agent_bench.runners.byzantine_orchestrator import (
    ByzantineFaultMode,
    ByzantineOrchestratorRunner,
    HandoffRecord,
    SubAgentSpec,
    compute_byzantine_resilience_metrics,
)
from agent_bench.runners.callable_runner import CallableAgentRunner
from agent_bench.runners.case_runner import DefaultAgentRunner, execute_task
from agent_bench.runners.counterfactual import (
    CounterfactualIntervention,
    CounterfactualReplayResult,
    CounterfactualSandbox,
)
from agent_bench.runners.episodic_memory_stress import (
    EpisodicConstraint,
    EpisodicMemoryStressRunner,
    EpisodicStressReport,
    InterferenceTurn,
    MemoryInterferenceInjector,
)
from agent_bench.runners.online_eval import (
    CUSUMDriftDetector,
    OnlineDriftMonitor,
    OnlineDriftReport,
    compute_jensen_shannon_divergence,
)
from agent_bench.runners.scripted import ScriptedAgentRunner
from agent_bench.runners.suite_runner import run_suite
from agent_bench.tools.mcp_bridge import (
    MCPClientToolAdapter,
    MCPToolServer,
    discover_mcp_tools,
)
from agent_bench.utils.observability import convert_run_traces_to_otel_genai

__version__ = "1.0.0"
__author__ = "Vinicius Caridá"

__all__ = [
    "__author__",
    "__version__",
    "ActivationProbeVerdict",
    "ActivationSafetyMonitor",
    "AgentRunner",
    "AttributionEdge",
    "AttributionGraph",
    "AttributionNode",
    "AutonomyTier",
    "BayesianCATEngine",
    "BenchConfig",
    "ByzantineFaultMode",
    "ByzantineOrchestratorRunner",
    "CATSessionReport",
    "CATStepTelemetry",
    "CDPBrowserSandbox",
    "CUSUMDriftDetector",
    "CallableAgentRunner",
    "CircuitGraphTracer",
    "CoTFaithfulnessMonitor",
    "CollusionAuditReport",
    "ConfidentialEnclaveAttestor",
    "ConflictOfLawsResolution",
    "CorpusPoisoningSanitizer",
    "CounterfactualIntervention",
    "CounterfactualReplayResult",
    "CounterfactualSandbox",
    "CrossLayerTranscoder",
    "DefaultAgentRunner",
    "Difficulty",
    "DocumentArtifact",
    "DualControlAction",
    "DualControlUserSimulator",
    "EgressViolationEvent",
    "EpisodicConstraint",
    "EpisodicMemoryStressRunner",
    "EpisodicStressReport",
    "EvalCase",
    "Evaluator",
    "EvolutionaryRedTeamFuzzer",
    "Family",
    "FeatureClampIntervention",
    "FederatedAggregationReport",
    "FederatedBenchmarkAggregator",
    "FederatedSiteSubmission",
    "FormalVerificationReport",
    "GOLDILOCKS_PRIME",
    "GUINavigationRuntime",
    "GUIScreenNode",
    "GatedEvaluator",
    "GradingStrategy",
    "HandoffRecord",
    "HumanUserSimulator",
    "HybridRRFCorpusRetriever",
    "IRTItemParameters",
    "IRTModelFit",
    "IVCAccumulatorBundle",
    "IVCFoldingStepProof",
    "InterferenceTurn",
    "JudgeAdapter",
    "JudgeVerdict",
    "JurisdictionalRule",
    "LinearConceptProbe",
    "MAPElitesArchive",
    "MCPClientToolAdapter",
    "MCPToolServer",
    "MechanisticActivationSteerer",
    "MemoryInterferenceInjector",
    "MerkleExecutionAttestor",
    "MerkleInclusionProof",
    "MetricCategory",
    "MetricResult",
    "ModelAdapter",
    "MultiAgentCollusionDetector",
    "MultiJurisdictionComplianceReport",
    "MultiModalFinancialSandbox",
    "MutationStrategy",
    "NetworkEgressFirewall",
    "NonStationaryWorldSimulator",
    "NovaIVCFolder",
    "OnlineDriftMonitor",
    "OnlineDriftReport",
    "OrderBookSnapshot",
    "PCATVerificationReport",
    "PCATVerifier",
    "PCRMeasurementSet",
    "PairwiseCollusionSignal",
    "PoisoningScanVerdict",
    "PrivacyBudgetAccountant",
    "PrivacyBudgetExceededError",
    "ProofCarryingStepEnvelope",
    "R1CSConstraint",
    "R1CSWitnessProof",
    "RedTeamCandidate",
    "RefusalMode",
    "RegimeAdaptationReport",
    "RegimeShiftShock",
    "RegulatoryComplianceHarmonizer",
    "RelaxedR1CSInstance",
    "RuleBasedUserSimulator",
    "RunArtifact",
    "RunAttestationBundle",
    "SAEAttributionReport",
    "SAECircuitAttributor",
    "SAEFeatureActivation",
    "SAEFeatureSpec",
    "SPRTDecision",
    "SafetyVerdict",
    "Scorecard",
    "ScriptedAgentRunner",
    "ScriptedUserSimulator",
    "SelectiveDisclosureClaim",
    "SelfHealingPolicySynthesizer",
    "Severity",
    "ShapleyContributionAuditor",
    "SparseAutoencoderDictionary",
    "Split",
    "SteeringEvaluationReport",
    "SubAgentSpec",
    "SuiteConfig",
    "SymbolicInvariant",
    "SymbolicTrajectoryVerifier",
    "SynthesizedPolicyPatch",
    "SystemConfig",
    "SystemicContagionReport",
    "SystemicContagionSandbox",
    "TEEAttestationQuote",
    "Task",
    "TaskEnvironment",
    "TemporalOperator",
    "ToolAdapter",
    "TraceEvent",
    "TraceEventType",
    "UIElementState",
    "UserSimulator",
    "VerifiableToolContract",
    "ZKArithmeticCircuit",
    "ZKEvaluationCircuitCompiler",
    "ZKFederatedLeaderboardAggregator",
    "ZKFederatedLeaderboardBundle",
    "ZKInstitutionSubmission",
    "classify_autonomy_tier",
    "compute_balanced_profile",
    "compute_bootstrap_ci",
    "compute_byzantine_resilience_metrics",
    "compute_hypervolume_2d",
    "compute_jensen_shannon_divergence",
    "compute_pareto_frontier",
    "compute_pass_hat_k",
    "compute_pass_k",
    "compute_scorecard",
    "compute_task_metrics",
    "compute_wilson_score_interval",
    "convert_run_traces_to_otel_genai",
    "discover_mcp_tools",
    "estimate_full_score_from_anchor",
    "evaluate_safety",
    "evaluate_wald_sprt",
    "execute_task",
    "fit_2pl_irt",
    "load_config",
    "run_suite",
    "select_irt_anchor_tasks",
]
