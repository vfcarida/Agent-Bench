# Scientific, Executive & Ecosystem References

This document catalogs the peer-reviewed academic literature, frontier AI lab research, executive governance frameworks, and open-source specifications that ground the architecture, mathematical formulations, and verification protocols of **Agent-Bench**.

---

## 1. Scientific & Frontier AI Lab Literature

### Agentic Evaluation, Dual-Control Environments & Psychometrics
1. **Barres et al. (2025)** — *"$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment"*, arXiv:2506.07982. Grounds `DualControlUserSimulator` (`core/user_simulator.py`) and `Pass^k` ($\text{Pass}\hat{}k$) reliability metrics.
2. **Yao et al. (2024)** — *"$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains"*, arXiv:2406.12045. Grounds multi-turn state-mutation grading and tool-use trajectory evaluation.
3. **Polo et al. (ICML 2024)** — *"tinyBenchmarks: evaluating LLMs with fewer examples"*, arXiv:2402.14992. Grounds 2-Parameter Logistic (2PL) Item Response Theory (`metrics/irt.py`) and Fisher-Information anchor task selection.
4. **Lord, F. M. (1980)** — *"Applications of Item Response Theory to Practical Testing Problems"*, Lawrence Erlbaum Associates; & **van der Linden, W. J., & Glas, C. A. W. (2010)** — *"Elements of Adaptive Testing"*, Springer. Grounds the sequential Bayesian Computerized Adaptive Testing (`BayesianCATEngine` in `metrics/adaptive_cat.py`).
5. **Wald, A. (1945)** — *"Sequential Tests of Statistical Hypotheses"*, Annals of Mathematical Statistics, 16(2), 117–186. Grounds `evaluate_wald_sprt` (`metrics/expanded.py`) for early-stopping regression detection.
6. **Wilson, E. B. (1927)** — *"Probable Inference, the Law of Succession, and Statistical Inference"*, Journal of the American Statistical Association, 22(158), 209–212. Grounds `compute_wilson_score_interval` (`metrics/expanded.py`).
7. **Zitzler, E., & Thiele, L. (1998)** — *"Multiobjective Optimization Using Evolutionary Algorithms — A Comparative Case Study"*, PPSN V, Springer. Grounds 2D Exact Hypervolume Indicator (`compute_hypervolume_2d` in `metrics/pareto.py`).

### Mechanistic Interpretability, Sparse Autoencoders & Circuit Tracing
8. **Bricken et al. (Anthropic, 2023)** — *"Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"*, Transformer Circuits Thread. Grounds `SparseAutoencoderDictionary` and `SAECircuitAttributor` (`graders/sae_attribution.py`).
9. **Templeton et al. (Anthropic, 2024)** — *"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"*, Transformer Circuits Thread. Grounds monosemantic safety feature attribution and clamping interventions (`graders/activation_steering.py`).
10. **Ameisen et al. / Lindsey et al. (Anthropic, 2025)** — *"Circuit Tracing: Revealing Computational Graphs in Language Models"*, Transformer Circuits Thread. Grounds `CrossLayerTranscoder` (CLT) and `CircuitGraphTracer` (`graders/circuit_tracer.py`).
11. **Zou et al. (2023)** — *"Representation Engineering: A Top-Down Approach to AI Transparency"*, arXiv:2310.01405; & **Turner et al. (2023)** — *"Activation Addition: Steering Language Models Without Optimization"*, arXiv:2308.10248. Grounds `LinearConceptProbe`, `ActivationSafetyMonitor` (`graders/activation_probe.py`), and `MechanisticActivationSteerer` (`graders/activation_steering.py`).
12. **Turpin et al. (NeurIPS 2023)** — *"Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting"*, arXiv:2305.04388. Grounds `CoTFaithfulnessMonitor` (`graders/activation_probe.py`).

### Formal Methods, Proof-Carrying Code & Cryptographic Verification
13. **Necula, G. C. (POPL 1997)** — *"Proof-Carrying Code"*, Proceedings of the 24th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages, 106–119. Grounds `PCATVerifier` and `ProofCarryingStepEnvelope` (`governance/proof_carrying_trajectory.py`).
14. **De Giacomo, G., & Vardi, M. Y. (IJCAI 2013)** — *"Linear Temporal Logic and Linear Dynamic Logic on Finite Traces"*, IJCAI. Grounds `SymbolicTrajectoryVerifier` (`graders/formal_verifier.py`) for LTLf/MTL safety invariant verification.
15. **Kothapalli, A., Setty, S., & Tzialla, I. (CRYPTO 2022)** — *"Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"*, IACR Cryptology ePrint Archive 2021/370. Grounds `NovaIVCFolder` and `RelaxedR1CSInstance` (`governance/ivc_folding.py`).
16. **Parno, B., Howell, J., Gentry, C., & Raykova, M. (IEEE S&P 2013)** — *"Pinocchio: Nearly Practical Verifiable Computation"*. Grounds `ZKEvaluationCircuitCompiler` (`governance/zk_circuit_compiler.py`) over the Goldilocks prime field ($p = 2^{64} - 2^{32} + 1$).
17. **Dwork, C., & Roth, A. (2014)** — *"The Algorithmic Foundations of Differential Privacy"*, Foundations and Trends in Theoretical Computer Science, 9(3–4), 211–407. Grounds `FederatedBenchmarkAggregator` (`governance/federated_aggregator.py`) and `ZKFederatedLeaderboardAggregator` (`governance/zk_federated_leaderboard.py`).
18. **Merkle, R. C. (CRYPTO 1987)** — *"A Digital Signature Based on a Conventional Encryption Function"*. Grounds `MerkleExecutionAttestor` (`governance/zk_attestation.py`).

### Multi-Agent Systems, Game Theory, Red-Teaming & World Simulation
19. **Castro, M., & Liskov, B. (OSDI 1999)** — *"Practical Byzantine Fault Tolerance"*. Grounds `ByzantineOrchestratorRunner` (`runners/byzantine_orchestrator.py`).
20. **Shapley, L. S. (1953)** — *"A Value for n-Person Games"*, Contributions to the Theory of Games, II, 307–317. Grounds `ShapleyContributionAuditor` and `MultiAgentCollusionDetector` (`governance/collusion_detector.py`).
21. **Mouret, J.-B., & Clune, J. (2015)** — *"Illuminating search spaces by mapping elites"*, arXiv:1504.04909; & **Samvelyan et al. (2024)** — *"Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts"*, NeurIPS 2024, arXiv:2402.16822. Grounds `EvolutionaryRedTeamFuzzer` and `MAPElitesArchive` (`generators/evolutionary_fuzzer.py`).
22. **Kyle, A. S. (1985)** — *"Continuous Auctions and Insider Trading"*, Econometrica, 53(6), 1315–1335; & **Brunnermeier, M. K., & Pedersen, L. H. (2009)** — *"Market Liquidity and Funding Liquidity"*, Review of Financial Studies, 22(6), 2201–2238. Grounds `SystemicContagionSandbox` (`environments/market_contagion_sandbox.py`) and `NonStationaryWorldSimulator` (`environments/regime_shift_simulator.py`).
23. **Pearl, J. (2009)** — *"Causality: Models, Reasoning, and Inference"* (2nd ed.), Cambridge University Press. Grounds `CounterfactualSandbox` (`runners/counterfactual.py`).
24. **Page, E. S. (1954)** — *"Continuous Inspection Schemes"*, Biometrika, 41(1/2), 100–115. Grounds `CUSUMDriftDetector` and `OnlineDriftMonitor` (`runners/online_eval.py`).
25. **Cormack, G. V., Clarke, C. L., & Buettcher, S. (SIGIR 2009)** — *"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"*. Grounds `HybridRRFCorpusRetriever` (`retrieval/hybrid_retriever.py`).

---

## 2. Executive, Analyst & Regulatory Frameworks

1. **Gartner AI TRiSM (AI Trust, Risk, and Security Management)** — Runtime enforcement of model interpretability, data privacy, adversarial resistance, and cryptographic auditability across autonomous agent deployments.
2. **MIT Sloan Management Review / BCG AI at Scale** — Risk-Adjusted Autonomy Tiers (`L0_BLOCKED` through `L4_AUTONOMOUS_BOUNDED` in `metrics/pareto.py`) linking empirical safety compliance and Pass^k reliability to operational deployment gates.
3. **McKinsey Digital / Agentic AI Risk Governance** — Dual-control human-in-the-loop authorization (`DualControlUserSimulator`), tool blast-radius isolation (`CDPBrowserSandbox`, `PCATVerifier`), and automated self-healing guardrail synthesis (`SelfHealingPolicySynthesizer`).
4. **NIST AI Risk Management Framework (AI RMF 1.0 & Generative AI Profile NIST AI 600-1)** — Quantitative measurement (`Measure`), continuous drift monitoring (`Manage`), and cross-jurisdictional governance (`Govern`).
5. **Cross-Border Statutory Frameworks (`governance/regulatory_harmonizer.py`)**:
   - **Brazil:** Banco Central do Brasil (BACEN) PIX Mecanismo Especial de Devolução (MED) & Lei Geral de Proteção de Dados (LGPD, Law No. 13.709/2018, Arts. 16 & 18).
   - **European Union:** EU AI Act (Regulation (EU) 2024/1689, Art. 14 Human Oversight for High-Risk AI) & GDPR (Regulation (EU) 2016/679, Art. 17 Right to Erasure).
   - **United States:** SEC Rule 17a-4 (17 CFR § 240.17a-4) 7-year immutable audit trail retention & FINRA Rule 2111 Suitability.

---

## 3. Open-Source Standards & Interoperability Exemplars

1. **UK AISI Inspect AI** — Interoperable dataset/solver export format (`export/inspect_export.py`, `export/inspect_solver.py`).
2. **Model Context Protocol (MCP)** — JSON-RPC 2.0 tool discovery and execution bridge (`tools/mcp_bridge.py`).
3. **OpenTelemetry (OTel) GenAI Semantic Conventions (`gen_ai.*`)** — Standardized span export (`utils/observability.py`, `bench export-otel`).
4. **W3C WAI-ARIA 1.2 & Chrome DevTools Protocol (CDP)** — Accessibility tree observation and network egress firewalling (`environments/gui_navigation_runtime.py`, `environments/cdp_browser_bridge.py`).
5. **AWS Nitro Enclaves / Intel TDX / AMD SEV-SNP Attestation Architecture** — PCR0/PCR1/PCR2/PCR8 cryptographic enclave measurement and Golden PCR verification (`governance/tee_enclave_attestation.py`).
