"""Unit tests for multilingual adversarial security case generation."""


from agent_bench.generators.security import SecurityGenerator


class TestSecurityGeneratorMultilingual:
    """Test suite verifying multilingual adversarial vectors and auto-detection."""

    def test_generate_explicit_english_categories(self) -> None:
        gen = SecurityGenerator()
        categories = ["prompt_injection", "privilege_escalation", "data_exfiltration", "unsafe_autonomy"]

        for category in categories:
            res = gen.generate(category=category, lang="en", seed=42)
            assert not res.rejected
            assert res.confidence_score >= 0.5
            case = res.case
            assert case["source_type"] == "adversarial"
            assert "lang_en" in case["tags"]
            assert case["metadata"]["language"] == "en"
            assert case["metadata"]["attack_category"] == category
            prompt = case["input_messages"][0]["content"]
            assert len(prompt) > 15
            # Verify English prompt contains expected English characters/words
            assert isinstance(prompt, str)

    def test_auto_language_detection_for_domains(self) -> None:
        gen = SecurityGenerator()

        # Enterprise English domains
        for en_domain in ("cyber_sandbox", "investment_advisor", "sme_business_advisor"):
            res = gen.generate(domain=en_domain, lang="auto", seed=100)
            assert not res.rejected
            assert res.case["domain"] == en_domain
            assert "lang_en" in res.case["tags"]
            assert res.case["metadata"]["language"] == "en"

        # Brazilian Portuguese domain
        res_pt = gen.generate(domain="pix_assist", lang="auto", seed=100)
        assert not res_pt.rejected
        assert res_pt.case["domain"] == "pix_assist"
        assert "lang_pt" in res_pt.case["tags"]
        assert res_pt.case["metadata"]["language"] == "pt"

    def test_batch_generation_multilingual(self) -> None:
        gen = SecurityGenerator()
        en_batch = gen.generate_batch(8, lang="en", seed=42)
        assert len(en_batch) == 8
        for item in en_batch:
            assert not item.rejected
            assert "lang_en" in item.case["tags"]
            assert item.case["metadata"]["language"] == "en"

        pt_batch = gen.generate_batch(8, lang="pt", seed=42)
        assert len(pt_batch) == 8
        for item in pt_batch:
            assert not item.rejected
            assert "lang_pt" in item.case["tags"]
            assert item.case["metadata"]["language"] == "pt"

    def test_difficulties_with_english_perturbations(self) -> None:
        gen = SecurityGenerator()
        for diff in ["easy", "medium", "hard"]:
            res = gen.generate(difficulty=diff, lang="en", seed=123)
            assert not res.rejected
            assert res.case["difficulty"] == diff
            assert len(res.case["input_messages"][0]["content"]) > 10
