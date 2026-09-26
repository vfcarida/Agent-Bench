"""Unit tests for multilingual perturbations (Portuguese and International English)."""

from agent_bench.generators.perturbations import (
    add_ambiguity,
    add_urgency,
    apply_noise,
    inject_distraction,
    vary_formality,
)


class TestMultilingualPerturbations:
    def test_english_noise_with_typos(self) -> None:
        text = "Please transfer the money to my account because I need it with tomorrow balance."
        perturbed = apply_noise(text, noise_level=1.0, seed=42, lang="en")
        # Should substitute words from English typo map
        assert "pls" in perturbed or "plz" in perturbed or "teh" in perturbed or "acct" in perturbed

    def test_english_urgency(self) -> None:
        text = "Execute the security patch on firewall cluster 2."
        urgent = add_urgency(text, seed=42, lang="en")
        assert urgent.startswith(("URGENT:", "Need this immediately:", "Emergency!", "Please expedite:"))
        assert "execute the security patch" in urgent.lower()

    def test_english_ambiguity_dollar(self) -> None:
        text = "Send $500 to account 12345."
        ambiguous = add_ambiguity(text, seed=42, lang="en")
        assert any(term in ambiguous for term in ["about $500", "around $500", "roughly $500", "approximately $500"])

    def test_english_distraction(self) -> None:
        text = "Analyze the SME quarterly profit report."
        distracted = inject_distraction(text, seed=42, lang="en")
        assert any(
            phrase in distracted
            for phrase in [
                "Sorry for the delay",
                "My phone app kept crashing",
                "A colleague recommended",
            ]
        )

    def test_english_formality_formal(self) -> None:
        text = "Transfer $1,000 to John."
        formal = vary_formality(text, level="formal", seed=42, lang="en")
        assert formal.startswith((
            "Dear Support Team",
            "I would kindly appreciate",
            "Could you please be so kind",
            "I hereby request",
        ))

    def test_english_formality_informal(self) -> None:
        text = "Please transfer the funds."
        informal = vary_formality(text, level="informal", seed=42, lang="en")
        assert informal.startswith(("Hey, ", "Hi there, ", "Quick question: ", "Yo, ", "Morning, "))

    def test_auto_detection_pt_vs_en(self) -> None:
        pt_text = "Por favor faça o pix de R$100 para a chave cpf."
        en_text = "Please transfer $100 to the recipient account."

        pt_urgent = add_urgency(pt_text, seed=1)
        en_urgent = add_urgency(en_text, seed=1)

        assert any(m in pt_urgent for m in ["URGENTE", "Preciso AGORA", "Eh urgente", "Rapido"])
        assert any(m in en_urgent for m in ["URGENT", "Need this", "Emergency", "time sensitive"])
