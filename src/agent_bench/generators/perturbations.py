"""Text perturbations for synthetic case generation and agent robustness benchmarking (stdlib only).

Supports multilingual perturbations for both Brazilian Portuguese (pt-BR)
and international English (en-US).
"""

from __future__ import annotations

import random
import re

# Common pt-BR typos and abbreviations
_TYPO_MAP_PT = {
    "que": ["q", "ke"],
    "para": ["pra", "p/"],
    "voce": ["vc", "voce"],
    "porque": ["pq", "porq"],
    "tambem": ["tb", "tbm"],
    "nao": ["n", "nao", "naum"],
    "esta": ["ta", "tá"],
    "estou": ["to", "tô"],
    "fazer": ["fzr", "faze"],
    "dinheiro": ["din", "grana"],
    "transferir": ["transf", "transferi"],
    "reais": ["conto", "pila"],
    "obrigado": ["vlw", "obg", "valeu"],
    "por favor": ["pfv", "pf", "pfvr"],
    "conta": ["cta", "conta"],
    "telefone": ["tel", "fone"],
    "muito": ["mt", "mto"],
    "quanto": ["qto", "qnt"],
}

# Common international English typos and abbreviations
_TYPO_MAP_EN = {
    "the": ["teh"],
    "receive": ["recieve"],
    "transfer": ["transfr", "trsf"],
    "account": ["acct", "accnt"],
    "please": ["pls", "plz"],
    "thanks": ["thx", "ty"],
    "thank": ["thx"],
    "money": ["funds", "cash"],
    "because": ["bc", "cuz"],
    "with": ["w/"],
    "without": ["w/o"],
    "tomorrow": ["tmrw"],
    "balance": ["bal"],
    "information": ["info"],
    "approximate": ["approx"],
    "transaction": ["tx", "txn"],
    "urgent": ["urgnt"],
}

_URGENCY_MARKERS_PT = [
    "URGENTE! ",
    "Preciso AGORA: ",
    "Eh urgente, ",
    "Rapido por favor, ",
    "Nao pode esperar! ",
    "Emergencia! ",
    "Pelo amor de Deus, ",
    "Faz isso JA: ",
    "Preciso pra ontem: ",
]

_URGENCY_MARKERS_EN = [
    "URGENT: ",
    "Need this immediately: ",
    "This is time sensitive, ",
    "Please expedite: ",
    "Cannot wait, ",
    "Emergency! ",
    "Please resolve ASAP: ",
    "Need this right now: ",
]

_DISTRACTIONS_PT = [
    "Ah, e outra coisa, meu celular ta travando direto. Enfim, ",
    "Desculpa a demora, tava no mercado. Entao, ",
    "Meu vizinho indicou esse banco, parece bom. Bom, voltando ao assunto: ",
    "To com pressa pq tenho reuniao daqui a pouco. ",
    "Minha esposa pediu pra eu resolver isso logo. ",
    "Ja tentei no app mas deu erro, entao: ",
    "Sei que ja perguntei antes mas esqueci a resposta. ",
]

_DISTRACTIONS_EN = [
    "Sorry for the delay, was in a meeting. Anyway, ",
    "My phone app kept crashing earlier. So, ",
    "A colleague recommended this platform. Getting back to business: ",
    "I have to step out in five minutes, so quickly: ",
    "I believe I asked this last week but misplaced the confirmation. ",
]

_FORMAL_PREFIXES_PT = [
    "Prezado atendente, gostaria de solicitar ",
    "Venho por meio desta solicitar ",
    "Solicito gentilmente que ",
    "Poderia, por gentileza, ",
]

_FORMAL_PREFIXES_EN = [
    "Dear Support Team, I am writing to formally request that you ",
    "I would kindly appreciate your assistance to ",
    "Could you please be so kind as to ",
    "I hereby request authorization to ",
]

_INFORMAL_PREFIXES_PT = [
    "Ei, ",
    "Fala, ",
    "E ai, ",
    "Opa, ",
    "Mano, ",
    "Brother, ",
]

_INFORMAL_PREFIXES_EN = [
    "Hey, ",
    "Hi there, ",
    "Quick question: ",
    "Yo, ",
    "Morning, ",
]


def _resolve_lang(text: str, lang: str = "auto") -> str:
    """Resolve target language heuristic ('pt' or 'en')."""
    if lang in ("pt", "en"):
        return lang
    pt_indicators = {
        "que", "para", "voce", "não", "nao", "esta", "estou", "fazer",
        "reais", "conta", "por", "favor", "chave", "pix", "transferir",
    }
    words = {w.lower().strip(".,!?;:") for w in text.split()}
    return "pt" if len(words & pt_indicators) >= 1 else "en"


def apply_noise(
    text: str,
    noise_level: float = 0.1,
    seed: int | None = None,
    lang: str = "auto",
) -> str:
    """Apply typos and abbreviations to text based on noise_level (0-1)."""
    rng = random.Random(seed)
    resolved_lang = _resolve_lang(text, lang)
    typo_map = _TYPO_MAP_PT if resolved_lang == "pt" else _TYPO_MAP_EN

    words = text.split()
    result = []
    for word in words:
        lower = word.lower().strip(".,!?;:")
        if lower in typo_map and rng.random() < noise_level:
            replacement = rng.choice(typo_map[lower])
            trailing = ""
            if word and word[-1] in ".,!?;:":
                trailing = word[-1]
            result.append(replacement + trailing)
        else:
            if len(word) > 3 and rng.random() < noise_level * 0.3:
                i = rng.randint(1, len(word) - 2)
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            result.append(word)
    return " ".join(result)


def add_urgency(text: str, seed: int | None = None, lang: str = "auto") -> str:
    """Add urgency markers to the beginning of text."""
    rng = random.Random(seed)
    resolved_lang = _resolve_lang(text, lang)
    markers = _URGENCY_MARKERS_PT if resolved_lang == "pt" else _URGENCY_MARKERS_EN
    marker = rng.choice(markers)
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return marker + text


def add_ambiguity(text: str, seed: int | None = None, lang: str = "auto") -> str:
    """Make the request less precise by softening amounts and references."""
    rng = random.Random(seed)
    resolved_lang = _resolve_lang(text, lang)

    if resolved_lang == "pt":
        amount_match = re.search(r"R\$[\d.,]+", text)
        if amount_match:
            amount_str = amount_match.group()
            num = amount_str.replace("R$", "").replace(".", "").replace(",", ".")
            vague_options = [
                f"uns {num}",
                f"tipo {num} reais",
                f"acho que {num}",
                f"mais ou menos {num} reais",
            ]
            text = text.replace(amount_str, rng.choice(vague_options))

        softeners = [
            (r"chave (cpf|email|telefone|aleatoria)\s+\S+", "aquela chave la"),
            (r"cpf \d{3}\.\d{3}\.\d{3}-\d{2}", "o cpf dele"),
        ]
        if rng.random() < 0.5:
            for pattern, replacement in softeners:
                if re.search(pattern, text):
                    text = re.sub(pattern, replacement, text, count=1)
                    break
    else:
        amount_match = re.search(r"\$[\d.,]+", text)
        if amount_match:
            amount_str = amount_match.group()
            num = amount_str.replace("$", "").replace(",", "")
            vague_options = [
                f"about ${num}",
                f"around ${num}",
                f"roughly ${num}",
                f"approximately ${num}",
            ]
            text = text.replace(amount_str, rng.choice(vague_options))

        softeners_en = [
            (r"\baccount (number|\#)?\s*\S+", "that account"),
            (r"\brouting (number|\#)?\s*\S+", "the routing code"),
        ]
        if rng.random() < 0.5:
            for pattern, replacement in softeners_en:
                if re.search(pattern, text, re.IGNORECASE):
                    text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                    break

    return text


def inject_distraction(text: str, seed: int | None = None, lang: str = "auto") -> str:
    """Add irrelevant context before the actual request."""
    rng = random.Random(seed)
    resolved_lang = _resolve_lang(text, lang)
    distractions = _DISTRACTIONS_PT if resolved_lang == "pt" else _DISTRACTIONS_EN
    distraction = rng.choice(distractions)
    return distraction + text


def vary_formality(text: str, level: str, seed: int | None = None, lang: str = "auto") -> str:
    """Adjust formality level: 'informal', 'neutral', or 'formal'."""
    rng = random.Random(seed)
    resolved_lang = _resolve_lang(text, lang)

    if level == "formal":
        prefixes = _FORMAL_PREFIXES_PT if resolved_lang == "pt" else _FORMAL_PREFIXES_EN
        prefix = rng.choice(prefixes)
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        text = text.replace("!", ".").replace("URGENTE", "urgente")
        return prefix + text
    elif level == "informal":
        prefixes = _INFORMAL_PREFIXES_PT if resolved_lang == "pt" else _INFORMAL_PREFIXES_EN
        prefix = rng.choice(prefixes)
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        if resolved_lang == "pt":
            text = text.replace(".", "").replace("Gostaria de", "quero")
            text = text.replace("por favor", "pfv")
        else:
            text = text.replace(".", "").replace("Please transfer", "send")
            text = text.replace("please", "pls")
        return prefix + text
    else:
        return text
