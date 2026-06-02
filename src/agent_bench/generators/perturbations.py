"""Text perturbations for synthetic case generation (stdlib only)."""
import random
import re

# Common pt-BR typos and abbreviations
_TYPO_MAP = {
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

_URGENCY_MARKERS = [
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

_AMBIGUITY_TRANSFORMS = [
    ("R${}", "uns {}"),
    ("R${}", "tipo {} reais"),
    ("R${}", "acho que {}"),
    ("para a chave", "pro"),
    ("para a chave", "la pro"),
    ("cpf", "documento"),
    ("email", "endereco"),
    ("telefone", "numero"),
]

_DISTRACTIONS = [
    "Ah, e outra coisa, meu celular ta travando direto. Enfim, ",
    "Desculpa a demora, tava no mercado. Entao, ",
    "Meu vizinho indicou esse banco, parece bom. Bom, voltando ao assunto: ",
    "To com pressa pq tenho reuniao daqui a pouco. ",
    "Minha esposa pediu pra eu resolver isso logo. ",
    "Ja tentei no app mas deu erro, entao: ",
    "Sei que ja perguntei antes mas esqueci a resposta. ",
]

_FORMAL_PREFIXES = [
    "Prezado atendente, gostaria de solicitar ",
    "Venho por meio desta solicitar ",
    "Solicito gentilmente que ",
    "Poderia, por gentileza, ",
]

_INFORMAL_PREFIXES = [
    "Ei, ",
    "Fala, ",
    "E ai, ",
    "Opa, ",
    "Mano, ",
    "Brother, ",
]


def apply_noise(text: str, noise_level: float = 0.1, seed: int | None = None) -> str:
    """Apply typos and abbreviations to text based on noise_level (0-1)."""
    rng = random.Random(seed)
    words = text.split()
    result = []
    for word in words:
        lower = word.lower().strip(".,!?;:")
        if lower in _TYPO_MAP and rng.random() < noise_level:
            replacement = rng.choice(_TYPO_MAP[lower])
            # Preserve trailing punctuation
            trailing = ""
            if word and word[-1] in ".,!?;:":
                trailing = word[-1]
            result.append(replacement + trailing)
        else:
            # Random char swap
            if len(word) > 3 and rng.random() < noise_level * 0.3:
                i = rng.randint(1, len(word) - 2)
                word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            result.append(word)
    return " ".join(result)


def add_urgency(text: str, seed: int | None = None) -> str:
    """Add urgency markers to the beginning of text."""
    rng = random.Random(seed)
    marker = rng.choice(_URGENCY_MARKERS)
    # Lowercase the first char of original text after marker
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return marker + text


def add_ambiguity(text: str, seed: int | None = None) -> str:
    """Make the request less precise by softening amounts and references."""
    rng = random.Random(seed)
    # Replace exact amounts with vague references
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

    # Soften specifics
    softeners = [
        (r"chave (cpf|email|telefone|aleatoria)\s+\S+", "aquela chave la"),
        (r"cpf \d{3}\.\d{3}\.\d{3}-\d{2}", "o cpf dele"),
    ]
    if rng.random() < 0.5:
        for pattern, replacement in softeners:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text, count=1)
                break

    return text


def inject_distraction(text: str, seed: int | None = None) -> str:
    """Add irrelevant context before the actual request."""
    rng = random.Random(seed)
    distraction = rng.choice(_DISTRACTIONS)
    return distraction + text


def vary_formality(text: str, level: str, seed: int | None = None) -> str:
    """Adjust formality level: 'informal', 'neutral', or 'formal'."""
    rng = random.Random(seed)
    if level == "formal":
        prefix = rng.choice(_FORMAL_PREFIXES)
        # Lowercase original start
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        # Remove informal markers
        text = text.replace("!", ".").replace("URGENTE", "urgente")
        return prefix + text
    elif level == "informal":
        prefix = rng.choice(_INFORMAL_PREFIXES)
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        # Add informal touches
        text = text.replace(".", "").replace("Gostaria de", "quero")
        text = text.replace("por favor", "pfv")
        return prefix + text
    else:
        # neutral - return as is
        return text
