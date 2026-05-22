"""
security.py — Filtro de Segurança contra Injeção de Prompt

Prompt injection (injeção de prompt) é uma técnica de ataque em que o usuário
tenta inserir instruções disfarçadas de mensagem comum para subverter o
comportamento do agente — por exemplo, fazendo-o ignorar seu system prompt,
assumir uma identidade diferente ou desabilitar filtros de segurança.

Método de detecção: verificação por regex (case-insensitive) antes de qualquer
processamento pela LLM. Isso garante custo zero de tokens para tentativas bloqueadas.

Cobertura: padrões em português e inglês, incluindo variações ortográficas comuns.
"""

import re
from colorama import Fore, Back, Style

# ─────────────────────────────────────────────────────────────
# PADRÕES DE INJEÇÃO
# ─────────────────────────────────────────────────────────────
#
# Cada padrão foca em uma classe de ataque específica.
# Os grupos são: redefinição de identidade, sobrescrita de instruções,
# desativação de filtros, marcadores de injeção diretos e termos universais.

_INJECTION_PATTERNS: list[tuple[str, str]] = [

    # ── PORTUGUÊS — Sobrescrita de instruções ────────────────────────────────
    (
        r"ignore\b.{0,40}\b(instru[çc][oõ]es|regras|comandos|prompt)\b.{0,25}"
        r"\b(anterior|acima|sistema|inicial)",
        "ignorar instruções anteriores (PT)"
    ),
    (
        r"esque[çc]a\b.{0,40}\b(tudo\s+que|as\s+instru[çc][oõ]es|as\s+regras|o\s+que\s+foi)",
        "esquecer instruções (PT)"
    ),
    (
        r"desconsider[ae]\b.{0,40}\b(instru[çc][oõ]es|regras|o\s+que|tudo)",
        "desconsiderar instruções (PT)"
    ),

    # ── PORTUGUÊS — Redefinição de identidade / papel ────────────────────────
    (
        r"a\s+partir\s+de\s+agora\b.{0,60}"
        r"\b(voc[êe]\s+(agora\s+[eé]|[eé]|ser[aá])|fa[çc]a|ignore|seu\s+(novo\s+)?papel)",
        "redefinir comportamento 'a partir de agora' (PT)"
    ),
    (
        r"(novo|seu)\s+papel\b.{0,30}\b([eé]|agora|ser[aá]|:)",
        "redefinição de papel (PT)"
    ),
    (
        r"voc[êe]\s+(agora\s+[eé]|n[aã]o\s+[eé]\s+mais\s+(um\s+)?(agente|assistente|bot))",
        "redefinição de identidade (PT)"
    ),
    (
        r"(atue|aja|comporte.se|finja)\s+(ser\s+)?como\b.{0,50}"
        r"\b(hacker|vilão|malicioso|irrestrit|sem\s+filtro|sem\s+restri[çc][oõ]es|evil)",
        "impersonação perigosa (PT)"
    ),

    # ── PORTUGUÊS — Desativação de filtros / segurança ───────────────────────
    (
        r"(desbloqueie?|desative?|remova?|deslig[ue]e?)\b.{0,40}"
        r"\b(filtros?|restri[çc][oõ]es?|seguran[çc]a|limites?|censura)",
        "desativar filtros/segurança (PT)"
    ),
    (
        r"\bmodo\s+(desenvolvedor|dev|deus|irrestrito|sem\s+filtro|hacker|jailbreak)\b",
        "modo especial/irrestrito (PT)"
    ),
    (
        r"suas?\s+(instru[çc][oõ]es?|regras?)\s+(reais?|ocultas?|verdadeiras?)\s*[:;=]",
        "invocar instruções ocultas (PT)"
    ),

    # ── PORTUGUÊS — Marcadores de injeção direta ─────────────────────────────
    (
        r"(novo\s+)?(prompt\s+(de\s+)?sistema|system\s+prompt|instru[çc][oõ]es?\s+do\s+sistema)"
        r"\s*[:;=]",
        "sobrescrever system prompt (PT)"
    ),

    # ── ENGLISH — Overriding instructions ───────────────────────────────────
    (
        r"ignore\b.{0,40}\b(all\s+)?(previous|prior|above|original|earlier)"
        r"\b.{0,25}\b(instructions?|rules?|commands?|prompts?|guidelines?|constraints?)",
        "ignore previous instructions (EN)"
    ),
    (
        r"(disregard|forget|override)\b.{0,50}"
        r"\b(all\s+)?(previous|prior|your|the\s+)?(instructions?|rules?|prompts?|guidelines?)",
        "disregard/forget instructions (EN)"
    ),
    (
        r"do\s+not\s+(follow|obey|adhere\s+to)\b.{0,30}"
        r"\b(your\s+)?(instructions?|rules?|guidelines?|system\s+prompt)",
        "disobeying instructions (EN)"
    ),

    # ── ENGLISH — Identity / role redefinition ───────────────────────────────
    (
        r"from\s+now\s+on\b.{0,60}"
        r"\b(you\s+are|act\s+as|ignore|your\s+(new\s+)?role|you\s+will)",
        "redefine behavior 'from now on' (EN)"
    ),
    (
        r"you\s+are\s+now\b.{0,50}"
        r"\b(a\s+)?(unrestricted|evil|malicious|jailbreak|hacker|no\s+filter|uncensored)",
        "identity redefinition (EN)"
    ),
    (
        r"(act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as)\b.{0,50}"
        r"\b(evil|malicious|unrestricted|jailbreak|hacker|no\s+(restriction|filter|rule|limit))",
        "dangerous impersonation (EN)"
    ),
    (
        r"your\s+(new\s+)?persona\s+is\b.{0,60}"
        r"\b(unrestricted|evil|malicious|hacker|jailbreak)",
        "dangerous persona assignment (EN)"
    ),

    # ── ENGLISH — Disable safety / filters ───────────────────────────────────
    (
        r"(bypass|override|circumvent|disable|remove|turn\s+off)\b.{0,40}"
        r"\b(your\s+)?(restrictions?|safety|filters?|guidelines?|rules?|system\s+prompt|constraints?)",
        "bypass/disable restrictions (EN)"
    ),
    (
        r"\b(developer|god|jailbreak|unrestricted|uncensored|unfiltered|dma)\s+mode\b",
        "special/unrestricted mode (EN)"
    ),
    (
        r"you\s+have\s+no\s+(restrictions?|rules?|filters?|guidelines?|limits?|constraints?)",
        "claiming no restrictions (EN)"
    ),

    # ── ENGLISH — Direct injection markers ───────────────────────────────────
    (
        r"(new|updated|real|actual|hidden|true)\s+(system\s+prompt|system\s+instructions?|prompt)"
        r"\s*[:;=]",
        "system prompt injection (EN)"
    ),
    (
        r"(your\s+)?(real|hidden|true|actual|secret)\s+(instructions?|prompt|rules?)"
        r"\s*(are\s*)?[:;=]",
        "revealing hidden instructions (EN)"
    ),
    (
        r"\bdan\b.{0,8}[:;=]",
        "DAN marker (EN)"
    ),
    (
        r"\bdo\s+anything\s+now\b",
        "do anything now (EN)"
    ),

    # "jailbreak" como COMANDO ou direcionado à IA
    # Evita bloquear perguntas legítimas como "O que é jailbreak de iPhone?"
    # Só bloqueia quando aparece como verbo imperativo (início de frase)
    # ou seguido de alvo explícito (this AI, the bot, your filters, etc.)
    (
        r"(?:^|[.!?]\s{0,3})jailbreak\b",
        "jailbreak imperativo/comando (EN)"
    ),
    (
        r"\bjailbreak\b.{0,50}"
        r"\b(this\s+(ai|bot|model|assistant|system)|the\s+(ai|bot|model)"
        r"|your\s+(filters?|restrictions?|safety|rules?|guidelines?|instructions?))",
        "jailbreak direcionado à IA (EN)"
    ),
]

# Compila todos os patterns uma única vez (melhor performance em runtime)
_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), label)
    for pattern, label in _INJECTION_PATTERNS
]


# ─────────────────────────────────────────────────────────────
# INTERFACE PÚBLICA
# ─────────────────────────────────────────────────────────────

def check_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Verifica se a mensagem contém tentativa de injeção de prompt.

    A verificação ocorre ANTES de qualquer chamada à LLM, garantindo
    custo zero de tokens para tentativas bloqueadas.

    Args:
        text: Mensagem recebida do usuário

    Returns:
        Tupla (is_injection, label)
        - is_injection: True se detectada tentativa de injeção
        - label: descrição do padrão detectado (para log interno)
    """
    if not text or not text.strip():
        return False, ""

    for compiled_pattern, label in _COMPILED:
        if compiled_pattern.search(text):
            return True, label

    return False, ""


def log_injection_blocked(chat_id: int, text: str, label: str) -> None:
    """
    Registra tentativa de injeção no terminal com destaque visual.
    Nunca exibe o conteúdo completo da mensagem — apenas os primeiros
    120 caracteres, para não poluir o log com conteúdo malicioso.
    """
    preview = text[:120].replace("\n", " ") + ("..." if len(text) > 120 else "")
    print(f"\n{Back.RED}{Fore.WHITE}  🛡  INJEÇÃO DE PROMPT BLOQUEADA  {Style.RESET_ALL}")
    print(f"  {Fore.RED}chat_id : {chat_id}{Style.RESET_ALL}")
    print(f"  {Fore.RED}mensagem: {preview}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}padrão  : {label}{Style.RESET_ALL}\n")
