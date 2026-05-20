"""
agent.py — Núcleo do Agente Autônomo (Agentic Loop)

Implementa o loop agentic em quatro estágios (inspirado no padrão ReAct, sem framework):

  1. Thought   (Pensamento)   — analisa o que precisa fazer
  2. Action    (Ação)         — chama uma ferramenta externa
  3. Observation (Observação) — recebe e processa o resultado
  4. Final Answer (Resposta)  — formula a resposta para o usuário

LLM utilizado: Google Gemini (via google-genai SDK)
  - Suporta function calling nativo
  - Modelo padrão: gemini-2.5-flash-lite

O que o usuário do Telegram VÊ: apenas a resposta final
O que aparece no TERMINAL: todo o raciocínio + ações + observações
"""

import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from colorama import init, Fore, Style, Back

from memory import ConversationMemory
from tools.web_search import web_search
from tools.python_exec import execute_python

# Inicializa colorama para suporte a cores no Windows
init(autoreset=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO CLIENTE GEMINI
# ─────────────────────────────────────────────────────────────

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Modelos descontinuados (404 para novos usuários) — fallback automático
_DEPRECATED_MODELS = {
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
}
# flash-lite tem function calling mais estável que gemini-2.5-flash com prompt longo
_FALLBACK_MODEL = "gemini-2.5-flash-lite"
_DEFAULT_MODEL = "gemini-2.5-flash-lite"


def get_gemini_model() -> str:
    """Retorna o modelo configurado, com fallback se estiver descontinuado."""
    model = os.getenv("GEMINI_MODEL", _DEFAULT_MODEL).strip()
    # flash completo falha com MALFORMED_FUNCTION_CALL em prompts longos
    if model == "gemini-2.5-flash":
        return _DEFAULT_MODEL
    if model in _DEPRECATED_MODELS:
        log_error(
            f"Modelo '{model}' descontinuado. Usando '{_FALLBACK_MODEL}'. "
            f"Atualize GEMINI_MODEL no .env."
        )
        return _FALLBACK_MODEL
    return model


def _get_candidate_parts(candidate) -> list:
    """Extrai partes da resposta do Gemini de forma segura (parts pode ser None)."""
    if not candidate or not getattr(candidate, "content", None):
        return []
    parts = getattr(candidate.content, "parts", None)
    if parts is None:
        return []
    try:
        return list(parts)
    except TypeError:
        return []


def _function_call_args_to_dict(function_call) -> dict:
    """Converte argumentos de function call do Gemini para dict Python."""
    if not function_call:
        return {}
    args = getattr(function_call, "args", None)
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    try:
        return dict(args)
    except (TypeError, ValueError):
        return {}


def _response_text_fallback(response) -> str:
    """
    Tenta extrair texto da resposta navegando manualmente pela estrutura,
    SEM usar response.text (que pode explodir com TypeError quando parts=None).
    """
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""
        candidate = candidates[0] if candidates else None
        if not candidate:
            return ""
        content = getattr(candidate, "content", None)
        if not content:
            return ""
        raw_parts = getattr(content, "parts", None)
        if raw_parts is None:
            return ""
        # Itera com segurança — proto-plus pode retornar iteráveis não-lista
        texts = []
        try:
            for p in raw_parts:
                t = getattr(p, "text", None)
                if t:
                    texts.append(str(t))
        except TypeError:
            pass
        return "\n".join(texts).strip()
    except Exception:
        return ""


def _get_finish_reason(candidate) -> str:
    """Retorna o motivo de término da geração (SAFETY, STOP, etc.)."""
    reason = getattr(candidate, "finish_reason", None)
    if reason is None:
        return ""
    return str(reason).upper()


def _empty_response_message(finish_reason: str) -> str:
    """Mensagem amigável quando o modelo não devolve texto nem tool call."""
    if "SAFETY" in finish_reason or "BLOCK" in finish_reason:
        return (
            "Não consegui analisar esse assunto: o modelo aplicou restrições de segurança. "
            "Tente reformular a pergunta de forma mais neutra ou objetiva."
        )
    if "RECITATION" in finish_reason:
        return (
            "Não consegui concluir a resposta por restrição de citação do modelo. "
            "Tente pedir um resumo geral em vez de conteúdo específico."
        )
    if "MAX_TOKENS" in finish_reason:
        return (
            "A resposta ficou grande demais e foi cortada. "
            "Tente dividir a pergunta em partes menores."
        )
    if "MALFORMED" in finish_reason:
        return (
            "O modelo gerou uma chamada de ferramenta inválida. "
            "Tente reformular a pergunta de forma mais direta e envie novamente."
        )
    return (
        "A IA não gerou texto utilizável nesta tentativa (resposta vazia). "
        "Envie a pergunta de novo ou use /clear para reiniciar a conversa."
    )


def _extract_text_and_calls(parts_iter: list) -> tuple[list[str], list, list[str]]:
    """
    Separa texto visível, function calls e pensamentos internos do modelo.
    O Gemini 2.5 pode colocar raciocínio em part.thought em vez de part.text.
    """
    text_parts: list[str] = []
    function_calls: list = []
    thought_parts: list[str] = []

    for part in parts_iter:
        thought = getattr(part, "thought", None)
        if thought:
            thought_parts.append(str(thought))

        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)

        if hasattr(part, "function_call") and part.function_call:
            function_calls.append(part.function_call)

    return text_parts, function_calls, thought_parts

# ─────────────────────────────────────────────────────────────
# SCHEMA DAS FERRAMENTAS (formato Gemini FunctionDeclaration)
# ─────────────────────────────────────────────────────────────
#
# O Gemini usa esses schemas para saber quando e como chamar
# cada ferramenta. O campo "description" é crucial: o modelo
# usa-o para decidir qual ferramenta usar em cada situação.

TOOLS_CONFIG = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Busca informações em tempo real na internet. "
                "OBRIGATÓRIO para: clima atual, cotações, notícias recentes, "
                "resultados e eventos esportivos, fatos de 2025 ou 2026, preços, lançamentos. "
                "Use sempre que o usuário perguntar sobre algo atual, recente ou 'hoje'. "
                "Se a primeira busca retornar resultado incompleto, faça uma segunda busca "
                "com query mais específica antes de concluir que a informação não existe."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "A consulta de busca. Seja específico. "
                            "Exemplo: 'clima São Paulo hoje temperatura' "
                            "em vez de apenas 'tempo'."
                        )
                    )
                },
                required=["query"]
            )
        ),
        types.FunctionDeclaration(
            name="execute_python",
            description=(
                "Executa código Python em sandbox seguro para cálculos e análises. "
                "Use para: cálculos matemáticos, estatísticas (média, desvio padrão, etc.), "
                "operações com listas, análise de dados numéricos. "
                "SEMPRE use print() para exibir resultados. "
                "Módulos pré-carregados: math, statistics, random (não precisam de import)."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Código Python a executar. Use print() para exibir resultados. "
                            "Módulos disponíveis sem import: math, statistics, random. "
                            "Exemplo: print(statistics.mean([10, 20, 30]))"
                        )
                    )
                },
                required=["code"]
            )
        )
    ]
)


# ─────────────────────────────────────────────────────────────
# FUNÇÕES DE LOG COLORIDO
# ─────────────────────────────────────────────────────────────

def _line(char: str = "─", width: int = 62) -> str:
    return f"{Fore.WHITE}{char * width}{Style.RESET_ALL}"


def log_user_message(chat_id: int, message: str) -> None:
    """Exibe a mensagem do usuário no terminal."""
    print(f"\n{_line('═')}")
    print(f"{Fore.WHITE}[👤 USUÁRIO]  chat_id={chat_id}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{message}{Style.RESET_ALL}")
    print(_line())


def log_thought(thought: str, iteration: int) -> None:
    """
    Exibe o bloco de pensamento do agente.
    VISÍVEL APENAS NO TERMINAL — não é enviado ao usuário no Telegram.
    """
    print(f"\n{Back.YELLOW}{Fore.BLACK}  💭 THOUGHT  (iteração {iteration})  {Style.RESET_ALL}")
    for line in (thought or "").strip().splitlines():
        print(f"  {Fore.YELLOW}{line}{Style.RESET_ALL}")


def log_action(tool_name: str, tool_args: dict) -> None:
    """Exibe a ação (chamada de ferramenta) que o agente vai executar."""
    print(f"\n{Back.MAGENTA}{Fore.WHITE}  ⚡ ACTION  {Style.RESET_ALL}")
    print(f"  {Fore.MAGENTA}Ferramenta:{Style.RESET_ALL} {Fore.WHITE}{tool_name}{Style.RESET_ALL}")
    args_formatted = json.dumps(tool_args, ensure_ascii=False, indent=4)
    for line in args_formatted.splitlines():
        print(f"  {Fore.CYAN}{line}{Style.RESET_ALL}")


def log_observation(result: str) -> None:
    """
    Exibe o resultado retornado pela ferramenta.
    Usa fundo vermelho se o resultado for um erro de ferramenta,
    verde se for sucesso. Trunca se muito longo.
    """
    is_error = result.lower().startswith("erro") or result.lower().startswith("error")

    if is_error:
        print(f"\n{Back.RED}{Fore.WHITE}  👁  OBSERVATION (ERRO DE FERRAMENTA)  {Style.RESET_ALL}")
        color = Fore.RED
    else:
        print(f"\n{Back.GREEN}{Fore.BLACK}  👁  OBSERVATION  {Style.RESET_ALL}")
        color = Fore.GREEN

    if len(result) > 700:
        print(f"  {color}{result[:700]}{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}[... {len(result) - 700} caracteres omitidos ...]{Style.RESET_ALL}")
    else:
        for line in result.splitlines():
            print(f"  {color}{line}{Style.RESET_ALL}")


def log_final_answer(answer: str) -> None:
    """Exibe a resposta final que será enviada ao usuário."""
    print(f"\n{Back.BLUE}{Fore.WHITE}  ✅ FINAL ANSWER  {Style.RESET_ALL}")
    for line in answer.strip().splitlines():
        print(f"  {Fore.CYAN}{line}{Style.RESET_ALL}")


def log_tokens(usage_metadata, total_accumulated: int = 0) -> None:
    """
    Exibe o consumo de tokens desta chamada.

    Critério de avaliação — Eficiência de Tokens:
    Mostra separadamente prompt vs completion para que o professor possa
    verificar que o prompt (contexto + histórico) é controlado pela
    sliding window e não cresce indefinidamente.
    """
    if not usage_metadata:
        return
    prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
    completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
    current = getattr(usage_metadata, "total_token_count", 0) or 0
    print(
        f"\n  {Fore.CYAN}📊 Tokens — prompt={prompt} | "
        f"completion={completion} | "
        f"nesta chamada={current} | "
        f"total do turno={total_accumulated}{Style.RESET_ALL}"
    )


def log_memory_stats(stats: dict) -> None:
    """
    Exibe estatísticas da janela de memória no início de cada turno.

    Critério de avaliação — Eficiência de Tokens:
    Demonstra que o agente NÃO reenvia o histórico inteiro à API —
    apenas a sliding window (últimas N mensagens).
    """
    total = stats.get("total", 0)
    window = stats.get("window", 0)
    omitted = stats.get("omitted", 0)
    max_w = stats.get("max_window", 10)

    if omitted > 0:
        efficiency_note = (
            f"{Fore.YELLOW}  (⚡ {omitted} mensagem(ns) omitida(s) para economizar tokens)"
        )
    else:
        efficiency_note = f"{Fore.GREEN}  (janela dentro do limite)"

    print(
        f"  {Fore.CYAN}🧠 Memória: "
        f"{window}/{max_w} mensagens no contexto | "
        f"{total} no histórico total{Style.RESET_ALL}"
    )
    print(f"{efficiency_note}{Style.RESET_ALL}")


def log_debug(message: str) -> None:
    """Log de debug: só aparece quando DEBUG=true no .env."""
    if DEBUG:
        print(f"  {Fore.WHITE}[DEBUG] {message}{Style.RESET_ALL}")


def log_error(message: str) -> None:
    """Exibe mensagem de erro no terminal."""
    print(f"\n{Back.RED}{Fore.WHITE}  ❌ ERRO  {Style.RESET_ALL}")
    print(f"  {Fore.RED}{message}{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────
# CONVERSÃO DE FORMATO: memória → conteúdo Gemini
# ─────────────────────────────────────────────────────────────

def build_gemini_contents(memory_messages: list[dict]) -> list[types.Content]:
    """
    Converte o formato interno da memória para o formato de conteúdo do Gemini.

    Memória interna usa roles "user" e "assistant" (padrão OpenAI/geral).
    Gemini usa "user" e "model".

    Args:
        memory_messages: Lista de dicts {"role": ..., "content": ...}

    Returns:
        Lista de types.Content para a API Gemini
    """
    if not memory_messages:
        return []

    contents = []
    for msg in memory_messages:
        if not msg:
            continue
        # Gemini usa "model" onde outros usam "assistant"
        role = "user" if msg.get("role") == "user" else "model"
        text = msg.get("content") or ""
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)]
            )
        )
    return contents


# ─────────────────────────────────────────────────────────────
# PARSING DE CONTEÚDO
# ─────────────────────────────────────────────────────────────

def _strip_thought_from_text(content: str) -> tuple[str, str]:
    """
    Remove blocos de pensamento do texto e retorna (thought_extraído, resto).

    Aceita formatos que o modelo às vezes gera por engano:
      [Thought]...[/Thought]  (correto)
      [Thought ... ]          (fecha com ] simples)
    """
    if not content:
        return "", ""

    # Ordem: padrão correto → variantes comuns do modelo (às vezes omite ] após Thought)
    patterns = [
        (r'\[Thought\](.*?)\[/Thought\]', r'\[Thought\].*?\[/Thought\]'),
        (r'\[Thought\]?\s*([\s\S]*?)\n\s*\]', r'\[Thought\]?\s*[\s\S]*?\n\s*\]'),
        (r'\[Thought\](.*?)\]', r'\[Thought\].*?\]'),
        (r'\[Thought\]([\s\S]*?)\]', r'\[Thought\][\s\S]*?\]'),
    ]

    for extract_pat, remove_pat in patterns:
        match = re.search(extract_pat, content, re.DOTALL | re.IGNORECASE)
        if match:
            thought = match.group(1).strip()
            remaining = re.sub(remove_pat, '', content, count=1,
                               flags=re.DOTALL | re.IGNORECASE).strip()
            return thought, remaining

    return "", content


def extract_thought(content: str) -> tuple[str, str]:
    """
    Extrai o bloco [Thought] do texto de resposta para exibir no terminal.

    Returns:
        Tupla (thought_text, content_without_thought)
    """
    return _strip_thought_from_text(content)


def clean_for_user(content: str) -> str:
    """
    Remove todos os marcadores internos antes de enviar ao usuário.
    O usuário deve ver apenas texto limpo e natural.
    """
    if not content:
        return "Desculpe, não consegui gerar uma resposta."

    # Remove blocos [Thought] (vários formatos)
    _, content = _strip_thought_from_text(content)

    # Remove blocos residuais (inclui [Thought sem ] de fechamento na tag de abertura)
    content = re.sub(
        r'\[Thought\]?\s*[\s\S]*?(?:\[/Thought\]|\n\s*\])',
        '',
        content,
        count=1,
        flags=re.IGNORECASE,
    )
    content = re.sub(r'\[Thought\][\s\S]*', '', content, count=1, flags=re.IGNORECASE)
    content = re.sub(r'\[/Thought\]', '', content, flags=re.IGNORECASE)

    # Remove marcadores avulsos de protocolo ReAct
    content = re.sub(
        r'\[(Thought|Action|Observation|Final Answer)\]',
        '',
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r'\n{3,}', '\n\n', content)

    result = content.strip()
    return result if result else "Desculpe, não consegui gerar uma resposta."


# ─────────────────────────────────────────────────────────────
# EXECUÇÃO DE FERRAMENTAS
# ─────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Despacha a chamada para a ferramenta correta e trata erros.

    Args:
        tool_name: "web_search" ou "execute_python"
        tool_args: Argumentos (dict)

    Returns:
        Resultado da ferramenta como string
    """
    try:
        if tool_name == "web_search":
            query = tool_args.get("query", "").strip()
            if not query:
                return "Erro: parâmetro 'query' vazio."
            return web_search(query)

        elif tool_name == "execute_python":
            code = tool_args.get("code", "").strip()
            if not code:
                return "Erro: parâmetro 'code' vazio."
            return execute_python(code)

        else:
            return (
                f"Ferramenta '{tool_name}' não encontrada. "
                "Ferramentas disponíveis: web_search, execute_python."
            )

    except Exception as e:
        error_msg = f"Erro ao executar '{tool_name}': {type(e).__name__}: {e}"
        log_error(error_msg)
        return f"A ferramenta '{tool_name}' encontrou um erro e não pôde ser executada."


# ─────────────────────────────────────────────────────────────
# CARREGAMENTO DO PROMPT DE SISTEMA
# ─────────────────────────────────────────────────────────────

def load_system_prompt(compact: bool = False) -> str:
    """
    Carrega o prompt de sistema e injeta a data atual.

    Prompts longos (>~4k chars com cabeçalho) causam MALFORMED_FUNCTION_CALL no Gemini.
    O modo compact=True é usado em retentativas automáticas.
    """
    if compact:
        base_prompt = (
            "Agente Telegram. Use [Thought] em texto antes de ferramentas. "
            "execute_python: parâmetro code = só Python com print(). "
            "web_search: parâmetro query = só texto da busca. "
            "Não invente dados atuais; busque na web se necessário."
        )
    else:
        path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                base_prompt = f.read().strip()
        except FileNotFoundError:
            log_error("system_prompt.txt não encontrado. Usando prompt de fallback.")
            base_prompt = (
                "Você é um assistente útil, honesto e preciso. "
                "Nunca invente informações. Use ferramentas quando precisar de dados atuais."
            )

    today = datetime.now().strftime("%d/%m/%Y")
    date_header = (
        f"DATA ATUAL: {today}. Treinamento até 2024 — para fatos de 2025/2026 use web_search. "
        "Não assuma que eventos futuros pelo treinamento ainda não ocorreram.\n\n"
    )

    return date_header + base_prompt


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL DO AGENTE
# ─────────────────────────────────────────────────────────────

def run_agent(user_message: str, chat_id: int, memory: ConversationMemory) -> str:
    """
    Executa o Agentic Loop completo para processar uma mensagem.

    Fluxo com Gemini (function calling):
      1. Monta contents = histórico da memória (formato Gemini)
      2. Chama client.models.generate_content com tools configurados
      3. Se resposta contém function_call → executa ferramenta → adiciona resultado
      4. Repete até obter resposta só com texto
      5. Remove marcadores internos → envia resposta limpa ao Telegram

    Separação de memória:
    - Apenas pares user/assistant finais são salvos na memória persistente
    - Tool calls intermediários ficam apenas no contexto local (economiza tokens)

    Args:
        user_message: Texto enviado pelo usuário no Telegram
        chat_id: ID do chat (identifica o usuário na memória)
        memory: Instância compartilhada da memória conversacional

    Returns:
        Resposta final limpa para enviar ao usuário
    """
    # ── ESCUDO EXTERNO ────────────────────────────────────────
    # Garante que nenhuma exceção inesperada propague para o event loop
    # do Telegram, o que causaria crash do bot.
    try:
        return _run_agent_loop(user_message, chat_id, memory)
    except Exception as top_exc:
        log_error(f"Exceção não capturada no agente: {type(top_exc).__name__}: {top_exc}")
        fallback = (
            "Ocorreu um erro interno inesperado. Por favor, tente novamente. "
            "Se o problema persistir, use /clear para reiniciar a conversa."
        )
        try:
            memory.add_message(chat_id, "assistant", fallback)
        except Exception:
            pass
        return fallback


def _run_agent_loop(user_message: str, chat_id: int, memory: ConversationMemory) -> str:
    """Implementação interna do loop — chamada pelo wrapper run_agent."""
    log_user_message(chat_id, user_message)

    system_prompt = load_system_prompt()
    malformed_retries = 0

    # Salva a mensagem do usuário na memória persistente ANTES da chamada
    memory.add_message(chat_id, "user", user_message)

    # ── EFICIÊNCIA DE TOKENS: exibe estatísticas da sliding window ──
    # Demonstra que o histórico completo existe, mas só as últimas N
    # mensagens são enviadas à API — critério de avaliação de tokens.
    stats = memory.get_stats(chat_id)
    log_memory_stats(stats)

    # Converte histórico da memória para o formato do Gemini
    # O histórico já inclui a mensagem atual (recém adicionada)
    contents = build_gemini_contents(memory.get_messages(chat_id))

    max_iterations = 6
    iteration = 0
    total_tokens = 0
    empty_response_retries = 0

    # ═══════════════════════════════════════════════════════════
    # AGENTIC LOOP
    # ═══════════════════════════════════════════════════════════
    while iteration < max_iterations:
        iteration += 1
        log_debug(f"Iteração {iteration}/{max_iterations}")

        try:
            # ── CHAMADA À API GEMINI (com retentativa para erros transitórios) ──
            response = None
            _api_retries = 2  # tenta até 3 vezes no total (1 + 2 retries)
            for _attempt in range(_api_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=get_gemini_model(),
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            tools=[TOOLS_CONFIG],
                            temperature=0.3,
                            max_output_tokens=3000,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        )
                    )
                    break  # sucesso — sai do loop de retentativa

                except Exception as api_error:
                    err_str = str(api_error).lower()
                    is_rate_limit = "429" in err_str or "quota" in err_str or "rate" in err_str
                    is_transient = "503" in err_str or "timeout" in err_str or "unavailable" in err_str

                    if (is_rate_limit or is_transient) and _attempt < _api_retries:
                        wait = 3 * (_attempt + 1)  # 3s, 6s
                        log_error(
                            f"Erro transitório na API ({type(api_error).__name__}). "
                            f"Retentativa {_attempt + 1}/{_api_retries} em {wait}s..."
                        )
                        time.sleep(wait)
                        continue

                    # Erro definitivo ou esgotou retentativas
                    error_detail = str(api_error)
                    if "401" in error_detail or "api_key" in error_detail.lower():
                        user_msg = "Erro de autenticação: verifique a GEMINI_API_KEY no .env."
                    elif "429" in error_detail or "quota" in error_detail.lower():
                        user_msg = "Limite de requisições atingido. Aguarde alguns segundos e tente novamente."
                    elif "503" in error_detail or "unavailable" in error_detail.lower():
                        user_msg = "O serviço Gemini está temporariamente indisponível. Tente novamente em instantes."
                    else:
                        user_msg = "Não consegui me conectar ao serviço de IA. Tente novamente."

                    log_error(f"Falha definitiva na API Gemini: {api_error}")
                    memory.add_message(chat_id, "assistant", user_msg)
                    return user_msg

            if response is None:
                fallback = "Não obtive resposta da API após múltiplas tentativas. Tente novamente."
                memory.add_message(chat_id, "assistant", fallback)
                return fallback

            # Contagem de tokens
            if response.usage_metadata:
                current = getattr(response.usage_metadata, "total_token_count", 0) or 0
                total_tokens += current
                log_tokens(response.usage_metadata, total_tokens)

            # Verifica se há candidatos válidos na resposta
            # Protege contra None e listas que podem gerar TypeError ao indexar
            try:
                has_candidates = bool(response.candidates)
            except TypeError:
                has_candidates = False

            if not has_candidates:
                log_error("Gemini retornou resposta sem candidatos (possível filtro de conteúdo).")
                fallback = "Desculpe, não consegui processar esta mensagem. Tente reformular."
                memory.add_message(chat_id, "assistant", fallback)
                return fallback

            try:
                candidate = response.candidates[0]
            except (IndexError, TypeError):
                candidate = None

            if candidate is None:
                fallback = "Resposta inválida recebida. Por favor, tente novamente."
                memory.add_message(chat_id, "assistant", fallback)
                return fallback

            finish_reason = _get_finish_reason(candidate)
            parts = _get_candidate_parts(candidate)

            if not parts:
                # Último recurso: tenta extrair texto manualmente
                text_fallback = _response_text_fallback(response)
                if text_fallback:
                    log_thought("(texto via fallback direto)", iteration)
                    clean_answer = clean_for_user(text_fallback)
                    log_final_answer(clean_answer)
                    memory.add_message(chat_id, "assistant", clean_answer)
                    return clean_answer

                if "MALFORMED" in finish_reason and malformed_retries < 1:
                    malformed_retries += 1
                    system_prompt = load_system_prompt(compact=True)
                    log_debug(
                        "MALFORMED_FUNCTION_CALL — retentando com prompt compacto..."
                    )
                    continue

                is_structural_error = any(kw in finish_reason for kw in (
                    "SAFETY", "BLOCK", "RECITATION"
                ))
                if not is_structural_error and empty_response_retries < 2:
                    empty_response_retries += 1
                    log_debug(
                        f"Resposta vazia (finish_reason={finish_reason or 'n/a'}), "
                        f"retentativa {empty_response_retries}/2..."
                    )
                    continue

                log_error(
                    f"Gemini sem partes utilizáveis "
                    f"(finish_reason={finish_reason or 'n/a'})."
                )
                fallback = _empty_response_message(finish_reason)
                memory.add_message(chat_id, "assistant", fallback)
                return fallback

            # ── EXTRAI PARTES: TEXTO, THOUGHT e FUNCTION CALLS ──
            try:
                parts_iter = list(parts)
            except TypeError:
                parts_iter = []

            text_parts, function_calls, thought_parts = _extract_text_and_calls(parts_iter)

            if thought_parts:
                log_thought("\n".join(thought_parts), iteration)

            if text_parts:
                full_text = "\n".join(text_parts)
                thought, _ = extract_thought(full_text)
                if thought:
                    log_thought(thought, iteration)
                elif function_calls and full_text.strip() and not thought_parts:
                    preview = full_text[:300] + ("..." if len(full_text) > 300 else "")
                    log_thought(f"[raciocínio do modelo]\n{preview}", iteration)

            if function_calls:
                if not text_parts:
                    # Thought sintético: modelo não incluiu texto, geramos um
                    # informativo com o nome das ferramentas E seus argumentos
                    thought_lines = ["[thought gerado pelo agente — modelo omitiu texto]"]
                    for fc in function_calls:
                        fc_name = getattr(fc, "name", "ferramenta")
                        fc_args = _function_call_args_to_dict(fc)
                        # Mostra o argumento principal para contextualizar o log
                        arg_summary = ""
                        if fc_name == "web_search":
                            arg_summary = f': buscar "{fc_args.get("query", "?")}"'
                        elif fc_name == "execute_python":
                            code_preview = fc_args.get("code", "")[:80]
                            arg_summary = f": executar código → {code_preview!r}"
                        thought_lines.append(
                            f"→ Usando {fc_name}{arg_summary}"
                        )
                    log_thought("\n".join(thought_lines), iteration)

                if candidate.content and _get_candidate_parts(candidate):
                    contents.append(candidate.content)

                tool_response_parts = []
                for fc in function_calls:
                    tool_name = getattr(fc, "name", None) or "unknown_tool"
                    tool_args = _function_call_args_to_dict(fc)
                    log_action(tool_name, tool_args)
                    result = execute_tool(tool_name, tool_args)
                    log_observation(result)
                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    )

                contents.append(
                    types.Content(role="user", parts=tool_response_parts)
                )
                continue

            if not text_parts:
                if empty_response_retries < 2:
                    empty_response_retries += 1
                    log_debug(
                        f"Sem texto/fc (finish_reason={finish_reason or 'n/a'}), "
                        f"retentativa {empty_response_retries}/2..."
                    )
                    continue

                log_error(
                    f"Iteração sem texto nem ferramenta (finish_reason={finish_reason or 'n/a'})."
                )
                fallback = _empty_response_message(finish_reason)
                memory.add_message(chat_id, "assistant", fallback)
                return fallback

            final_text = "\n".join(text_parts)
            clean_answer = clean_for_user(final_text)
            log_final_answer(clean_answer)

            # ── SUMÁRIO DE EFICIÊNCIA DO TURNO ──────────────────
            final_stats = memory.get_stats(chat_id)
            print(
                f"\n  {Fore.CYAN}📊 Tokens do turno: {total_tokens} total "
                f"| Contexto enviado: {final_stats['window']}/{final_stats['total']} msgs "
                f"({final_stats['ratio']}% do histórico){Style.RESET_ALL}"
            )
            print(_line('═'))

            memory.add_message(chat_id, "assistant", clean_answer)
            return clean_answer

        except TypeError as e:
            log_error(f"Erro ao processar resposta do modelo: {e}")
            fallback = (
                "Encontrei um problema ao interpretar a resposta da IA. "
                "Tente enviar a pergunta de novo de forma mais direta."
            )
            memory.add_message(chat_id, "assistant", fallback)
            return fallback

    # ── LIMITE DE ITERAÇÕES ATINGIDO ────────────────────────
    log_error(f"Limite de {max_iterations} iterações atingido sem resposta final.")
    timeout_msg = (
        "Desculpe, não consegui concluir o processamento. "
        "Tente reformular a pergunta de forma mais específica."
    )
    memory.add_message(chat_id, "assistant", timeout_msg)
    return timeout_msg
