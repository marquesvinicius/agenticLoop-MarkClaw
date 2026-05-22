"""
main.py — Entry Point do Agente Autônomo Telegram

Responsabilidades deste módulo:
  - Inicializar o bot Telegram usando python-telegram-bot v20+
  - Registrar os handlers de comandos (/start, /clear, /help)
  - Receber mensagens dos usuários e delegar ao agente
  - Gerenciar a instância de memória compartilhada

Arquitetura assíncrona:
  python-telegram-bot v20 usa asyncio internamente.
  A função run_agent() é bloqueante (síncrona), então usamos
  asyncio.to_thread() para executá-la sem bloquear o event loop.
  Isso permite que o bot responda a múltiplos usuários simultaneamente.

Requisito: Python 3.9+ (para asyncio.to_thread)
"""

import asyncio
import atexit
import logging
import os
import sys

from dotenv import load_dotenv

# Carrega .env antes de importar agent (o cliente Gemini lê GEMINI_API_KEY na importação)
load_dotenv()

# Terminal Windows (cp1252) não imprime ═/emojis por padrão
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from colorama import init, Fore, Style

from telegram import Update
from telegram.error import Conflict
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from agent import run_agent, get_session_cost, reset_session_cost
from memory import ConversationMemory
from security import check_prompt_injection, log_injection_blocked

# ─────────────────────────────────────────────────────────────
# INICIALIZAÇÃO
# ─────────────────────────────────────────────────────────────

# Inicializa colorama para cores no Windows
init(autoreset=True)

# Suprime logs verbosos do python-telegram-bot e httpx
# (mantém apenas WARNING e acima para não poluir o terminal)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.WARNING
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────
# INSTÂNCIA DE MEMÓRIA COMPARTILHADA
# ─────────────────────────────────────────────────────────────
# Uma única instância é compartilhada entre todos os handlers.
# Cada chat_id tem seu próprio histórico dentro da instância.

memory = ConversationMemory(
    max_messages=int(os.getenv("MAX_HISTORY", "10")),
    save_to_file=True
)

# Impede duas instâncias do bot (causa Conflict + respostas duplicadas no Telegram)
_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bot.lock")


def _pid_is_running(pid: int) -> bool:
    """Verifica se um processo ainda está ativo (Windows/Linux)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_single_instance_lock() -> None:
    """Garante que apenas um main.py use o token do Telegram por vez."""
    if os.path.exists(_LOCK_FILE):
        try:
            with open(_LOCK_FILE, encoding="utf-8") as f:
                old_pid = int(f.read().strip())
            if _pid_is_running(old_pid):
                print(
                    f"{Fore.RED}[❌ ERRO] Outra instância do bot já está rodando "
                    f"(PID {old_pid}).{Style.RESET_ALL}"
                )
                print(
                    f"{Fore.WHITE}Encerre o outro terminal (Ctrl+C) ou execute:{Style.RESET_ALL}\n"
                    f"  Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
                    f"| Where-Object {{ $_.CommandLine -match 'main\\.py' }} "
                    f"| ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}\n"
                )
                sys.exit(1)
        except (ValueError, OSError):
            pass

    with open(_LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def release_single_instance_lock() -> None:
    try:
        if os.path.exists(_LOCK_FILE):
            with open(_LOCK_FILE, encoding="utf-8") as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(_LOCK_FILE)
    except OSError:
        pass


atexit.register(release_single_instance_lock)


# ─────────────────────────────────────────────────────────────
# HANDLERS DE COMANDOS
# ─────────────────────────────────────────────────────────────

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler para /start
    Limpa o histórico e apresenta o agente ao usuário.
    """
    chat_id = update.effective_chat.id
    memory.clear(chat_id)
    reset_session_cost(chat_id)

    welcome = (
        "👋 *Olá! Sou um agente autônomo inteligente.*\n\n"
        "Diferente de um chatbot comum, eu *penso antes de agir*:\n"
        "analiso sua pergunta, decido se preciso de ferramentas externas,\n"
        "uso-as se necessário, e formulo a melhor resposta possível.\n\n"
        "*O que posso fazer por você:*\n"
        "🔍 Buscar informações em tempo real na web\n"
        "🐍 Executar cálculos e código Python\n"
        "💬 Lembrar do contexto da nossa conversa\n\n"
        "*Comandos disponíveis:*\n"
        "/start — reiniciar a conversa\n"
        "/clear — limpar o histórico\n"
        "/custo — ver gasto da sessão atual\n"
        "/help  — ver exemplos de uso\n\n"
        "Como posso ajudar você hoje?"
    )

    await update.message.reply_text(welcome, parse_mode="Markdown")

    print(
        f"\n{Fore.GREEN}[🤖 BOT] Nova conversa iniciada — "
        f"chat_id={chat_id}{Style.RESET_ALL}"
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para /clear — limpa o histórico do chat atual e zera custo da sessão."""
    chat_id = update.effective_chat.id
    memory.clear(chat_id)
    reset_session_cost(chat_id)
    await update.message.reply_text(
        "🗑️ Histórico limpo! Começando uma conversa nova.\n"
        "O que você gostaria de saber?"
    )
    print(f"{Fore.YELLOW}[🗑️  CLEAR] Histórico limpo — chat_id={chat_id}{Style.RESET_ALL}")


async def cost_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler para /custo — exibe o custo acumulado da sessão atual.

    Cada interação com a IA consome tokens cobrados pela Google.
    Este comando mostra o gasto desde o início da conversa (ou desde o último /clear).
    """
    chat_id = update.effective_chat.id
    session_cost, turns = get_session_cost(chat_id)

    if turns == 0:
        await update.message.reply_text(
            "💰 *Custo da sessão atual*\n\n"
            "Nenhuma interação realizada ainda nesta sessão.\n"
            "_Envie uma mensagem e depois use /custo para ver o gasto._",
            parse_mode="Markdown"
        )
        return

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    await update.message.reply_text(
        f"💰 *Custo da sessão atual*\n\n"
        f"• Interações realizadas: *{turns}*\n"
        f"• Custo total: *${session_cost:.6f} USD*\n"
        f"• Custo médio por turno: *${session_cost / turns:.6f} USD*\n"
        f"• Modelo: `{model}`\n\n"
        f"_Use /clear para zerar o contador e iniciar nova sessão._",
        parse_mode="Markdown"
    )
    print(
        f"{Fore.GREEN}[💰 CUSTO] Sessão chat_id={chat_id} — "
        f"${session_cost:.6f} USD em {turns} turno(s){Style.RESET_ALL}"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler para /help — exibe exemplos de uso."""
    help_text = (
        "🤖 *Agente Autônomo — Exemplos de Uso*\n\n"
        "*Busca em tempo real:*\n"
        "• \"Qual o clima em Brasília hoje?\"\n"
        "• \"Como está o dólar agora?\"\n"
        "• \"Últimas notícias sobre inteligência artificial\"\n\n"
        "*Cálculos e análises:*\n"
        "• \"Calcule a média de 15, 27, 33, 42, 58\"\n"
        "• \"Qual o desvio padrão de [100, 200, 150, 175]?\"\n"
        "• \"Quanto é 15% de 3.450?\"\n\n"
        "*Memória de contexto:*\n"
        "• \"Meu nome é [seu nome], pode me chamar assim\"\n"
        "• \"Trabalho com [área] e quero saber sobre...\"\n\n"
        "*Perguntas gerais:*\n"
        "• \"O que é machine learning?\"\n"
        "• \"Explique o conceito de recursão\"\n\n"
        "*Comandos:*\n"
        "/start — reiniciar a conversa\n"
        "/clear — limpar o histórico\n"
        "/custo — ver gasto em USD da sessão atual\n"
        "/help  — esta mensagem\n\n"
        "_Dica: o raciocínio do agente aparece no terminal, "
        "não aqui no Telegram._"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


# ─────────────────────────────────────────────────────────────
# HANDLER PRINCIPAL DE MENSAGENS
# ─────────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler para mensagens de texto comuns.

    Fluxo:
    1. Exibe indicador "digitando..." no Telegram
    2. Executa run_agent() em thread separada (não bloqueia o event loop)
    3. Envia a resposta final ao usuário
    4. Divide automaticamente respostas longas (limite Telegram: 4096 chars)
    """
    chat_id = update.effective_chat.id
    user_message = update.message.text

    # ── FILTRO DE SEGURANÇA — injeção de prompt ──────────────────────────────
    # Executado ANTES de qualquer chamada à LLM: custo zero de tokens
    # e resposta imediata ao usuário em caso de ataque detectado.
    is_injection, matched_label = check_prompt_injection(user_message)
    if is_injection:
        log_injection_blocked(chat_id, user_message, matched_label)
        await update.message.reply_text(
            "⚠️ Mensagem bloqueada.\n\n"
            "Detectei uma tentativa de manipulação das minhas instruções de sistema. "
            "Esse tipo de mensagem não é processado por segurança.\n\n"
            "Se quiser fazer uma pergunta legítima, estou à disposição! 😊"
        )
        return
    # ────────────────────────────────────────────────────────────────────────

    # Exibe "digitando..." enquanto o agente processa
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Executa o agente em thread separada para não bloquear o event loop
    # asyncio.to_thread requer Python 3.9+
    try:
        response = await asyncio.to_thread(run_agent, user_message, chat_id, memory)
    except Exception as e:
        print(f"{Fore.RED}[❌ ERRO INESPERADO] {type(e).__name__}: {e}{Style.RESET_ALL}")
        response = (
            "Ocorreu um erro inesperado ao processar sua mensagem. "
            "Por favor, tente novamente."
        )

    # Envia a resposta — divide em partes se ultrapassar 4096 caracteres
    # (limite máximo do Telegram por mensagem)
    try:
        if len(response) <= 4096:
            await update.message.reply_text(response)
        else:
            # Divide em chunks sem quebrar palavras no meio
            for i in range(0, len(response), 4096):
                chunk = response[i:i + 4096]
                await update.message.reply_text(chunk)
                # Pequena pausa entre mensagens longas para não ser bloqueado
                if i + 4096 < len(response):
                    await asyncio.sleep(0.3)
    except Exception as send_error:
        print(f"{Fore.RED}[❌ ERRO AO ENVIAR] {send_error}{Style.RESET_ALL}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Trata erros do polling sem stack trace repetitivo no terminal."""
    err = context.error
    if isinstance(err, Conflict):
        print(
            f"\n{Fore.RED}[❌ CONFLITO TELEGRAM] Duas instâncias do bot estão rodando.{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}Encerre todas:{Style.RESET_ALL}\n"
            f"  Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
            f"| Where-Object {{ $_.CommandLine -match 'main\\.py' }} "
            f"| ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}\n"
        )
    else:
        print(f"\n{Fore.RED}[❌ TELEGRAM] {type(err).__name__}: {err}{Style.RESET_ALL}\n")


async def on_startup(application: Application) -> None:
    """Limpa webhook e fila pendente antes do polling."""
    await application.bot.delete_webhook(drop_pending_updates=True)


# ─────────────────────────────────────────────────────────────
# VALIDAÇÃO DE CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────

def validate_config() -> bool:
    """
    Verifica se as variáveis de ambiente obrigatórias estão configuradas.
    Exibe avisos para chaves opcionais ausentes.

    Returns:
        True se configuração está OK, False caso contrário
    """
    errors = []
    warnings = []

    if not os.getenv("TELEGRAM_TOKEN"):
        errors.append("TELEGRAM_TOKEN não configurado")

    if not os.getenv("GEMINI_API_KEY"):
        errors.append("GEMINI_API_KEY não configurado")

    # Verifica chave do provedor de busca configurado
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower()
    if provider == "tavily" and not os.getenv("TAVILY_API_KEY"):
        warnings.append("TAVILY_API_KEY não configurada — busca web não funcionará")
    elif provider == "serper" and not os.getenv("SERPER_API_KEY"):
        warnings.append("SERPER_API_KEY não configurada — busca web não funcionará")

    for error in errors:
        print(f"{Fore.RED}[❌ ERRO] {error}{Style.RESET_ALL}")

    for warning in warnings:
        print(f"{Fore.YELLOW}[⚠️  AVISO] {warning}{Style.RESET_ALL}")

    return len(errors) == 0


# ─────────────────────────────────────────────────────────────
# INICIALIZAÇÃO DO BOT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Ponto de entrada principal.
    Configura e inicia o bot Telegram em modo polling.
    """
    # Banner de inicialização
    print(f"\n{Fore.GREEN}{'═' * 62}")
    print(f"{Fore.GREEN}   🤖  AGENTE AUTÔNOMO TELEGRAM  —  Agentic Loop")
    print(f"{Fore.GREEN}{'═' * 62}{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Modelo:    {os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')} (Google Gemini)")
    print(f"  {Fore.CYAN}Memória:   últimas {os.getenv('MAX_HISTORY', '10')} mensagens (sliding window)")
    print(f"  {Fore.CYAN}Busca:     {os.getenv('SEARCH_PROVIDER', 'tavily')}")
    print(f"  {Fore.CYAN}Debug:     {os.getenv('DEBUG', 'false')}")
    print(f"  {Fore.CYAN}Histórico: salvo em history/chat_{{id}}.json")
    print(f"{Fore.GREEN}{'─' * 62}{Style.RESET_ALL}\n")

    # Valida configuração antes de iniciar
    if not validate_config():
        print(f"\n{Fore.RED}Configure o arquivo .env e tente novamente.{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Copie .env.example → .env e preencha as chaves.{Style.RESET_ALL}\n")
        sys.exit(1)

    acquire_single_instance_lock()

    token = os.getenv("TELEGRAM_TOKEN")

    # Cria a aplicação Telegram
    app = (
        Application.builder()
        .token(token)
        .post_init(on_startup)
        .build()
    )

    # Registra os handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("custo", cost_command))
    app.add_handler(CommandHandler("help", help_command))
    # Captura todas as mensagens de texto que não são comandos
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print(f"{Fore.GREEN}[✅ BOT INICIADO] Aguardando mensagens no Telegram...{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Pressione Ctrl+C para encerrar.\n{Style.RESET_ALL}")

    # Inicia o polling (verificação contínua de novas mensagens)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
