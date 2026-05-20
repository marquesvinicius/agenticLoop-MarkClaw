"""
tools/web_search.py — Ferramenta de Busca Web

Suporta dois provedores de busca configuráveis via variável SEARCH_PROVIDER:

  - Tavily (padrão, recomendado):
      API otimizada para uso com LLMs
      Retorna conteúdo limpo e resposta direta quando disponível
      Plano gratuito: https://tavily.com

  - Serper (alternativa):
      API baseada em Google Search
      Plano gratuito: https://serper.dev

Configure no .env:
  SEARCH_PROVIDER=tavily   ← ou serper
  TAVILY_API_KEY=...
  SERPER_API_KEY=...
"""

import os
from typing import Optional

import requests


# Timeout padrão para requisições de busca (segundos)
REQUEST_TIMEOUT = 15

# Número padrão de resultados a retornar
DEFAULT_MAX_RESULTS = 4

# Consultas que pedem listas completas (convocados, elencos, etc.)
_DETAIL_KEYWORDS = (
    "lista", "listagem", "convocados", "convocação", "convocacao",
    "elenco", "completa", "completo", "todos os", "todas as",
    "26 jogadores", "plantel", "relacionados", "escalacao", "escalação",
)


def _query_needs_detail(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _DETAIL_KEYWORDS)


def tavily_search(query: str, max_results: Optional[int] = None) -> str:
    """
    Realiza busca web usando a Tavily API.

    Tavily é projetada para LLMs: retorna conteúdo limpo,
    já processado e com resposta direta quando possível.

    Args:
        query: Consulta de busca
        max_results: Número máximo de resultados

    Returns:
        String com os resultados formatados
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not api_key:
        return (
            "Erro de configuração: TAVILY_API_KEY não encontrada no .env\n"
            "Obtenha sua chave gratuita em: https://tavily.com"
        )

    detailed = _query_needs_detail(query)
    if max_results is None:
        max_results = 8 if detailed else DEFAULT_MAX_RESULTS

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced" if detailed else "basic",
                "include_answer": True,
                "include_raw_content": False,
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        parts = []

        # Resposta direta (summary) quando a Tavily consegue inferir
        if data.get("answer"):
            parts.append(f"Resposta direta: {data['answer']}\n")

        # Resultados individuais — conteúdo mais longo em buscas de lista
        content_limit = 1200 if detailed else 600
        for result in data.get("results", [])[:max_results]:
            title = result.get("title", "Sem título")
            content = result.get("content", "Sem conteúdo")
            if len(content) > content_limit:
                content = content[:content_limit] + "..."
            url = result.get("url", "")
            parts.append(f"• {title}\n  {content}\n  Fonte: {url}")

        if detailed:
            parts.append(
                "\n[Instrução ao agente: extraia TODOS os nomes de jogadores "
                "mencionados nos trechos acima. Se ainda faltar nome, faça "
                "outra busca com query mais específica antes de dizer que não encontrou.]"
            )

        if not parts:
            return f"Nenhum resultado encontrado para: '{query}'"

        return "\n\n".join(parts)

    except requests.Timeout:
        return (
            f"Erro: A busca por '{query}' demorou muito (timeout de {REQUEST_TIMEOUT}s). "
            "Tente uma consulta mais específica ou tente novamente."
        )
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 401:
            return "Erro: TAVILY_API_KEY inválida ou expirada."
        if status == 429:
            return "Erro: Limite de requisições Tavily atingido. Tente novamente em alguns instantes."
        return f"Erro HTTP {status} ao acessar Tavily: {str(e)}"
    except requests.RequestException as e:
        return f"Erro de rede ao realizar busca: {str(e)}"
    except Exception as e:
        return f"Erro inesperado na busca: {str(e)}"


def serper_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """
    Realiza busca web usando a Serper API (Google Search via API).

    Use como alternativa ao Tavily quando SEARCH_PROVIDER=serper.

    Args:
        query: Consulta de busca
        max_results: Número máximo de resultados orgânicos

    Returns:
        String com os resultados formatados
    """
    api_key = os.getenv("SERPER_API_KEY", "").strip()

    if not api_key:
        return (
            "Erro de configuração: SERPER_API_KEY não encontrada no .env\n"
            "Obtenha sua chave gratuita em: https://serper.dev"
        )

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            },
            json={
                "q": query,
                "num": max_results,
                "gl": "br",   # Região: Brasil
                "hl": "pt"    # Idioma: Português
            },
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        parts = []

        # Answer box quando disponível (resposta direta do Google)
        answer_box = data.get("answerBox", {})
        if answer_box.get("answer"):
            parts.append(f"Resposta direta: {answer_box['answer']}\n")
        elif answer_box.get("snippet"):
            parts.append(f"Resposta direta: {answer_box['snippet']}\n")

        # Resultados orgânicos
        for result in data.get("organic", [])[:max_results]:
            title = result.get("title", "Sem título")
            snippet = result.get("snippet", "Sem descrição")
            link = result.get("link", "")
            parts.append(f"• {title}\n  {snippet}\n  Fonte: {link}")

        if not parts:
            return f"Nenhum resultado encontrado para: '{query}'"

        return "\n\n".join(parts)

    except requests.Timeout:
        return f"Erro: Timeout ao buscar '{query}' via Serper."
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        if status == 401:
            return "Erro: SERPER_API_KEY inválida."
        return f"Erro HTTP {status} ao acessar Serper: {str(e)}"
    except requests.RequestException as e:
        return f"Erro de rede ao realizar busca: {str(e)}"
    except Exception as e:
        return f"Erro inesperado: {str(e)}"


def web_search(query: str) -> str:
    """
    Interface unificada para busca web.

    Seleciona automaticamente o provedor configurado em SEARCH_PROVIDER.
    Padrão: tavily

    Args:
        query: Consulta de busca

    Returns:
        Resultados formatados como string
    """
    if not query or not query.strip():
        return "Erro: consulta de busca vazia."

    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower().strip()

    if provider == "serper":
        return serper_search(query)
    else:
        return tavily_search(query)
