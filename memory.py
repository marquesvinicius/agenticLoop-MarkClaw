"""
memory.py — Gerenciamento de Memória Conversacional

Implementa uma sliding window memory (janela deslizante):
mantém apenas as últimas N mensagens para evitar estourar
o limite de tokens da API de LLM.

Por que sliding window?
- APIs de LLM têm limite de tokens por requisição
- Enviar o histórico inteiro é caro e lento
- Mensagens recentes são as mais relevantes para o contexto

Separação de responsabilidades:
- O histórico COMPLETO fica salvo em JSON (para auditoria/relatório)
- Apenas as últimas MAX_HISTORY mensagens são enviadas à API
"""

import json
import os
from datetime import datetime


class ConversationMemory:
    """
    Memória conversacional com janela deslizante.

    Cada conversa é identificada pelo chat_id do Telegram.
    Suporta múltiplos usuários simultaneamente.
    """

    def __init__(self, max_messages: int = 10, save_to_file: bool = True):
        """
        Args:
            max_messages: Número máximo de mensagens enviadas à API (sliding window)
            save_to_file: Se True, salva histórico completo em JSON
        """
        self.max_messages = max_messages
        self.save_to_file = save_to_file
        # Dicionário principal: chat_id (int) → lista de mensagens
        self.conversations: dict[int, list] = {}

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        """
        Adiciona uma mensagem ao histórico de um chat.

        Args:
            chat_id: ID do chat no Telegram
            role: "user" ou "assistant"
            content: Texto da mensagem
        """
        if chat_id not in self.conversations:
            self.conversations[chat_id] = []

        self.conversations[chat_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        if self.save_to_file:
            self._save_to_file(chat_id)

    def get_messages(self, chat_id: int) -> list[dict]:
        """
        Retorna as últimas `max_messages` mensagens no formato padrão de chat.

        A sliding window é aplicada aqui: o histórico completo permanece
        em memória/arquivo, mas apenas as últimas N mensagens são retornadas
        para envio à API — eficiência de tokens.

        Returns:
            Lista de dicts {"role": ..., "content": ...} sem timestamp
        """
        if chat_id not in self.conversations:
            return []

        # Aplica a janela deslizante: últimas N mensagens
        recent = self.conversations[chat_id][-self.max_messages:]

        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in recent
        ]

    def get_message_count(self, chat_id: int) -> int:
        """Retorna o total de mensagens no histórico completo (não limitado)."""
        return len(self.conversations.get(chat_id, []))

    def get_stats(self, chat_id: int) -> dict:
        """
        Retorna estatísticas da janela de memória para exibição nos logs.

        Útil para demonstrar a eficiência de tokens no terminal:
        mostra quantas mensagens são enviadas à API vs quantas existem
        no histórico completo.

        Returns:
            Dict com:
              - total: total de mensagens no histórico completo
              - window: quantas serão enviadas à API (sliding window)
              - omitted: quantas foram omitidas para economizar tokens
              - ratio: porcentagem do histórico que é enviada
        """
        all_msgs = self.conversations.get(chat_id, [])
        total = len(all_msgs)
        window = min(total, self.max_messages)
        omitted = total - window
        ratio = round((window / total * 100) if total > 0 else 100)

        return {
            "total": total,
            "window": window,
            "omitted": omitted,
            "ratio": ratio,
            "max_window": self.max_messages,
        }

    def clear(self, chat_id: int) -> None:
        """Limpa todo o histórico de um chat específico."""
        self.conversations[chat_id] = []
        if self.save_to_file:
            self._save_to_file(chat_id)

    def _save_to_file(self, chat_id: int) -> None:
        """
        Persiste o histórico completo em um arquivo JSON.
        Útil para relatórios, debug e auditoria.
        """
        try:
            os.makedirs("history", exist_ok=True)
            filename = os.path.join("history", f"chat_{chat_id}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    self.conversations[chat_id],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception as e:
            # Falha ao salvar não deve interromper o agente
            print(f"[Aviso] Não foi possível salvar histórico: {e}")
