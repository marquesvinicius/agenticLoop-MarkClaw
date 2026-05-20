# Agente Autônomo Telegram — Agentic Loop

Trabalho da disciplina **Tópicos em Engenharia de Software** (UNIRV).

Um agente autônomo para Telegram que *pensa antes de agir*, usa ferramentas externas, mantém memória de contexto e exibe todo o seu raciocínio no terminal.

**LLM:** Google Gemini (via Google AI Studio)

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    USUÁRIO (Telegram)                       │
│                  "Qual o clima hoje?"                       │
└────────────────────────┬────────────────────────────────────┘
                         │ mensagem
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      main.py                                │
│              Bot Telegram (handlers async)                  │
│         asyncio.to_thread → não bloqueia event loop         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      agent.py                               │
│                   AGENTIC LOOP                              │
│                                                             │
│  [system_prompt] + [memory.get_messages()] → Gemini API     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. THOUGHT  — raciocínio interno (só no terminal)  │    │
│  │  2. ACTION   — chama uma ferramenta                 │    │
│  │  3. OBSERVATION — recebe resultado                  │    │
│  │  4. (repete se necessário)                          │    │
│  │  5. FINAL ANSWER — resposta limpa para o usuário    │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐      ┌─────────────────────────┐
│  tools/          │      │  memory.py              │
│  web_search.py   │      │  ConversationMemory     │
│  (Tavily/Serper) │      │  sliding window N msgs  │
│                  │      │  save → history/*.json  │
│  python_exec.py  │      └─────────────────────────┘
│  (sandbox exec)  │
└──────────────────┘
```

---

## O Agentic Loop: Thought → Action → Observation → Answer

O conceito central é um **loop agentic manual** (Thought → Action → Observation → Answer),
*inspirado* no padrão ReAct da literatura, sem usar biblioteca ReAct.
O agente não responde diretamente — ele *raciocina* primeiro.

### O que aparece no TERMINAL:

```
══════════════════════════════════════════════════════════════
[👤 USUÁRIO]  chat_id=123456789
Qual o clima em Goiânia hoje?
──────────────────────────────────────────────────────────────

  💭 THOUGHT  (iteração 1)
  O usuário quer saber o clima atual em Goiânia.
  Isso é informação em tempo real — devo usar web_search
  para obter dados atuais em vez de inventar.

  ⚡ ACTION
  Ferramenta: web_search
  {
      "query": "clima Goiânia hoje temperatura"
  }

  👁  OBSERVATION
  Resposta direta: Goiânia, 31°C, parcialmente nublado
  • Previsão do tempo Goiânia — 31°C com chance de chuva...
    Fonte: https://...

  ✅ FINAL ANSWER
  Hoje em Goiânia está fazendo **31°C** com tempo parcialmente
  nublado e chance de chuva à tarde.
```

### O que o usuário VÊ no Telegram:

```
Hoje em Goiânia está fazendo **31°C** com tempo parcialmente
nublado e chance de chuva à tarde.
```

**O raciocínio fica apenas no terminal!** O usuário recebe apenas a resposta final.

---

## Estrutura de Arquivos

```
AgenticLoop/
├── main.py               ← Entry point: inicializa e roda o bot
├── agent.py              ← Agentic Loop: Thought→Action→Observation→Answer
├── memory.py             ← Sliding window memory (últimas N mensagens)
├── tools/
│   ├── __init__.py
│   ├── web_search.py     ← Busca Tavily (padrão) ou Serper
│   └── python_exec.py    ← Sandbox Python com timeout
├── prompts/
│   └── system_prompt.txt ← Instruções do sistema (anti-alucinação, etc.)
├── history/              ← Histórico JSON por chat (criado automaticamente)
├── requirements.txt
├── .env.example          ← Template de configuração
└── README.md
```

---

## Instalação e Configuração

### 1. Pré-requisitos

- Python 3.9 ou superior
- Conta no Telegram
- Conta no Google AI Studio (para o Gemini — **gratuito**)
- Conta no Tavily ou Serper (para busca web)

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

```bash
# Copie o template
copy .env.example .env   # Windows
# ou
cp .env.example .env     # Linux/Mac
```

Abra o arquivo `.env` e preencha:

```env
TELEGRAM_TOKEN=seu_token_aqui
GEMINI_API_KEY=sua_chave_gemini_aqui
TAVILY_API_KEY=sua_chave_tavily_aqui
```

### 4. Executar

```bash
python main.py
```

---

## Como Obter as Chaves de API

### Token do Telegram Bot
1. Abra o Telegram e procure por **@BotFather**
2. Envie `/newbot`
3. Escolha um nome e username para o bot
4. O BotFather enviará o token — copie e cole no `.env`

### Chave do Gemini (Google AI Studio) — GRATUITO
1. Acesse [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Clique em **Create API Key**
3. Copie a chave e cole no `.env` como `GEMINI_API_KEY`

> **Custo:** O `gemini-2.5-flash-lite` tem cota gratuita generosa no Google AI Studio.
> Para testes e trabalhos acadêmicos, o plano gratuito é mais do que suficiente.

### Chave Tavily (recomendado para busca web)
1. Acesse [tavily.com](https://tavily.com)
2. Crie uma conta gratuita
3. Vá em **API Keys** e copie sua chave
4. Cole no `.env` como `TAVILY_API_KEY`

> O plano gratuito inclui 1.000 buscas/mês.

### Chave Serper (alternativa de busca)
1. Acesse [serper.dev](https://serper.dev)
2. Crie uma conta gratuita (2.500 buscas grátis)
3. Copie a chave e configure `SEARCH_PROVIDER=serper` no `.env`

---

## Exemplos de Interação

### Busca em tempo real
```
Você: Qual a cotação do dólar hoje?
Agente: O dólar está sendo negociado a R$ X,XX ...
```

### Cálculo com Python
```
Você: Calcule a média e o desvio padrão de [85, 92, 78, 95, 88, 76]
Agente: Média: 85.67 | Desvio padrão: 7.06
```

### Memória de contexto
```
Você: Meu nome é Marques e estudo engenharia de software
Agente: Olá Marques! ...
[mais tarde]
Você: Qual é o meu nome?
Agente: Seu nome é Marques e você estuda engenharia de software.
```

### Encadeamento de ferramentas
```
Você: Qual o PIB do Brasil e quanto é isso dividido pela população?
[Agente busca PIB → busca população → calcula divisão via Python]
```

---

## Três Pilares Implementados

### Pilar A — Reasoning (Agentic Loop)
- Implementado em `agent.py`
- O modelo recebe um system prompt que o instrui a escrever `[Thought]...[/Thought]`
- O agente extrai e exibe o pensamento apenas no terminal
- A resposta final para o usuário é limpa de marcadores internos

### Pilar B — Memória (Sliding Window)
- Implementado em `memory.py`
- Mantém apenas as últimas `MAX_HISTORY` mensagens (padrão: 10)
- Tool calls intermediários NÃO são salvos — apenas pares user/assistant
- Isso economiza tokens e evita "poluir" o contexto com detalhes técnicos
- Histórico completo salvo em `history/chat_{id}.json` para auditoria

### Pilar C — Tool Calling (Function Calling)
- Implementado em `tools/web_search.py` e `tools/python_exec.py`
- O Gemini decide automaticamente quando usar cada ferramenta
- `web_search`: Tavily API ou Serper API (configurável)
- `execute_python`: sandbox com namespace restrito, timeout de 10s, captura de stdout

---

## Configurações Disponíveis (.env)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `TELEGRAM_TOKEN` | — | Token do bot (obrigatório) |
| `GEMINI_API_KEY` | — | Chave Google AI Studio (obrigatório) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Modelo Gemini a usar |
| `SEARCH_PROVIDER` | `tavily` | Provedor de busca (`tavily` ou `serper`) |
| `TAVILY_API_KEY` | — | Chave Tavily |
| `SERPER_API_KEY` | — | Chave Serper |
| `MAX_HISTORY` | `10` | Tamanho da janela de memória |
| `DEBUG` | `false` | Logs extras no terminal |

---

## Tratamento de Erros

| Situação | Comportamento |
|----------|---------------|
| API Gemini indisponível | Mensagem amigável ao usuário, log no terminal |
| Resposta filtrada pelo Gemini | Mensagem de erro amigável |
| Timeout de busca web | Informa usuário, sugere reformular |
| Chave de API ausente | Log de aviso ao iniciar, encerra com instrução |
| Código Python inválido | Retorna erro de sintaxe legível |
| Loop infinito no código | Timeout de 10 segundos |
| Padrão perigoso no código | Bloqueio com explicação |
| Resposta > 4096 chars | Divisão automática em múltiplas mensagens |

---

## Tecnologias Utilizadas

- **Python 3.9+**
- **python-telegram-bot 20.7** — framework do bot (asyncio)
- **google-genai >= 0.7** — SDK oficial do Google Gemini
- **requests** — chamadas HTTP para APIs de busca
- **python-dotenv** — carregamento de variáveis de ambiente
- **colorama** — logs coloridos no terminal (compatível com Windows)
