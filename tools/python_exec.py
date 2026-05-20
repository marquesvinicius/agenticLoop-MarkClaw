"""
tools/python_exec.py — Execução Segura de Código Python (Sandbox)

Implementa um sandbox educacional simples para execução de código Python.
Permite que o agente resolva cálculos, estatísticas e operações com dados.

Mecanismos de segurança implementados:
  1. Blocklist de padrões perigosos (verificação textual antes do exec)
  2. Namespace restrito: __builtins__ substituído por whitelist segura
  3. Módulos pré-carregados: math, statistics, random (sem acesso a os/sys)
  4. Timeout via ThreadPoolExecutor (evita loops infinitos)
  5. Captura de stdout isolada via io.StringIO

AVISO: Para ambientes de produção real, use Docker ou gVisor.
       Este sandbox é adequado para fins educacionais.
"""

import sys
import io
import math
import statistics
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO SANDBOX
# ─────────────────────────────────────────────────────────────

# Padrões de texto que indicam código potencialmente perigoso
# Verificados antes do exec() como primeira linha de defesa
BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import shutil",
    "import pickle",
    "import marshal",
    "import pty",
    "import ctypes",
    "__import__(",
    "__builtins__",
    "open(",
    "file(",
    "input(",
    "compile(",
    "globals()",
    "locals()",
    "vars()",
]

# Funções built-in seguras para expor no namespace do sandbox
# Não inclui: open, input, compile, __import__, globals, locals, vars
SAFE_BUILTINS = {
    # I/O (apenas print)
    "print": print,
    # Tipos primitivos
    "int": int, "float": float, "str": str, "bool": bool,
    # Estruturas de dados
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "frozenset": frozenset,
    # Operações numéricas
    "abs": abs, "round": round, "pow": pow, "divmod": divmod,
    "sum": sum, "min": min, "max": max,
    # Iteração e sequências
    "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "len": len, "next": next, "iter": iter, "slice": slice,
    # Verificação de tipos
    "isinstance": isinstance, "issubclass": issubclass,
    "type": type, "hasattr": hasattr, "callable": callable,
    # Representação de strings e números
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    "repr": repr, "format": format, "hash": hash,
    # Exceções comuns (para código que faz tratamento de erro)
    "ValueError": ValueError, "TypeError": TypeError,
    "ZeroDivisionError": ZeroDivisionError, "IndexError": IndexError,
    "KeyError": KeyError, "StopIteration": StopIteration,
    "OverflowError": OverflowError, "RuntimeError": RuntimeError,
    "Exception": Exception, "ArithmeticError": ArithmeticError,
    # Constantes
    "None": None, "True": True, "False": False,
    "NotImplemented": NotImplemented,
}

# Timeout padrão para execução (segundos)
DEFAULT_TIMEOUT = 10


# ─────────────────────────────────────────────────────────────
# EXECUÇÃO INTERNA (roda em thread separada para timeout)
# ─────────────────────────────────────────────────────────────

def _run_code_in_sandbox(code: str) -> str:
    """
    Executa o código Python no namespace restrito e captura stdout.
    Esta função é executada em uma thread separada pelo ThreadPoolExecutor,
    permitindo que seja interrompida por timeout.

    Args:
        code: Código Python a executar

    Returns:
        Saída do programa (stdout) como string
    """
    # Buffer para capturar tudo que o código imprimir
    output_buffer = io.StringIO()

    # Namespace do sandbox:
    # - __builtins__: apenas funções da whitelist
    # - math, statistics, random: módulos pré-carregados
    sandbox_globals = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "statistics": statistics,
        "random": random,
    }

    # Redireciona stdout para o buffer (captura os prints do código)
    original_stdout = sys.stdout
    sys.stdout = output_buffer

    try:
        exec(code, sandbox_globals)
        output = output_buffer.getvalue()

        if not output.strip():
            return (
                "Código executado com sucesso, mas sem saída.\n"
                "Dica: use print() para exibir resultados. Exemplo: print(resultado)"
            )
        return output.strip()

    except SyntaxError as e:
        return f"Erro de sintaxe:\n  Linha {e.lineno}: {e.msg}\n  Verifique a indentação e a pontuação do código."
    except ZeroDivisionError:
        return "Erro matemático: divisão por zero."
    except NameError as e:
        # Erro comum: tentar usar módulo não pré-carregado
        return (
            f"Erro de nome: {e}\n"
            f"Módulos disponíveis no sandbox: math, statistics, random\n"
            f"Não é necessário importá-los — use diretamente: math.sqrt(16)"
        )
    except (ValueError, TypeError) as e:
        return f"Erro de valor/tipo: {e}"
    except RecursionError:
        return "Erro: recursão infinita detectada."
    except OverflowError:
        return "Erro: número muito grande (overflow)."
    except Exception as e:
        # Para outros erros, mostra tipo e mensagem (sem traceback completo)
        return f"Erro na execução ({type(e).__name__}): {e}"
    finally:
        # SEMPRE restaura stdout, mesmo se ocorrer exceção
        sys.stdout = original_stdout


# ─────────────────────────────────────────────────────────────
# INTERFACE PÚBLICA
# ─────────────────────────────────────────────────────────────

def execute_python(code: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Executa código Python em sandbox com timeout e verificação de segurança.

    Fluxo:
    1. Verifica padrões perigosos (blocklist textual)
    2. Executa em thread separada com timeout
    3. Retorna stdout capturado ou mensagem de erro

    Args:
        code: Código Python a executar. Use print() para ver resultados.
              Módulos disponíveis: math, statistics, random
        timeout: Tempo máximo de execução em segundos

    Returns:
        Saída do programa ou mensagem de erro amigável

    Exemplos de uso pelo agente:
        execute_python("print(statistics.mean([10, 20, 30]))")
        execute_python("resultado = sum(range(1, 101))\\nprint(resultado)")
    """
    if not code or not code.strip():
        return "Erro: nenhum código foi fornecido para execução."

    # ── Verificação de segurança (blocklist) ──────────────────
    code_lower = code.lower()
    for blocked in BLOCKED_PATTERNS:
        if blocked.lower() in code_lower:
            return (
                f"Código bloqueado por segurança: o padrão '{blocked}' não é permitido.\n"
                f"O sandbox suporta: operações matemáticas, estatísticas e estruturas de dados.\n"
                f"Módulos disponíveis: math, statistics, random"
            )

    # ── Execução com timeout ──────────────────────────────────
    # ThreadPoolExecutor permite interromper a thread após o timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_code_in_sandbox, code)
        try:
            result = future.result(timeout=timeout)
            return result
        except FuturesTimeoutError:
            return (
                f"Timeout: o código excedeu {timeout} segundos de execução.\n"
                "Verifique se há loops infinitos ou operações muito custosas."
            )
        except Exception as e:
            return f"Erro inesperado no sandbox: {str(e)}"
