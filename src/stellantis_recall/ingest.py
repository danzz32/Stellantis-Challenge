"""Camada de ingestao: le a planilha de origem e valida o contrato bruto.

Responsabilidade unica: trazer o dado para dentro do projeto com nomes de coluna
estaveis e tipos declarados. Nenhuma regra de negocio e aplicada aqui -- nao ha
remocao de duplicatas, tratamento de outlier ou conversao do alvo. Isso e
deliberado: a analise exploratoria precisa enxergar o dado como ele chegou, e as
regras de limpeza sao consequencia dessa analise, nao premissa dela.

Uso pelo terminal:

    uv run recall-ingest
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.errors

from . import config
from .schemas import COLUNAS_RAW, RawSchema

# --------------------------------------------------------------------------- #
# Normalizacao de nomes de coluna
# --------------------------------------------------------------------------- #

_NAO_ALFANUMERICO = re.compile(r"[^0-9a-z]+")


def normalizar_nome(nome: str) -> str:
    """Converte um cabecalho da planilha em identificador snake_case ASCII.

    Preferido a um dicionario fixo de-para porque nao depende de a planilha ter
    sido salva na mesma codificacao: "Idade Veiculo", "IDADE VEICULO" e
    "idade_veiculo" convergem todos para o mesmo nome.

    >>> normalizar_nome("Reclamações")
    'reclamacoes'
    >>> normalizar_nome("Idade Veículo")
    'idade_veiculo'
    """
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return _NAO_ALFANUMERICO.sub("_", sem_acento.strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Relatorio de validacao
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RelatorioValidacao:
    """Resultado da validacao do contrato bruto.

    Guardar as falhas em um DataFrame, em vez de simplesmente propagar a
    excecao, permite que o relatorio de qualidade da Parte 1 mostre *todas* as
    violacoes de uma vez -- e nao apenas a primeira que o pandera encontrou.
    """

    conforme: bool
    n_linhas: int
    n_colunas: int
    falhas: pd.DataFrame

    @property
    def n_falhas(self) -> int:
        return len(self.falhas)

    def resumo(self) -> str:
        if self.conforme:
            return (
                f"Contrato RawSchema: OK "
                f"({self.n_linhas} linhas x {self.n_colunas} colunas)."
            )
        checagens = ", ".join(sorted(self.falhas["check"].astype(str).unique()))
        return (
            f"Contrato RawSchema: {self.n_falhas} violacao(oes) "
            f"em {self.n_linhas} linhas. Checagens: {checagens}."
        )


_COLUNAS_FALHA = ["schema_context", "column", "check", "failure_case", "index"]


def validar(df: pd.DataFrame) -> RelatorioValidacao:
    """Valida o DataFrame contra `RawSchema` sem interromper a execucao.

    Usa validacao preguicosa (`lazy=True`) para coletar o conjunto completo de
    violacoes em uma unica passada.
    """
    try:
        RawSchema.validate(df, lazy=True)
    except pandera.errors.SchemaErrors as erro:
        falhas = erro.failure_cases.reindex(columns=_COLUNAS_FALHA)
        return RelatorioValidacao(
            conforme=False,
            n_linhas=len(df),
            n_colunas=df.shape[1],
            falhas=falhas,
        )
    return RelatorioValidacao(
        conforme=True,
        n_linhas=len(df),
        n_colunas=df.shape[1],
        falhas=pd.DataFrame(columns=_COLUNAS_FALHA),
    )


# --------------------------------------------------------------------------- #
# Leitura
# --------------------------------------------------------------------------- #


def ler_planilha(caminho: Path | None = None) -> pd.DataFrame:
    """Le a planilha de origem e devolve o dado com colunas normalizadas."""
    caminho = caminho or config.RAW_XLSX
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Planilha de origem nao encontrada em {caminho}. "
            "O arquivo em data/raw/ e a unica entrada do projeto."
        )

    df = pd.read_excel(caminho)
    df = df.rename(columns={coluna: normalizar_nome(coluna) for coluna in df.columns})

    faltantes = set(COLUNAS_RAW) - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Colunas ausentes apos normalizacao: {sorted(faltantes)}. "
            f"Encontradas: {sorted(df.columns)}."
        )

    return df.loc[:, list(COLUNAS_RAW)]


def carregar_bruto(
    caminho: Path | None = None,
    *,
    estrito: bool = False,
) -> tuple[pd.DataFrame, RelatorioValidacao]:
    """Le a planilha e valida o contrato bruto.

    Args:
        caminho: planilha de origem; usa `config.RAW_XLSX` quando omitido.
        estrito: quando True, levanta excecao se o contrato for violado. Os
            notebooks usam False (querem inspecionar as falhas); o pipeline de
            producao usa True (nao deve seguir com dado fora do contrato).
    """
    df = ler_planilha(caminho)
    relatorio = validar(df)

    if estrito and not relatorio.conforme:
        raise ValueError(
            f"{relatorio.resumo()}\n{relatorio.falhas.to_string(index=False)}"
        )

    return df, relatorio


def carregar(caminho: Path | None = None) -> pd.DataFrame:
    """Atalho para os notebooks: devolve apenas o DataFrame validado."""
    df, _ = carregar_bruto(caminho)
    return df


# --------------------------------------------------------------------------- #
# Interface de linha de comando
# --------------------------------------------------------------------------- #


def main() -> int:
    """Executa a ingestao e imprime o resultado da validacao."""
    df, relatorio = carregar_bruto()

    print(relatorio.resumo())
    print()
    print("Colunas e tipos:")
    print(df.dtypes.to_string())

    if not relatorio.conforme:
        print()
        print("Violacoes encontradas:")
        print(relatorio.falhas.to_string(index=False))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
