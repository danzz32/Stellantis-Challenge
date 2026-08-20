"""Camada `trusted`: aplica as regras de limpeza e persiste o dado confiavel.

As regras implementadas aqui nao foram escolhidas a priori -- cada uma responde
a um achado da analise exploratoria (`qmd/01-analise.qmd`), e o motivo esta
registrado junto da regra. Esse encadeamento e o ponto: a limpeza e consequencia
do diagnostico, nao um passo padrao aplicado antes de olhar o dado.

Uso pelo terminal:

    uv run python -m stellantis_recall.transform
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config, ingest
from .schemas import COLUNAS_TRUSTED, TrustedSchema

# --------------------------------------------------------------------------- #
# Regras de limpeza
# --------------------------------------------------------------------------- #


def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove linhas integralmente repetidas, preservando a primeira ocorrencia.

    Achado da Parte 1: existe uma unica linha duplicada, e o valor de `km`
    (1.000, o minimo do dataset) sugere colisao no piso do gerador sintetico e
    nao erro de carga. Ela e removida assim mesmo: manter a mesma observacao
    fisica em treino e teste durante a validacao cruzada e um vazamento pequeno,
    porem gratuito. Criterio de risco assimetrico, nao de magnitude.
    """
    limpo = df.drop_duplicates(keep="first").reset_index(drop=True)
    return limpo, len(df) - len(limpo)


def binarizar_alvo(serie: pd.Series) -> pd.Series:
    """Converte o alvo textual 'Sim'/'Nao' em booleano.

    'Sim'/'Nao' e codificacao de apresentacao. Mante-la ate a modelagem
    obrigaria comparacoes por string em todo o projeto -- inclusive contra
    "Nao" com til, que e a fonte classica de bug silencioso por codificacao.

    Uma serie ja booleana passa adiante inalterada. Sem isso, `tipar` deixaria
    de ser idempotente e reprocessar a camada trusted quebraria.
    """
    if pd.api.types.is_bool_dtype(serie):
        return serie

    invalidos = set(serie.dropna().unique()) - set(config.ROTULOS_ALVO)
    if invalidos:
        raise ValueError(
            f"Valores inesperados em '{config.COLUNA_ALVO}': {sorted(invalidos)}. "
            f"Esperado: {list(config.ROTULOS_ALVO)}."
        )
    return serie.eq(config.ROTULO_POSITIVO)


def rotular_alvo(serie: pd.Series) -> pd.Series:
    """Inverso de `binarizar_alvo`, para exibicao em tabelas e graficos."""
    return serie.map({True: config.ROTULO_POSITIVO, False: config.ROTULO_NEGATIVO})


def tipar(df: pd.DataFrame) -> pd.DataFrame:
    """Fixa os tipos do contrato `trusted`.

    `modelo` permanece como texto, e nao como categorica. A conversao para
    `category` era a decisao registrada na Parte 1, mas o pandas 3 ja usa dtype
    de string respaldado por Arrow: o ganho de memoria desapareceu, o Parquet
    faz dicionarizacao por conta propria, e a enumeracao dos 9 niveis validos ja
    e garantida pelo `isin` do contrato. Manter texto elimina um dtype com
    comportamento peculiar em groupby e merge sem perder nenhuma garantia.
    """
    tipado = df.copy()
    tipado[config.COLUNA_ALVO] = binarizar_alvo(tipado[config.COLUNA_ALVO])
    for coluna in ("idade_veiculo", "km", "reclamacoes"):
        tipado[coluna] = tipado[coluna].astype("int64")
    tipado["modelo"] = tipado["modelo"].astype("str")
    return tipado.loc[:, list(COLUNAS_TRUSTED)]


def transformar(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Aplica a limpeza completa. Funcao pura: nao le nem escreve arquivo."""
    limpo, n_removidas = remover_duplicatas(df)
    return tipar(limpo), n_removidas


# --------------------------------------------------------------------------- #
# Relatorio de qualidade
# --------------------------------------------------------------------------- #


def relatorio_qualidade(
    origem: pd.DataFrame,
    resultado: pd.DataFrame,
    n_removidas: int,
) -> pd.DataFrame:
    """Tabela de verificacoes da transformacao, persistida como artefato.

    Existe para que a "avaliacao de qualidade" pedida no enunciado seja um
    arquivo auditavel e nao um paragrafo de notebook.
    """
    # `valor` e texto de proposito: a tabela mistura contagens, proporcoes e
    # nomes de tipo, e uma coluna heterogenea nao tem representacao em Parquet.
    verificacoes = [
        ("linhas na origem", f"{len(origem)}", "-"),
        (
            "duplicatas removidas",
            f"{n_removidas}",
            "regra: drop_duplicates(keep='first')",
        ),
        ("linhas na trusted", f"{len(resultado)}", "-"),
        ("colunas na trusted", f"{resultado.shape[1]}", ", ".join(COLUNAS_TRUSTED)),
        (
            "valores nulos",
            f"{int(resultado.isna().sum().sum())}",
            "contrato exige zero",
        ),
        (
            "modelos distintos",
            f"{resultado['modelo'].nunique()}",
            f"esperado: {len(config.MODELOS)}",
        ),
        (
            "taxa do alvo",
            f"{resultado[config.COLUNA_ALVO].mean():.4f}",
            f"proporcao de '{config.ROTULO_POSITIVO}'",
        ),
        (
            "tipo do alvo",
            str(resultado[config.COLUNA_ALVO].dtype),
            "booleano apos binarizacao",
        ),
        (
            "contrato TrustedSchema",
            "conforme",
            "validado antes da gravacao",
        ),
    ]
    return pd.DataFrame(verificacoes, columns=["verificacao", "valor", "detalhe"])


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResultadoTransformacao:
    """Saida da etapa, com o dado e a evidencia de que ele esta conforme."""

    df: pd.DataFrame
    n_removidas: int
    relatorio: pd.DataFrame
    destino: Path | None


def executar(
    origem: Path | None = None,
    destino: Path | None = None,
    *,
    persistir: bool = True,
) -> ResultadoTransformacao:
    """Le a origem, limpa, valida contra `TrustedSchema` e grava o Parquet.

    A validacao aqui e estrita: diferente da ingestao, que coleta as violacoes
    para diagnostico, esta camada nao deve produzir arquivo fora do contrato.
    """
    bruto, _ = ingest.carregar_bruto(origem, estrito=True)
    limpo, n_removidas = transformar(bruto)

    validado = TrustedSchema.validate(limpo, lazy=True)
    relatorio = relatorio_qualidade(bruto, validado, n_removidas)

    caminho = destino or config.TRUSTED_VEICULOS
    if persistir:
        config.garantir_diretorios()
        validado.to_parquet(caminho, index=False)
        config.RELATORIO_QUALIDADE.parent.mkdir(parents=True, exist_ok=True)
        relatorio.to_parquet(config.RELATORIO_QUALIDADE, index=False)

    return ResultadoTransformacao(
        df=validado,
        n_removidas=n_removidas,
        relatorio=relatorio,
        destino=caminho if persistir else None,
    )


def carregar(caminho: Path | None = None) -> pd.DataFrame:
    """Le a camada `trusted` ja materializada."""
    caminho = caminho or config.TRUSTED_VEICULOS
    if not caminho.is_file():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode `uv run recall-pipeline` para construi-la."
        )
    return pd.read_parquet(caminho)


def main() -> int:
    """Executa a transformacao e imprime o relatorio de qualidade."""
    resultado = executar()
    print(resultado.relatorio.to_string(index=False))
    print()
    print(f"Gravado em: {resultado.destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
