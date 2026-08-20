"""Biblioteca de atributos derivados.

Modulo deliberadamente puro: recebe e devolve DataFrame, nao le nem escreve
arquivo, nao guarda estado. A orquestracao e a persistencia ficam em
`build_mart.py`, o que torna cada atributo testavel sem tocar em disco.

Restricao de projeto que vale explicitar: **toda feature aqui e linha a linha**.
Nenhuma depende de estatistica calculada sobre o conjunto (media, quantil,
residuo de regressao, codificacao por alvo). Transformacoes que *aprendem*
parametros pertencem ao `Pipeline` do scikit-learn, onde sao ajustadas apenas na
particao de treino de cada dobra da validacao cruzada. Colocadas aqui, elas
veriam o conjunto inteiro e vazariam informacao entre dobras -- inflando as
metricas de forma invisivel.

Isso restringe, em particular, a solucao mais direta para a colinearidade
`km` x `idade_veiculo` (0,947): ortogonalizar `km` pelo residuo de uma regressao
contra a idade seria uma transformacao ajustada, e nao entra no mart. A
alternativa linha a linha e `km_por_ano`, que separa intensidade de uso de tempo
de vida sem estimar nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Faixas de segmentacao
#
# Limites fixos, de leitura de negocio -- nao quantis da amostra. Quantis
# mudariam de significado a cada carga e nao poderiam ser comunicados a
# diretoria como uma regra estavel.
# --------------------------------------------------------------------------- #

LIMITES_IDADE = (-0.1, 1, 3, 5, config.IDADE_MAX)
ORDEM_FAIXA_IDADE = (
    "0-1 anos (novo)",
    "2-3 anos (garantia)",
    "4-5 anos (pos-garantia)",
    "6+ anos (maduro)",
)

LIMITES_KM = (-0.1, 25_000, 75_000, 125_000, config.KM_MAX)
ORDEM_FAIXA_KM = (
    "ate 25 mil km",
    "25-75 mil km",
    "75-125 mil km",
    "acima de 125 mil km",
)

#: Colunas produzidas por `adicionar_features`, na ordem em que sao criadas.
COLUNAS_DERIVADAS: tuple[str, ...] = (
    "km_por_ano",
    "reclamacoes_por_ano",
    "reclamacoes_por_10k_km",
    "muitas_reclamacoes",
    "faixa_idade",
    "faixa_km",
)

#: Subconjunto numerico das derivadas -- o que entra no modelo.
DERIVADAS_NUMERICAS: tuple[str, ...] = (
    "km_por_ano",
    "reclamacoes_por_ano",
    "reclamacoes_por_10k_km",
)

#: Subconjunto categorico -- serve segmentacao de dashboard, nao o modelo.
DERIVADAS_CATEGORICAS: tuple[str, ...] = ("faixa_idade", "faixa_km")


# --------------------------------------------------------------------------- #
# Atributos
# --------------------------------------------------------------------------- #


def _idade_ajustada(df: pd.DataFrame) -> pd.Series:
    """Idade com meio ano somado; ver `config.OFFSET_IDADE`."""
    return df["idade_veiculo"] + config.OFFSET_IDADE


def km_por_ano(df: pd.DataFrame) -> pd.Series:
    """Rodagem media anual.

    Separa *intensidade de uso* de *tempo de vida*, que na base bruta estao
    confundidos (correlacao de 0,947). Dois veiculos de 4 anos com 40 mil e 120
    mil km tem desgaste muito diferente, e essa distincao desaparece se o modelo
    so enxerga idade e quilometragem acumulada.
    """
    return df["km"] / _idade_ajustada(df)


def reclamacoes_por_ano(df: pd.DataFrame) -> pd.Series:
    """Frequencia anual de reclamacoes.

    Normaliza a contagem pelo tempo de exposicao: tres reclamacoes em um veiculo
    de 1 ano e um sinal mais forte do que tres em um de 8 anos.
    """
    return df["reclamacoes"] / _idade_ajustada(df)


def reclamacoes_por_10k_km(df: pd.DataFrame) -> pd.Series:
    """Reclamacoes a cada 10 mil km rodados.

    Normaliza pelo uso, e nao pelo tempo. E a leitura mais proxima de "taxa de
    falha por exposicao" que as quatro colunas disponiveis permitem construir.
    """
    return df["reclamacoes"] / (df["km"] / 10_000)


def muitas_reclamacoes(df: pd.DataFrame) -> pd.Series:
    """Indicadora do degrau identificado na Parte 1.

    A taxa de recall salta de 28,0% para 58,3% entre 2 e 3 reclamacoes, e depois
    estabiliza -- comportamento de limiar, nao de rampa. Modelos lineares nao
    capturam degrau em variavel continua; esta indicadora entrega o corte pronto.
    """
    return df["reclamacoes"] >= config.LIMIAR_RECLAMACOES


def faixa_idade(df: pd.DataFrame) -> pd.Series:
    """Segmenta a frota em quatro estagios de ciclo de vida."""
    return pd.cut(
        df["idade_veiculo"],
        bins=list(LIMITES_IDADE),
        labels=list(ORDEM_FAIXA_IDADE),
    ).astype("str")


def faixa_km(df: pd.DataFrame) -> pd.Series:
    """Segmenta a frota em quatro faixas de quilometragem."""
    return pd.cut(
        df["km"],
        bins=list(LIMITES_KM),
        labels=list(ORDEM_FAIXA_KM),
    ).astype("str")


# --------------------------------------------------------------------------- #
# Composicao
# --------------------------------------------------------------------------- #

_CONSTRUTORES = {
    "km_por_ano": km_por_ano,
    "reclamacoes_por_ano": reclamacoes_por_ano,
    "reclamacoes_por_10k_km": reclamacoes_por_10k_km,
    "muitas_reclamacoes": muitas_reclamacoes,
    "faixa_idade": faixa_idade,
    "faixa_km": faixa_km,
}


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma copia do DataFrame com todas as derivadas anexadas."""
    return df.assign(**{nome: _CONSTRUTORES[nome] for nome in COLUNAS_DERIVADAS})


def matriz_correlacao_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Correlacao entre originais e derivadas.

    Serve de checagem: uma derivada que correlaciona acima de ~0,95 com a sua
    origem nao acrescentou informacao, apenas reescalou a coluna.
    """
    colunas = ["idade_veiculo", "km", "reclamacoes", *DERIVADAS_NUMERICAS]
    return df[colunas].corr()


def resumo_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Descritivas das derivadas numericas, para conferencia rapida."""
    resumo = df[list(DERIVADAS_NUMERICAS)].describe().T
    resumo["assimetria"] = df[list(DERIVADAS_NUMERICAS)].skew()
    resumo["n_infinitos"] = [
        int(np.isinf(df[coluna]).sum()) for coluna in DERIVADAS_NUMERICAS
    ]
    return resumo
