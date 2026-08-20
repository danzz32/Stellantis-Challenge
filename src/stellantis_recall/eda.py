"""Analise exploratoria: consultas DuckDB e estatisticas de apoio.

Este modulo concentra o *calculo* da Parte 1. O documento Quarto apenas chama
estas funcoes e narra o resultado -- nenhuma logica analitica vive no notebook,
de modo que cada numero do relatorio tem um teste possivel e um unico lugar de
manutencao.

Sobre o DuckDB: 500 linhas nao exigem um motor analitico por desempenho. Ele
esta aqui por duas razoes de projeto. Primeira, as agregacoes ficam em SQL
declarativo versionado em `sql/`, e a mesma consulta serve relatorio e
dashboard, sem reimplementacao em pandas. Segunda, a leitura direta de Parquet
mantem o desenho valido se a base real de ocorrencias de garantia tiver ordens
de grandeza a mais.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from . import config

TABELA_PADRAO = "veiculos"

COLUNAS_NUMERICAS: tuple[str, ...] = ("idade_veiculo", "km", "reclamacoes")
COLUNAS_CATEGORICAS: tuple[str, ...] = ("modelo",)


# --------------------------------------------------------------------------- #
# Acesso ao DuckDB
# --------------------------------------------------------------------------- #


def conectar(
    df: pd.DataFrame,
    *,
    tabela: str = TABELA_PADRAO,
) -> duckdb.DuckDBPyConnection:
    """Abre uma conexao em memoria expondo a *visao analitica canonica*.

    A visao canonica tem uma unica regra: `recall` e sempre booleano. Isso
    permite que o mesmo conjunto de consultas em `sql/` sirva a analise
    exploratoria (que le a origem, onde o alvo e texto 'Sim'/'Nao') e a camada
    `mart` (que le o Parquet ja tipado) sem duas versoes de cada consulta.

    A conversao feita aqui e de *consulta*, nao de limpeza: nada e persistido.
    A conversao autoritativa do alvo vive em `transform.py`.
    """
    con = duckdb.connect(":memory:")
    con.register(f"_{tabela}_origem", df)

    if pd.api.types.is_bool_dtype(df[config.COLUNA_ALVO]):
        expressao_alvo = config.COLUNA_ALVO
    else:
        expressao_alvo = f"{config.COLUNA_ALVO} = $rotulo_positivo"

    con.execute(
        f"""
        create or replace table {tabela} as
        select * exclude ({config.COLUNA_ALVO}),
               {expressao_alvo} as {config.COLUNA_ALVO}
        from _{tabela}_origem
        """,
        {"rotulo_positivo": config.ROTULO_POSITIVO},
    )
    return con


def conectar_parquet(
    caminho: Path,
    *,
    tabela: str = TABELA_PADRAO,
) -> duckdb.DuckDBPyConnection:
    """Abre uma conexao lendo o Parquet diretamente, sem passar pelo pandas.

    E este o caminho usado pela camada `mart` e pelo dashboard: o DuckDB le o
    arquivo colunar sem materializar o dado em memoria antes de agregar.
    """
    con = duckdb.connect(":memory:")
    con.execute(
        f"create or replace view {tabela} as "
        f"select * from read_parquet($caminho)",
        {"caminho": str(caminho)},
    )
    return con


@lru_cache(maxsize=None)
def ler_sql(nome: str) -> str:
    """Le uma consulta de `sql/<nome>.sql`."""
    caminho = config.SQL_DIR / f"{nome}.sql"
    if not caminho.is_file():
        disponiveis = sorted(p.stem for p in config.SQL_DIR.glob("*.sql"))
        raise FileNotFoundError(
            f"Consulta '{nome}' nao encontrada em {config.SQL_DIR}. "
            f"Disponiveis: {disponiveis}."
        )
    return caminho.read_text(encoding="utf-8")


#: Parametros que praticamente toda consulta do projeto usa. O alvo nao aparece
#: aqui: na visao canonica ele ja e booleano.
PARAMETROS_PADRAO: dict[str, object] = {
    "z": config.Z_NIVEL_CONFIANCA,
}


def consultar(
    con: duckdb.DuckDBPyConnection,
    nome: str,
    **parametros: object,
) -> pd.DataFrame:
    """Executa a consulta nomeada e devolve o resultado como DataFrame.

    Os parametros sao passados de forma nomeada ao DuckDB (`$nome`), nunca por
    interpolacao de string.
    """
    sql = ler_sql(nome)
    argumentos = {
        chave: valor
        for chave, valor in {**PARAMETROS_PADRAO, **parametros}.items()
        if f"${chave}" in sql
    }
    return con.execute(sql, argumentos).df()


# --------------------------------------------------------------------------- #
# Qualidade
# --------------------------------------------------------------------------- #


def duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve as linhas integralmente duplicadas, com a contagem de ocorrencias.

    Sem chave primaria no dataset, uma linha repetida e ambigua: pode ser dois
    veiculos distintos com o mesmo perfil ou o mesmo registro carregado duas
    vezes. A decisao de tratamento e tomada em `transform.py`; aqui apenas se
    mede a extensao do problema.
    """
    marcadas = df[df.duplicated(keep=False)]
    if marcadas.empty:
        return marcadas.assign(n_ocorrencias=pd.Series(dtype="int64"))
    return (
        marcadas.groupby(list(df.columns), as_index=False, observed=True)
        .size()
        .rename(columns={"size": "n_ocorrencias"})
        .sort_values("n_ocorrencias", ascending=False)
    )


def outliers_iqr(df: pd.DataFrame, coluna: str, fator: float = 1.5) -> dict[str, float]:
    """Limites de Tukey e contagem de pontos fora deles."""
    q1, q3 = df[coluna].quantile([0.25, 0.75])
    iqr = q3 - q1
    inferior, superior = q1 - fator * iqr, q3 + fator * iqr
    fora = df[coluna].lt(inferior) | df[coluna].gt(superior)
    return {
        "coluna": coluna,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "limite_inferior": float(inferior),
        "limite_superior": float(superior),
        "n_outliers": int(fora.sum()),
        "pct_outliers": float(fora.mean()),
    }


def resumo_outliers(
    df: pd.DataFrame,
    colunas: tuple[str, ...] = COLUNAS_NUMERICAS,
) -> pd.DataFrame:
    """Aplica `outliers_iqr` a varias colunas."""
    return pd.DataFrame([outliers_iqr(df, coluna) for coluna in colunas])


def consistencia_km_ano(df: pd.DataFrame) -> pd.DataFrame:
    """Rodagem anual implicita, usada como checagem de coerencia interna.

    Km e idade sao declarados de forma independente na origem; a razao entre
    eles precisa cair numa faixa fisicamente plausivel. Veiculos com idade 0
    ficam de fora por divisao indefinida -- e sao reportados a parte.
    """
    elegiveis = df.loc[df["idade_veiculo"] > 0].copy()
    elegiveis["km_por_ano"] = elegiveis["km"] / elegiveis["idade_veiculo"]
    elegiveis["fora_da_faixa"] = ~elegiveis["km_por_ano"].between(
        config.KM_ANO_MIN, config.KM_ANO_MAX
    )
    return elegiveis


# --------------------------------------------------------------------------- #
# Estatistica descritiva
# --------------------------------------------------------------------------- #


def estatisticas_numericas(
    df: pd.DataFrame,
    colunas: tuple[str, ...] = COLUNAS_NUMERICAS,
) -> pd.DataFrame:
    """Descritivas estendidas: posicao, dispersao e forma."""
    recorte = df[list(colunas)]
    resumo = recorte.describe().T
    resumo["cv"] = resumo["std"] / resumo["mean"]
    resumo["assimetria"] = recorte.skew()
    resumo["curtose"] = recorte.kurtosis()
    resumo["iqr"] = resumo["75%"] - resumo["25%"]
    return resumo.rename(
        columns={
            "count": "n",
            "mean": "media",
            "std": "desvio",
            "min": "minimo",
            "max": "maximo",
            "50%": "mediana",
        }
    )


def estatisticas_categoricas(
    df: pd.DataFrame,
    colunas: tuple[str, ...] = COLUNAS_CATEGORICAS,
) -> pd.DataFrame:
    """Cardinalidade, moda e concentracao das colunas categoricas."""
    linhas = []
    for coluna in colunas:
        contagem = df[coluna].value_counts()
        linhas.append(
            {
                "coluna": coluna,
                "n_categorias": int(contagem.size),
                "moda": contagem.index[0],
                "freq_moda": int(contagem.iloc[0]),
                "pct_moda": float(contagem.iloc[0] / len(df)),
                "menor_categoria": contagem.index[-1],
                "freq_menor": int(contagem.iloc[-1]),
            }
        )
    return pd.DataFrame(linhas)


def distribuicao_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """Frequencia absoluta e relativa do alvo."""
    contagem = df[config.COLUNA_ALVO].value_counts()
    return pd.DataFrame(
        {
            "classe": contagem.index,
            "n": contagem.to_numpy(),
            "proporcao": (contagem / len(df)).to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# Associacao entre variaveis
# --------------------------------------------------------------------------- #


def alvo_binario(df: pd.DataFrame) -> pd.Series:
    """Alvo como 0/1, sem materializar a coluna no dado de origem."""
    return df[config.COLUNA_ALVO].eq(config.ROTULO_POSITIVO).astype("int8")


def matriz_correlacao(
    df: pd.DataFrame,
    metodo: str = "pearson",
    colunas: tuple[str, ...] = COLUNAS_NUMERICAS,
) -> pd.DataFrame:
    """Correlacao entre preditoras numericas e o alvo binarizado."""
    base = df[list(colunas)].assign(recall_bin=alvo_binario(df))
    return base.corr(method=metodo)


def correlacao_com_alvo(
    df: pd.DataFrame,
    colunas: tuple[str, ...] = COLUNAS_NUMERICAS,
) -> pd.DataFrame:
    """Correlacao ponto-bisserial de cada preditora com o alvo, com p-valor.

    Ponto-bisserial e o caso da correlacao de Pearson com uma das variaveis
    dicotomica; o p-valor acompanha para separar sinal de ruido amostral.
    """
    y = alvo_binario(df)
    linhas = []
    for coluna in colunas:
        r, p = stats.pointbiserialr(y, df[coluna])
        rho, p_rho = stats.spearmanr(y, df[coluna])
        linhas.append(
            {
                "variavel": coluna,
                "r_ponto_bisserial": float(r),
                "p_valor": float(p),
                "rho_spearman": float(rho),
                "p_valor_spearman": float(p_rho),
                "significante_5pct": bool(p < config.ALFA),
            }
        )
    return pd.DataFrame(linhas).sort_values(
        "r_ponto_bisserial", key=np.abs, ascending=False
    )


def fator_inflacao_variancia(
    df: pd.DataFrame,
    colunas: tuple[str, ...] = COLUNAS_NUMERICAS,
) -> pd.DataFrame:
    """VIF de cada preditora numerica.

    Calculado diretamente (VIF = 1 / (1 - R^2) da regressao de cada variavel
    contra as demais) para nao arrastar o statsmodels como dependencia por
    conta de tres colunas.

    Leitura usual: VIF > 5 indica colinearidade preocupante; VIF > 10, severa.
    """
    matriz = df[list(colunas)].to_numpy(dtype=float)
    linhas = []
    for indice, coluna in enumerate(colunas):
        y = matriz[:, indice]
        outras = np.delete(matriz, indice, axis=1)
        x = np.column_stack([np.ones(len(outras)), outras])
        coeficientes, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuo = y - x @ coeficientes
        sq_total = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float((residuo**2).sum()) / sq_total if sq_total else 0.0
        linhas.append(
            {
                "variavel": coluna,
                "r2_contra_demais": r2,
                "vif": float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2),
            }
        )
    return pd.DataFrame(linhas).sort_values("vif", ascending=False)


@dataclass(frozen=True, slots=True)
class TesteAssociacao:
    """Resultado de um teste qui-quadrado de independencia."""

    variavel: str
    qui_quadrado: float
    p_valor: float
    graus_liberdade: int
    v_cramer: float
    n: int

    @property
    def significante(self) -> bool:
        return self.p_valor < config.ALFA

    def frase(self) -> str:
        veredito = "ha" if self.significante else "nao ha"
        return (
            f"chi2({self.graus_liberdade}) = {self.qui_quadrado:.2f}, "
            f"p = {self.p_valor:.3f}, V de Cramer = {self.v_cramer:.3f} "
            f"-- {veredito} evidencia de associacao a 5%."
        )


def qui_quadrado_com_alvo(df: pd.DataFrame, coluna: str) -> TesteAssociacao:
    """Testa independencia entre uma categorica e o alvo.

    O V de Cramer acompanha o p-valor porque um teste significante em amostra
    grande ainda pode corresponder a um efeito irrelevante -- e o inverso vale
    aqui, onde a amostra e pequena.
    """
    tabela = pd.crosstab(df[coluna], df[config.COLUNA_ALVO])
    qui2, p, gl, _ = stats.chi2_contingency(tabela)
    n = int(tabela.to_numpy().sum())
    menor_dimensao = min(tabela.shape) - 1
    v_cramer = float(np.sqrt(qui2 / (n * menor_dimensao))) if menor_dimensao else 0.0
    return TesteAssociacao(
        variavel=coluna,
        qui_quadrado=float(qui2),
        p_valor=float(p),
        graus_liberdade=int(gl),
        v_cramer=v_cramer,
        n=n,
    )


# --------------------------------------------------------------------------- #
# Taxas com incerteza
# --------------------------------------------------------------------------- #


def intervalo_wilson(
    sucessos: np.ndarray | pd.Series,
    total: np.ndarray | pd.Series,
    z: float = config.Z_NIVEL_CONFIANCA,
) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo de Wilson para uma proporcao, vetorizado.

    Espelha em Python o calculo de `sql/ranking_risco.sql`; a duplicacao e
    intencional e coberta por teste, para que a versao SQL (dashboard) e a
    versao pandas (analise) sejam verificaveis uma contra a outra.
    """
    sucessos = np.asarray(sucessos, dtype=float)
    total = np.asarray(total, dtype=float)
    p = np.divide(sucessos, total, out=np.zeros_like(total), where=total > 0)
    z2 = z * z
    denominador = 1.0 + z2 / total
    centro = (p + z2 / (2 * total)) / denominador
    margem = z * np.sqrt(p * (1 - p) / total + z2 / (4 * total**2)) / denominador
    return np.clip(centro - margem, 0.0, 1.0), np.clip(centro + margem, 0.0, 1.0)


def taxa_por_grupo(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Taxa de recall por nivel de uma variavel, com intervalo de Wilson."""
    agrupado = (
        df.assign(_alvo=alvo_binario(df))
        .groupby(coluna, observed=True)["_alvo"]
        .agg(n="size", n_recalls="sum")
        .reset_index()
    )
    agrupado["taxa_recall"] = agrupado["n_recalls"] / agrupado["n"]
    inferior, superior = intervalo_wilson(agrupado["n_recalls"], agrupado["n"])
    agrupado["ic_inferior"] = inferior
    agrupado["ic_superior"] = superior
    agrupado["amplitude_ic"] = superior - inferior
    return agrupado


def sobreposicao_de_intervalos(ranking: pd.DataFrame) -> bool:
    """Indica se o intervalo do 1o colocado cobre o do ultimo.

    Serve de teste rapido para a pergunta que o ranking de risco levanta: as
    posicoes sao estatisticamente distinguiveis, ou o ranking e ruido ordenado?
    """
    primeiro = ranking.iloc[0]
    ultimo = ranking.iloc[-1]
    return bool(primeiro["ic_inferior"] <= ultimo["ic_superior"])
