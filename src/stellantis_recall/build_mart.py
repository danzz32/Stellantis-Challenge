"""Camada `mart`: monta as tabelas prontas para consumo.

Duas familias de saida, com consumidores distintos:

* `features.parquet` -- uma linha por veiculo, com as derivadas de
  `features.py`. Entrada da modelagem.
* agregados (`perfil_por_modelo`, `evolucao_por_idade`, `ranking_risco`) --
  resultado das consultas de `sql/`, executadas pelo DuckDB lendo o Parquet
  `trusted` diretamente. Entrada do relatorio e do dashboard.

O motivo de os agregados serem materializados, e nao recalculados em cada
documento: o numero da pagina do PDF, o numero do relatorio e o numero do painel
passam a vir do mesmo arquivo. Divergencia entre eles deixa de ser possivel.

Uso pelo terminal:

    uv run python -m stellantis_recall.build_mart
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config, eda, features, transform
from .schemas import AgregadoRiscoSchema, FeaturesSchema

#: Consulta -> arquivo de destino na camada mart.
AGREGADOS: dict[str, Path] = {
    "perfil_por_modelo": config.MART_PERFIL_MODELO,
    "evolucao_por_idade": config.MART_EVOLUCAO_IDADE,
    "ranking_risco": config.MART_RANKING_RISCO,
}

#: Agregados que expoem taxa com intervalo de confianca.
AGREGADOS_COM_IC: frozenset[str] = frozenset({"ranking_risco"})


# --------------------------------------------------------------------------- #
# Construcao
# --------------------------------------------------------------------------- #


def construir_features(trusted: pd.DataFrame) -> pd.DataFrame:
    """Anexa as derivadas e valida contra `FeaturesSchema`."""
    return FeaturesSchema.validate(features.adicionar_features(trusted), lazy=True)


def construir_agregados(origem: Path | None = None) -> dict[str, pd.DataFrame]:
    """Executa as consultas de `sql/` sobre o Parquet `trusted`."""
    origem = origem or config.TRUSTED_VEICULOS
    con = eda.conectar_parquet(origem)
    try:
        resultados = {nome: eda.consultar(con, nome) for nome in AGREGADOS}
    finally:
        con.close()

    for nome in AGREGADOS_COM_IC:
        AgregadoRiscoSchema.validate(resultados[nome], lazy=True)

    return resultados


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResultadoMart:
    """Saida da etapa: a tabela de atributos e os agregados."""

    features: pd.DataFrame
    agregados: dict[str, pd.DataFrame]
    caminhos: dict[str, Path]


def executar(
    origem: Path | None = None,
    *,
    persistir: bool = True,
) -> ResultadoMart:
    """Constroi a camada `mart` a partir da `trusted` ja materializada."""
    origem = origem or config.TRUSTED_VEICULOS
    trusted = transform.carregar(origem)

    tabela_features = construir_features(trusted)
    agregados = construir_agregados(origem)

    caminhos: dict[str, Path] = {"features": config.MART_FEATURES, **AGREGADOS}
    if persistir:
        config.garantir_diretorios()
        tabela_features.to_parquet(config.MART_FEATURES, index=False)
        for nome, destino in AGREGADOS.items():
            agregados[nome].to_parquet(destino, index=False)

    return ResultadoMart(
        features=tabela_features,
        agregados=agregados,
        caminhos=caminhos,
    )


def carregar_features(caminho: Path | None = None) -> pd.DataFrame:
    """Le a tabela de atributos ja materializada."""
    caminho = caminho or config.MART_FEATURES
    if not caminho.is_file():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode `uv run recall-pipeline` para construi-la."
        )
    return pd.read_parquet(caminho)


def carregar_agregado(nome: str) -> pd.DataFrame:
    """Le um agregado do mart pelo nome da consulta que o originou."""
    if nome not in AGREGADOS:
        raise KeyError(f"Agregado '{nome}' desconhecido. Opcoes: {sorted(AGREGADOS)}.")
    caminho = AGREGADOS[nome]
    if not caminho.is_file():
        raise FileNotFoundError(
            f"{caminho} nao existe. Rode `uv run recall-pipeline` para construi-lo."
        )
    return pd.read_parquet(caminho)


def main() -> int:
    """Constroi a camada mart e imprime um resumo do que foi gravado."""
    resultado = executar()

    print(
        f"features: {len(resultado.features)} linhas x "
        f"{resultado.features.shape[1]} colunas -> {config.MART_FEATURES.name}"
    )
    for nome, tabela in resultado.agregados.items():
        print(f"{nome}: {len(tabela)} linhas -> {AGREGADOS[nome].name}")

    print()
    print("Derivadas numericas:")
    print(features.resumo_derivadas(resultado.features).round(2).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
