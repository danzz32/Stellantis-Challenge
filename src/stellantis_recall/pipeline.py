"""Encadeia as etapas de dados: ingestao -> trusted -> mart.

Um comando reconstroi tudo o que e derivado a partir da unica entrada imutavel
do projeto, a planilha em `data/raw/`. Nenhum artefato gerado precisa ser
versionado, e nenhum passo depende de ter sido executado manualmente antes.

Uso pelo terminal:

    uv run recall-pipeline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from . import build_mart, config, transform


@dataclass(frozen=True, slots=True)
class ResultadoPipeline:
    """Sumario da execucao completa."""

    n_linhas_trusted: int
    n_duplicatas_removidas: int
    n_colunas_features: int
    n_agregados: int


def executar(*, persistir: bool = True) -> ResultadoPipeline:
    """Roda as duas etapas de dados em ordem."""
    resultado_transform = transform.executar(persistir=persistir)
    resultado_mart = build_mart.executar(persistir=persistir)

    return ResultadoPipeline(
        n_linhas_trusted=len(resultado_transform.df),
        n_duplicatas_removidas=resultado_transform.n_removidas,
        n_colunas_features=resultado_mart.features.shape[1],
        n_agregados=len(resultado_mart.agregados),
    )


def main() -> int:
    """Ponto de entrada `recall-pipeline`."""
    analisador = argparse.ArgumentParser(
        prog="recall-pipeline",
        description="Reconstroi as camadas trusted e mart a partir de data/raw/.",
    )
    analisador.add_argument(
        "--sem-persistir",
        action="store_true",
        help="executa as etapas em memoria, sem gravar arquivos (util em teste).",
    )
    argumentos = analisador.parse_args()

    resultado = executar(persistir=not argumentos.sem_persistir)

    print("Pipeline concluido.")
    print(f"  trusted            : {resultado.n_linhas_trusted} linhas")
    print(f"  duplicatas removidas: {resultado.n_duplicatas_removidas}")
    print(f"  features           : {resultado.n_colunas_features} colunas")
    print(f"  agregados          : {resultado.n_agregados}")
    print(f"  destino            : {config.DATA_DIR}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
