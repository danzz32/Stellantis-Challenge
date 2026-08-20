"""Testes dos contratos de dados.

Um contrato so vale se rejeitar o que deve rejeitar. Estes testes verificam as
duas direcoes: o dado conforme passa, e cada tipo de violacao e efetivamente
barrado.
"""

from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from stellantis_recall import config, ingest
from stellantis_recall.schemas import COLUNAS_TRUSTED, RawSchema, TrustedSchema


class TestRawSchema:
    def test_aceita_a_base_de_origem(self, bruto: pd.DataFrame) -> None:
        assert ingest.validar(bruto).conforme

    def test_aceita_amostra_valida(self, amostra_valida: pd.DataFrame) -> None:
        assert ingest.validar(amostra_valida).conforme

    def test_rejeita_modelo_fora_do_catalogo(
        self, amostra_valida: pd.DataFrame
    ) -> None:
        amostra_valida.loc[0, "modelo"] = "Uno"
        relatorio = ingest.validar(amostra_valida)
        assert not relatorio.conforme
        assert "modelo" in set(relatorio.falhas["column"])

    def test_rejeita_km_negativo(self, amostra_valida: pd.DataFrame) -> None:
        amostra_valida.loc[1, "km"] = -1
        assert not ingest.validar(amostra_valida).conforme

    def test_rejeita_idade_implausivel(self, amostra_valida: pd.DataFrame) -> None:
        amostra_valida.loc[2, "idade_veiculo"] = config.IDADE_MAX + 1
        assert not ingest.validar(amostra_valida).conforme

    def test_rejeita_rotulo_de_alvo_desconhecido(
        self, amostra_valida: pd.DataFrame
    ) -> None:
        # "Nao" sem til: o caso classico de arquivo salvo em codificacao errada.
        amostra_valida.loc[0, "recall"] = "Nao"
        relatorio = ingest.validar(amostra_valida)
        assert not relatorio.conforme
        assert "recall" in set(relatorio.falhas["column"])

    def test_rejeita_nulo(self, amostra_valida: pd.DataFrame) -> None:
        amostra_valida.loc[0, "reclamacoes"] = None
        assert not ingest.validar(amostra_valida).conforme

    def test_relatorio_acumula_multiplas_violacoes(
        self, amostra_valida: pd.DataFrame
    ) -> None:
        """Validacao preguicosa: o diagnostico precisa ser completo, nao o primeiro erro."""
        amostra_valida.loc[0, "modelo"] = "Uno"
        amostra_valida.loc[1, "km"] = -1
        amostra_valida.loc[2, "recall"] = "Talvez"

        relatorio = ingest.validar(amostra_valida)

        assert not relatorio.conforme
        assert {"modelo", "km", "recall"} <= set(relatorio.falhas["column"])


class TestTrustedSchema:
    def test_aceita_a_camada_trusted(self, trusted: pd.DataFrame) -> None:
        TrustedSchema.validate(trusted, lazy=True)

    def test_alvo_e_booleano(self, trusted: pd.DataFrame) -> None:
        assert trusted[config.COLUNA_ALVO].dtype == bool

    def test_ordem_das_colunas_e_canonica(self, trusted: pd.DataFrame) -> None:
        assert tuple(trusted.columns) == COLUNAS_TRUSTED

    def test_rejeita_alvo_textual(self, trusted: pd.DataFrame) -> None:
        """O contrato trusted nao pode aceitar o alvo na codificacao de origem."""
        regressao = trusted.assign(
            recall=trusted[config.COLUNA_ALVO].map(
                {True: config.ROTULO_POSITIVO, False: config.ROTULO_NEGATIVO}
            )
        )
        with pytest.raises(pandera.errors.SchemaErrors):
            TrustedSchema.validate(regressao, lazy=True)

    def test_rejeita_coluna_inesperada(self, trusted: pd.DataFrame) -> None:
        with pytest.raises(pandera.errors.SchemaErrors):
            TrustedSchema.validate(trusted.assign(extra=1), lazy=True)
