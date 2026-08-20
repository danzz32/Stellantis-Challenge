"""Testes da biblioteca de atributos derivados.

Alem da aritmetica de cada feature, dois testes guardam propriedades de projeto
do modulo: nenhuma feature pode depender de estatistica do conjunto (o que
vazaria informacao entre dobras da validacao cruzada), e `km_por_ano` precisa de
fato reduzir a colinearidade que motivou a sua criacao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stellantis_recall import config, features


class TestAritmetica:
    def test_km_por_ano(self, trusted: pd.DataFrame) -> None:
        esperado = trusted["km"] / (trusted["idade_veiculo"] + config.OFFSET_IDADE)
        pd.testing.assert_series_equal(
            features.km_por_ano(trusted), esperado, check_names=False
        )

    def test_reclamacoes_por_10k_km(self) -> None:
        df = pd.DataFrame({"reclamacoes": [3], "km": [30_000]})
        assert features.reclamacoes_por_10k_km(df).iloc[0] == pytest.approx(1.0)

    def test_indicadora_usa_o_limiar_da_analise(self) -> None:
        df = pd.DataFrame({"reclamacoes": [0, 2, 3, 13]})
        esperado = [False, False, True, True]
        assert features.muitas_reclamacoes(df).tolist() == esperado
        assert config.LIMIAR_RECLAMACOES == 3

    def test_veiculo_novo_nao_gera_divisao_por_zero(self) -> None:
        df = pd.DataFrame({"idade_veiculo": [0], "km": [1_000], "reclamacoes": [2]})
        assert np.isfinite(features.km_por_ano(df)).all()
        assert np.isfinite(features.reclamacoes_por_ano(df)).all()


class TestFaixas:
    def test_cobrem_todo_o_dominio(self, trusted: pd.DataFrame) -> None:
        com_features = features.adicionar_features(trusted)
        assert com_features["faixa_idade"].notna().all()
        assert com_features["faixa_km"].notna().all()

    def test_rotulos_pertencem_a_ordem_declarada(self, trusted: pd.DataFrame) -> None:
        com_features = features.adicionar_features(trusted)
        assert set(com_features["faixa_idade"]) <= set(features.ORDEM_FAIXA_IDADE)
        assert set(com_features["faixa_km"]) <= set(features.ORDEM_FAIXA_KM)

    def test_faixa_de_idade_e_monotonica(self, trusted: pd.DataFrame) -> None:
        """Idades maiores nunca podem cair em faixa anterior."""
        com_features = features.adicionar_features(trusted)
        ordem = {rotulo: i for i, rotulo in enumerate(features.ORDEM_FAIXA_IDADE)}
        posicao = com_features["faixa_idade"].map(ordem)
        assert posicao.groupby(com_features["idade_veiculo"]).nunique().eq(1).all()
        assert (
            posicao.groupby(com_features["idade_veiculo"]).first().is_monotonic_increasing
        )


class TestPropriedadesDeProjeto:
    def test_features_sao_linha_a_linha(self, trusted: pd.DataFrame) -> None:
        """Nenhuma feature pode depender das outras linhas do conjunto.

        Se alguma dependesse (media, quantil, residuo de regressao), ela veria o
        conjunto inteiro e vazaria informacao entre dobras da validacao cruzada.
        O teste calcula as features sobre uma metade do dado e verifica que os
        valores batem com os obtidos sobre o conjunto completo.
        """
        completo = features.adicionar_features(trusted)
        metade = features.adicionar_features(trusted.iloc[::2].copy())

        pd.testing.assert_frame_equal(
            metade[list(features.COLUNAS_DERIVADAS)],
            completo.iloc[::2][list(features.COLUNAS_DERIVADAS)],
        )

    def test_km_por_ano_reduz_a_colinearidade(self, trusted: pd.DataFrame) -> None:
        """A feature existe para separar intensidade de uso de tempo de vida.

        `km` correlaciona 0,947 com a idade. Se `km_por_ano` nao ficasse
        substancialmente mais ortogonal do que isso, o offset estaria mal
        calibrado e a feature nao cumpriria o proposito.
        """
        com_features = features.adicionar_features(trusted)
        bruta = com_features["km"].corr(com_features["idade_veiculo"])
        derivada = com_features["km_por_ano"].corr(com_features["idade_veiculo"])

        assert abs(bruta) > 0.9
        assert abs(derivada) < 0.2

    def test_derivadas_sao_finitas(self, trusted: pd.DataFrame) -> None:
        com_features = features.adicionar_features(trusted)
        numericas = com_features[list(features.DERIVADAS_NUMERICAS)]
        assert np.isfinite(numericas.to_numpy()).all()

    def test_nao_altera_o_dataframe_de_entrada(self, trusted: pd.DataFrame) -> None:
        antes = trusted.copy()
        features.adicionar_features(trusted)
        pd.testing.assert_frame_equal(trusted, antes)
