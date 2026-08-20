"""Testes da camada de analise exploratoria.

O teste central deste arquivo e o de paridade do intervalo de Wilson. O calculo
existe duas vezes no projeto -- em `sql/ranking_risco.sql`, que alimenta o
dashboard, e em `eda.intervalo_wilson`, que alimenta a analise. A duplicacao e
deliberada, mas so se justifica se uma implementacao verificar a outra: sem este
teste, as duas versoes podem divergir em silencio e o relatorio passa a discordar
do painel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stellantis_recall import config, eda, transform


@pytest.fixture(scope="module")
def conexao(trusted: pd.DataFrame):
    con = eda.conectar(trusted)
    yield con
    con.close()


class TestParidadeWilson:
    def test_sql_e_python_concordam_no_ranking(
        self, conexao, trusted: pd.DataFrame
    ) -> None:
        sql = eda.consultar(conexao, "ranking_risco").sort_values("modelo")
        python = eda.taxa_por_grupo(trusted, "modelo").sort_values("modelo")

        np.testing.assert_allclose(
            sql["taxa_recall"].to_numpy(), python["taxa_recall"].to_numpy(), rtol=1e-9
        )
        np.testing.assert_allclose(
            sql["ic_inferior"].to_numpy(), python["ic_inferior"].to_numpy(), rtol=1e-9
        )
        np.testing.assert_allclose(
            sql["ic_superior"].to_numpy(), python["ic_superior"].to_numpy(), rtol=1e-9
        )

    @pytest.mark.parametrize(
        ("sucessos", "total"),
        [(0, 10), (10, 10), (1, 1), (5, 100), (50, 100), (99, 100), (3, 7)],
    )
    def test_casos_de_borda(self, conexao, sucessos: int, total: int) -> None:
        """Wilson foi escolhido por nao degenerar em p = 0 e p = 1; verificar isso."""
        inferior, superior = eda.intervalo_wilson(
            np.array([sucessos]), np.array([total])
        )
        consulta = conexao.execute(
            """
            select
                ($z * $z)                              as z2,
                ($s * 1.0 / $n)                        as p
            """,
            {"z": config.Z_NIVEL_CONFIANCA, "s": sucessos, "n": total},
        ).df()

        p = consulta["p"].iloc[0]
        z2 = consulta["z2"].iloc[0]
        centro = (p + z2 / (2 * total)) / (1 + z2 / total)
        margem = (
            config.Z_NIVEL_CONFIANCA
            * np.sqrt(p * (1 - p) / total + z2 / (4 * total**2))
            / (1 + z2 / total)
        )

        assert inferior[0] == pytest.approx(max(centro - margem, 0.0))
        assert superior[0] == pytest.approx(min(centro + margem, 1.0))

    def test_intervalo_contem_a_estimativa(self, trusted: pd.DataFrame) -> None:
        taxas = eda.taxa_por_grupo(trusted, "reclamacoes")
        assert (taxas["ic_inferior"] <= taxas["taxa_recall"]).all()
        assert (taxas["taxa_recall"] <= taxas["ic_superior"]).all()

    def test_intervalo_encolhe_com_a_amostra(self) -> None:
        pequeno = eda.intervalo_wilson(np.array([5]), np.array([10]))
        grande = eda.intervalo_wilson(np.array([500]), np.array([1_000]))
        assert (grande[1] - grande[0]) < (pequeno[1] - pequeno[0])


class TestVisaoCanonica:
    def test_alvo_e_booleano_partindo_do_texto(self, bruto: pd.DataFrame) -> None:
        con = eda.conectar(bruto)
        try:
            tipo = con.execute(
                "select typeof(recall) as t from veiculos limit 1"
            ).df()["t"].iloc[0]
        finally:
            con.close()
        assert tipo == "BOOLEAN"

    def test_mesma_consulta_serve_as_duas_camadas(
        self, bruto: pd.DataFrame, trusted: pd.DataFrame
    ) -> None:
        """Bruto e trusted diferem por uma linha; as agregacoes devem ser proximas."""
        con_bruto = eda.conectar(bruto)
        con_trusted = eda.conectar(trusted)
        try:
            de_bruto = eda.consultar(con_bruto, "perfil_por_modelo")
            de_trusted = eda.consultar(con_trusted, "perfil_por_modelo")
        finally:
            con_bruto.close()
            con_trusted.close()

        assert list(de_bruto.columns) == list(de_trusted.columns)
        assert de_bruto["n_veiculos"].sum() == 500
        assert de_trusted["n_veiculos"].sum() == 499


class TestEstatisticas:
    def test_vif_detecta_a_colinearidade_conhecida(self, trusted: pd.DataFrame) -> None:
        vif = eda.fator_inflacao_variancia(trusted).set_index("variavel")["vif"]
        assert vif["idade_veiculo"] > 5
        assert vif["km"] > 5
        assert vif["reclamacoes"] < 3

    def test_modelo_nao_associa_ao_alvo(self, trusted: pd.DataFrame) -> None:
        """Achado da Parte 1 que sustenta a ressalva do ranking no dashboard."""
        teste = eda.qui_quadrado_com_alvo(
            trusted.assign(recall=transform.rotular_alvo(trusted["recall"])), "modelo"
        )
        assert not teste.significante
        assert teste.v_cramer < 0.2

    def test_ranking_de_modelos_tem_intervalos_sobrepostos(
        self, trusted: pd.DataFrame
    ) -> None:
        ranking = eda.taxa_por_grupo(trusted, "modelo").sort_values(
            "taxa_recall", ascending=False
        )
        assert eda.sobreposicao_de_intervalos(ranking)

    def test_duplicatas_detecta_a_linha_conhecida(self, bruto: pd.DataFrame) -> None:
        assert len(eda.duplicatas(bruto)) == 1
