"""Testes da camada trusted.

Cada regra de limpeza aqui responde a um achado da Parte 1. O teste amarra a
regra ao achado: se alguem trocar a regra sem revisar o diagnostico, quebra.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stellantis_recall import config, transform


class TestDeduplicacao:
    def test_remove_a_duplicata_conhecida(self, bruto: pd.DataFrame) -> None:
        """A base tem exatamente uma linha repetida; 500 -> 499."""
        limpo, n_removidas = transform.remover_duplicatas(bruto)
        assert n_removidas == 1
        assert len(limpo) == len(bruto) - 1

    def test_resultado_nao_tem_duplicata(self, trusted: pd.DataFrame) -> None:
        assert not trusted.duplicated().any()

    def test_preserva_a_primeira_ocorrencia(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        limpo, n = transform.remover_duplicatas(df)
        assert n == 1
        assert limpo.equals(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))

    def test_e_idempotente(self, bruto: pd.DataFrame) -> None:
        uma_vez, _ = transform.remover_duplicatas(bruto)
        duas_vezes, n = transform.remover_duplicatas(uma_vez)
        assert n == 0
        assert len(duas_vezes) == len(uma_vez)


class TestBinarizacaoDoAlvo:
    def test_mapeia_os_dois_rotulos(self) -> None:
        serie = pd.Series([config.ROTULO_POSITIVO, config.ROTULO_NEGATIVO])
        assert transform.binarizar_alvo(serie).tolist() == [True, False]

    def test_rejeita_rotulo_desconhecido(self) -> None:
        with pytest.raises(ValueError, match="Valores inesperados"):
            transform.binarizar_alvo(pd.Series(["Sim", "Talvez"]))

    def test_rejeita_nao_sem_til(self) -> None:
        """Guarda contra arquivo salvo em codificacao errada."""
        with pytest.raises(ValueError):
            transform.binarizar_alvo(pd.Series(["Sim", "Nao"]))

    def test_rotular_e_o_inverso_de_binarizar(self, bruto: pd.DataFrame) -> None:
        original = bruto[config.COLUNA_ALVO]
        ida_e_volta = transform.rotular_alvo(transform.binarizar_alvo(original))
        pd.testing.assert_series_equal(
            ida_e_volta, original, check_names=False, check_dtype=False
        )


class TestTransformacaoCompleta:
    def test_preserva_a_proporcao_do_alvo(
        self, bruto: pd.DataFrame, trusted: pd.DataFrame
    ) -> None:
        """A limpeza remove 1 linha em 500; a taxa nao pode mudar de forma material."""
        taxa_origem = bruto[config.COLUNA_ALVO].eq(config.ROTULO_POSITIVO).mean()
        assert trusted[config.COLUNA_ALVO].mean() == pytest.approx(
            taxa_origem, abs=0.005
        )

    def test_nao_introduz_nulos(self, trusted: pd.DataFrame) -> None:
        assert trusted.isna().sum().sum() == 0

    def test_e_idempotente(self, trusted: pd.DataFrame) -> None:
        """Aplicar a limpeza sobre o dado ja limpo nao pode altera-lo."""
        limpo, n = transform.remover_duplicatas(trusted)
        assert n == 0
        pd.testing.assert_frame_equal(transform.tipar(limpo), trusted)

    def test_nao_persiste_quando_pedido(self) -> None:
        resultado = transform.executar(persistir=False)
        assert resultado.destino is None
        assert len(resultado.df) == 499


class TestRelatorioQualidade:
    def test_registra_a_duplicata_removida(
        self, bruto: pd.DataFrame, trusted: pd.DataFrame
    ) -> None:
        relatorio = transform.relatorio_qualidade(bruto, trusted, n_removidas=1)
        linha = relatorio.set_index("verificacao").loc["duplicatas removidas", "valor"]
        assert linha == "1"

    def test_coluna_valor_e_homogenea(
        self, bruto: pd.DataFrame, trusted: pd.DataFrame
    ) -> None:
        """Coluna heterogenea nao tem representacao em Parquet."""
        relatorio = transform.relatorio_qualidade(bruto, trusted, n_removidas=1)
        assert relatorio["valor"].map(type).eq(str).all()
