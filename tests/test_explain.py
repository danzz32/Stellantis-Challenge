"""Testes da Parte 4.

O teste mais importante deste arquivo e o de mascaramento por colinearidade:
ele constroi um caso com resposta conhecida -- duas copias quase identicas de
uma variavel util -- e verifica que a permutacao individual de fato subestima
a importancia enquanto a permutacao em bloco a recupera. Sem esse teste, a
correcao aplicada na Parte 4 seria uma afirmacao sem verificacao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stellantis_recall import config
from stellantis_recall.modeling import explain


@pytest.fixture
def dados_colineares() -> tuple[pd.DataFrame, np.ndarray]:
    """Duas copias quase identicas de um preditor util, mais um ruido puro.

    `a` e `b` correlacionam acima de 0,99 e ambas carregam o sinal; `ruido` nao
    carrega nada. O resultado esperado e conhecido de antemao.
    """
    gerador = np.random.default_rng(config.SEED)
    n = 600
    a = gerador.normal(size=n)
    b = a + gerador.normal(0, 0.05, n)
    ruido = gerador.normal(size=n)

    logito = 1.5 * a
    y = (gerador.uniform(size=n) < 1 / (1 + np.exp(-logito))).astype(int)

    return pd.DataFrame({"a": a, "b": b, "ruido": ruido}), y


@pytest.fixture
def pipeline_linear() -> Pipeline:
    from sklearn.compose import ColumnTransformer

    return Pipeline(
        [
            (
                "preproc",
                ColumnTransformer(
                    [("simetricas", StandardScaler(), ["a", "b", "ruido"])],
                    verbose_feature_names_out=False,
                ),
            ),
            ("modelo", LogisticRegression(max_iter=1_000, random_state=config.SEED)),
        ]
    )


class TestPermutacao:
    def test_preserva_a_estrutura_interna_do_grupo(self) -> None:
        """A mesma reordenacao vale para todas as colunas do bloco."""
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40], "c": [0, 0, 1, 1]})
        gerador = np.random.default_rng(config.SEED)

        embaralhado = explain._permutar(X, ("a", "b"), gerador)

        # a e b continuam pareadas: b permanece igual a 10 * a.
        assert (embaralhado["b"] == embaralhado["a"] * 10).all()
        # A coluna fora do grupo nao e tocada.
        pd.testing.assert_series_equal(embaralhado["c"], X["c"])

    def test_nao_altera_o_dataframe_de_entrada(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
        antes = X.copy()
        explain._permutar(X, ("a",), np.random.default_rng(config.SEED))
        pd.testing.assert_frame_equal(X, antes)


class TestMascaramentoPorColinearidade:
    def test_permutacao_individual_subestima_variaveis_colineares(
        self, dados_colineares, pipeline_linear
    ) -> None:
        """O nucleo da Parte 4: uma variavel util parece irrelevante sozinha.

        Com `a` e `b` quase identicas, permutar apenas uma delas quase nao
        degrada o modelo -- a outra cobre a ausencia. Permutadas em bloco, a
        contribuicao real aparece.
        """
        X, y = dados_colineares

        individual = explain.resumir_importancia(
            explain.importancia_por_permutacao(
                pipeline_linear, X, y, {"a": ("a",), "b": ("b",), "ruido": ("ruido",)}
            )
        ).set_index("grupo")["queda_auc"]

        bloco = explain.resumir_importancia(
            explain.importancia_por_permutacao(
                pipeline_linear, X, y, {"a_e_b": ("a", "b"), "ruido": ("ruido",)}
            )
        ).set_index("grupo")["queda_auc"]

        # O bloco supera com folga a soma das partes -- efeito de mascaramento.
        assert bloco["a_e_b"] > (individual["a"] + individual["b"])

    def test_variavel_sem_sinal_nao_e_relevante(
        self, dados_colineares, pipeline_linear
    ) -> None:
        X, y = dados_colineares
        resumo = explain.resumir_importancia(
            explain.importancia_por_permutacao(
                pipeline_linear, X, y, {"a_e_b": ("a", "b"), "ruido": ("ruido",)}
            )
        ).set_index("grupo")

        assert not bool(resumo.loc["ruido", "relevante"])
        assert bool(resumo.loc["a_e_b", "relevante"])

    def test_intervalo_contem_a_estimativa(
        self, dados_colineares, pipeline_linear
    ) -> None:
        X, y = dados_colineares
        resumo = explain.resumir_importancia(
            explain.importancia_por_permutacao(
                pipeline_linear, X, y, {"a": ("a",), "ruido": ("ruido",)}
            )
        )
        assert (resumo["ic_inferior"] <= resumo["queda_auc"]).all()
        assert (resumo["queda_auc"] <= resumo["ic_superior"]).all()


class TestCoeficientes:
    def test_converte_para_a_escala_natural(
        self, dados_colineares, pipeline_linear
    ) -> None:
        """coeficiente_natural = coeficiente_padronizado / desvio-padrao."""
        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        tabela = explain.coeficientes(pipeline_linear).set_index("variavel")

        for variavel in ("a", "b", "ruido"):
            esperado = (
                tabela.loc[variavel, "coeficiente_padronizado"]
                / X[variavel].std(ddof=0)
            )
            assert tabela.loc[variavel, "coeficiente_natural"] == pytest.approx(
                esperado, rel=1e-6
            )

    def test_razao_de_chances_e_a_exponencial(
        self, dados_colineares, pipeline_linear
    ) -> None:
        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        tabela = explain.coeficientes(pipeline_linear)

        np.testing.assert_allclose(
            tabela["razao_chances_por_desvio"],
            np.exp(tabela["coeficiente_padronizado"]),
            rtol=1e-9,
        )

    def test_ordena_por_magnitude(self, dados_colineares, pipeline_linear) -> None:
        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        tabela = explain.coeficientes(pipeline_linear)
        assert tabela["magnitude"].is_monotonic_decreasing

    def test_recusa_modelo_sem_coeficientes(self, dados_colineares) -> None:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier

        X, y = dados_colineares
        arvore = Pipeline(
            [
                (
                    "preproc",
                    ColumnTransformer(
                        [("num", "passthrough", ["a", "b", "ruido"])],
                        verbose_feature_names_out=False,
                    ),
                ),
                ("modelo", RandomForestClassifier(n_estimators=10, random_state=1)),
            ]
        ).fit(X, y)

        with pytest.raises(TypeError, match="nao expoe coeficientes"):
            explain.coeficientes(arvore)


class TestUnidadesComunicaveis:
    def test_cobre_todas_as_variaveis_do_projeto(self) -> None:
        from stellantis_recall.modeling import train

        usadas = {c for cols in train.CONJUNTOS_FEATURES.values() for c in cols}
        numericas = usadas - {"modelo"}
        assert numericas <= set(explain.UNIDADES_COMUNICAVEIS)

    def test_escala_a_razao_de_chances(
        self, dados_colineares, pipeline_linear, monkeypatch
    ) -> None:
        """Razao por 10 unidades e a razao por unidade elevada a 10."""
        monkeypatch.setitem(explain.UNIDADES_COMUNICAVEIS, "a", (10.0, "por 10"))

        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        tabela = explain.coeficientes(pipeline_linear).set_index("variavel")

        assert tabela.loc["a", "razao_chances_comunicavel"] == pytest.approx(
            tabela.loc["a", "razao_chances_por_unidade"] ** 10
        )


class TestContribuicoesIndividuais:
    def test_reconstroi_a_predicao(self, dados_colineares, pipeline_linear) -> None:
        """A decomposição é exata: contribuições + referência devolvem a predição.

        Este é o teste que sustenta o score de veículo no painel interativo. Se
        a soma não fechasse, o painel exibiria uma explicação que não
        corresponde ao número exibido ao lado dela.
        """
        X, y = dados_colineares
        pipeline_linear.fit(X, y)

        contribuicoes, referencia = explain.contribuicoes_individuais(
            pipeline_linear, X
        )
        logito = contribuicoes.sum(axis=1).to_numpy() + referencia
        reconstruido = 1.0 / (1.0 + np.exp(-logito))

        np.testing.assert_allclose(
            reconstruido, pipeline_linear.predict_proba(X)[:, 1], rtol=1e-9
        )

    def test_sinal_acompanha_o_efeito(self, dados_colineares, pipeline_linear) -> None:
        """Valor acima da média em preditor positivo empurra para o risco."""
        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        contribuicoes, _ = explain.contribuicoes_individuais(pipeline_linear, X)

        acima_da_media = X["a"] > X["a"].mean()
        assert (contribuicoes.loc[acima_da_media, "a"] > 0).all()
        assert (contribuicoes.loc[~acima_da_media, "a"] < 0).all()

    def test_recusa_modelo_sem_coeficientes(self, dados_colineares) -> None:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier

        X, y = dados_colineares
        arvore = Pipeline(
            [
                (
                    "preproc",
                    ColumnTransformer(
                        [("num", "passthrough", ["a", "b", "ruido"])],
                        verbose_feature_names_out=False,
                    ),
                ),
                ("modelo", RandomForestClassifier(n_estimators=10, random_state=1)),
            ]
        ).fit(X, y)

        with pytest.raises(TypeError, match="nao expoe coeficientes"):
            explain.contribuicoes_individuais(arvore, X)


class TestDependenciaParcial:
    def test_produz_curva_para_cada_variavel(
        self, dados_colineares, pipeline_linear
    ) -> None:
        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        curva = explain.dependencia_parcial(pipeline_linear, X, ("a", "ruido"))

        assert set(curva["variavel"]) == {"a", "ruido"}
        assert curva["risco_previsto"].between(0, 1).all()

    def test_risco_cresce_com_o_preditor_positivo(
        self, dados_colineares, pipeline_linear
    ) -> None:
        X, y = dados_colineares
        pipeline_linear.fit(X, y)
        curva = explain.dependencia_parcial(pipeline_linear, X, ("a",))
        assert curva["risco_previsto"].is_monotonic_increasing
