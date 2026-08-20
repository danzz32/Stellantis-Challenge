"""Testes da Parte 2.

Os testes evitam rodar `train.executar()`, que ajusta 300 pipelines e leva
cerca de um minuto. Verificam as pecas: montagem do pre-processamento, regra de
selecao e a correcao do teste pareado -- que e onde um erro passaria despercebido
e contaminaria a justificativa da escolha.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.pipeline import Pipeline

from stellantis_recall import config
from stellantis_recall.modeling import train


class TestPreprocessamento:
    @pytest.mark.parametrize("conjunto", list(train.CONJUNTOS_FEATURES))
    def test_monta_para_todo_conjunto(self, conjunto: str) -> None:
        colunas = train.CONJUNTOS_FEATURES[conjunto]
        for sensivel in (True, False):
            preproc = train.construir_preprocessador(
                colunas, sensivel_a_escala=sensivel
            )
            usadas = {c for _, _, cols in preproc.transformers for c in cols}
            assert usadas == set(colunas)

    def test_nao_cria_etapa_vazia(self) -> None:
        """`sem_modelo` nao tem categoricas nem booleanas."""
        preproc = train.construir_preprocessador(
            train.CONJUNTOS_FEATURES["sem_modelo"], sensivel_a_escala=True
        )
        nomes = {nome for nome, _, _ in preproc.transformers}
        assert nomes == {"simetricas"}

    def test_modelo_linear_descarta_uma_dummy(self) -> None:
        """Sob colinearidade ja severa, manter as 9 dummies agrava a matriz."""
        linear = train.construir_preprocessador(
            train.CONJUNTOS_FEATURES["originais"], sensivel_a_escala=True
        )
        arvore = train.construir_preprocessador(
            train.CONJUNTOS_FEATURES["originais"], sensivel_a_escala=False
        )
        assert dict(
            (n, t) for n, t, _ in linear.transformers
        )["categoricas"].drop == "first"
        assert dict(
            (n, t) for n, t, _ in arvore.transformers
        )["categoricas"].drop is None

    def test_arvores_nao_escalam(self) -> None:
        arvore = train.construir_preprocessador(
            train.CONJUNTOS_FEATURES["com_derivadas"], sensivel_a_escala=False
        )
        trilhas = {nome: t for nome, t, _ in arvore.transformers}
        assert trilhas["simetricas"] == "passthrough"
        assert trilhas["assimetricas"] == "passthrough"


class TestPipelines:
    @pytest.mark.parametrize("nome_modelo", list(train.construir_modelos()))
    @pytest.mark.parametrize("conjunto", list(train.CONJUNTOS_FEATURES))
    def test_ajusta_e_prediz(
        self, nome_modelo: str, conjunto: str, trusted: pd.DataFrame
    ) -> None:
        from stellantis_recall import features

        dados = features.adicionar_features(trusted)
        colunas = train.CONJUNTOS_FEATURES[conjunto]
        y = dados[config.COLUNA_ALVO].to_numpy(dtype=int)

        pipeline = train.montar_pipeline(nome_modelo, colunas)
        assert isinstance(pipeline, Pipeline)

        pipeline.fit(dados[list(colunas)], y)
        prob = pipeline.predict_proba(dados[list(colunas)])[:, 1]
        assert prob.shape == (len(dados),)
        assert ((prob >= 0) & (prob <= 1)).all()


class TestNadeauBengio:
    def test_e_mais_conservador_que_o_ingenuo(self) -> None:
        """A correcao existe para nao declarar significancia por artefato do protocolo.

        A diferenca simulada -- pequena, positiva em media, com ruido -- reproduz
        o que aparece entre dois modelos razoaveis sobre as mesmas dobras.
        """
        gerador = np.random.default_rng(config.SEED)
        a = gerador.normal(0.80, 0.04, 25)
        b = a - 0.008 + gerador.normal(0.0, 0.005, 25)

        _, p_ingenuo = stats.ttest_rel(a, b)
        _, p_corrigido = train.teste_pareado_corrigido(a, b)

        assert p_ingenuo < 0.05
        assert p_corrigido > p_ingenuo

    def test_diferenca_nula_nao_e_significante(self) -> None:
        a = np.linspace(0.75, 0.85, 25)
        _, p = train.teste_pareado_corrigido(a, a.copy())
        assert p == pytest.approx(1.0)

    def test_diferenca_grande_continua_significante(self) -> None:
        """A correcao nao pode apagar a distancia para o baseline."""
        gerador = np.random.default_rng(config.SEED)
        a = gerador.normal(0.79, 0.04, 25)
        b = np.full(25, 0.50)
        _, p = train.teste_pareado_corrigido(a, b)
        assert p < 0.001

    def test_sinal_da_estatistica_segue_a_diferenca(self) -> None:
        gerador = np.random.default_rng(config.SEED)
        a = gerador.normal(0.80, 0.04, 25)
        b = a + 0.05 + gerador.normal(0.0, 0.005, 25)
        t, _ = train.teste_pareado_corrigido(a, b)
        assert t < 0

    def test_diferenca_deterministica_nao_vira_indistinguivel(self) -> None:
        """Variancia nula com media nao nula: diferenca certa, nao ausente."""
        a = np.linspace(0.75, 0.85, 25)
        t, p = train.teste_pareado_corrigido(a, a - 0.05)
        assert t == np.inf
        assert p == 0.0


class TestSelecao:
    @staticmethod
    def _tabela(linhas: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(linhas)

    def test_prefere_parcimonia_dentro_de_um_erro_padrao(self) -> None:
        """Diferenca de milesimos nao justifica o dobro de atributos."""
        tabela = self._tabela(
            [
                {"modelo": "a", "conjunto": "grande", "n_features": 8,
                 "roc_auc": 0.7950, "roc_auc_desvio": 0.045},
                {"modelo": "b", "conjunto": "pequeno", "n_features": 3,
                 "roc_auc": 0.7940, "roc_auc_desvio": 0.045},
            ]
        )
        assert train.selecionar(tabela)["conjunto"] == "pequeno"

    def test_nao_sacrifica_desempenho_real_por_parcimonia(self) -> None:
        """Fora da faixa de um erro padrao, a metrica volta a mandar."""
        tabela = self._tabela(
            [
                {"modelo": "a", "conjunto": "grande", "n_features": 8,
                 "roc_auc": 0.8500, "roc_auc_desvio": 0.010},
                {"modelo": "b", "conjunto": "pequeno", "n_features": 3,
                 "roc_auc": 0.7000, "roc_auc_desvio": 0.010},
            ]
        )
        assert train.selecionar(tabela)["conjunto"] == "grande"

    def test_nunca_seleciona_o_baseline(self) -> None:
        tabela = self._tabela(
            [
                {"modelo": "baseline", "conjunto": "x", "n_features": 1,
                 "roc_auc": 0.7960, "roc_auc_desvio": 0.0},
                {"modelo": "regressao_logistica", "conjunto": "x", "n_features": 3,
                 "roc_auc": 0.7940, "roc_auc_desvio": 0.045},
            ]
        )
        assert train.selecionar(tabela)["modelo"] == "regressao_logistica"


class TestMetricas:
    def test_predicao_perfeita(self) -> None:
        y = np.array([0, 0, 1, 1])
        metricas = train._metricas_da_dobra(y, np.array([0.1, 0.2, 0.8, 0.9]))
        assert metricas["accuracy"] == 1.0
        assert metricas["roc_auc"] == 1.0
        assert metricas["brier"] < 0.05

    def test_baseline_constante_nao_quebra(self) -> None:
        """Sem positivos preditos, a precisao e indefinida e deve virar zero."""
        y = np.array([0, 0, 1, 1])
        metricas = train._metricas_da_dobra(y, np.array([0.2, 0.2, 0.2, 0.2]))
        assert metricas["precision"] == 0.0
        assert metricas["recall"] == 0.0
        assert metricas["roc_auc"] == 0.5
