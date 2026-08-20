"""Testes da Parte 3.

O foco esta na logica de custo e limiar, que e onde um erro seria mais caro:
uma inversao de sinal na funcao de custo produziria um limiar aparentemente
razoavel e uma recomendacao operacional errada, sem quebrar nada.

Os testes usam probabilidades sinteticas construidas para ter resposta conhecida,
e nao as predicoes reais -- assim a propriedade verificada e a do algoritmo, nao
a do dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stellantis_recall import config
from stellantis_recall.modeling import evaluate


@pytest.fixture
def oof_sintetico() -> pd.DataFrame:
    """Predicoes fora da amostra com separacao imperfeita, em 3 repeticoes.

    A sobreposicao e proposital: com separacao perfeita, todo limiar entre as
    duas nuvens seria otimo e os testes de limiar perderiam sentido.
    """
    gerador = np.random.default_rng(config.SEED)
    n = 200
    y = np.repeat([0, 1], n // 2)

    quadros = []
    for repeticao in range(3):
        prob = np.where(
            y == 1,
            gerador.beta(6, 3, n),
            gerador.beta(3, 6, n),
        )
        quadros.append(
            pd.DataFrame(
                {
                    "repeticao": repeticao,
                    "indice": np.arange(n),
                    "y_verdadeiro": y,
                    "probabilidade": prob,
                }
            )
        )
    return pd.concat(quadros, ignore_index=True)


class TestCusto:
    def test_pesa_falso_negativo_pela_razao(self) -> None:
        y = np.array([1, 1, 0, 0])
        prob = np.array([0.9, 0.1, 0.9, 0.1])  # 1 FN e 1 FP no limiar 0,5

        assert evaluate.custo_esperado(y, prob, 0.5, razao=1.0) == 2.0
        assert evaluate.custo_esperado(y, prob, 0.5, razao=3.0) == 4.0
        assert evaluate.custo_esperado(y, prob, 0.5, razao=10.0) == 11.0

    def test_classificacao_perfeita_custa_zero(self) -> None:
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert evaluate.custo_esperado(y, prob, 0.5, razao=3.0) == 0.0

    def test_limiar_alto_troca_falso_positivo_por_falso_negativo(self) -> None:
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.4, 0.45, 0.55, 0.6])
        # Limiar 0,0 sinaliza tudo: 2 FP, 0 FN.
        assert evaluate.custo_esperado(y, prob, 0.0, razao=1.0) == 2.0
        # Limiar 1,0 nao sinaliza nada: 0 FP, 2 FN.
        assert evaluate.custo_esperado(y, prob, 1.0, razao=1.0) == 2.0
        assert evaluate.custo_esperado(y, prob, 1.0, razao=5.0) == 10.0


class TestLimiarOtimo:
    def test_cai_quando_o_falso_negativo_encarece(
        self, oof_sintetico: pd.DataFrame
    ) -> None:
        """Propriedade central: custo maior de FN empurra o limiar para baixo."""
        y = oof_sintetico.query("repeticao == 0")["y_verdadeiro"].to_numpy()
        prob = oof_sintetico.query("repeticao == 0")["probabilidade"].to_numpy()

        limiares = [evaluate.limiar_otimo(y, prob, r) for r in (1.0, 3.0, 10.0, 50.0)]
        assert limiares == sorted(limiares, reverse=True)

    def test_razao_extrema_sinaliza_quase_tudo(
        self, oof_sintetico: pd.DataFrame
    ) -> None:
        y = oof_sintetico.query("repeticao == 0")["y_verdadeiro"].to_numpy()
        prob = oof_sintetico.query("repeticao == 0")["probabilidade"].to_numpy()

        limiar = evaluate.limiar_otimo(y, prob, razao=200.0)
        metricas = evaluate.metricas_no_limiar(y, prob, limiar)
        assert metricas["recall"] > 0.98

    def test_minimiza_de_fato_o_custo(self, oof_sintetico: pd.DataFrame) -> None:
        y = oof_sintetico.query("repeticao == 0")["y_verdadeiro"].to_numpy()
        prob = oof_sintetico.query("repeticao == 0")["probabilidade"].to_numpy()

        otimo = evaluate.limiar_otimo(y, prob, razao=3.0)
        custo_otimo = evaluate.custo_esperado(y, prob, otimo, 3.0)
        custos = [
            evaluate.custo_esperado(y, prob, limiar, 3.0)
            for limiar in evaluate.GRADE_LIMIARES
        ]
        assert custo_otimo == pytest.approx(min(custos))


class TestMetricas:
    def test_contagens_somam_o_total(self, oof_sintetico: pd.DataFrame) -> None:
        y = oof_sintetico.query("repeticao == 0")["y_verdadeiro"].to_numpy()
        prob = oof_sintetico.query("repeticao == 0")["probabilidade"].to_numpy()
        m = evaluate.metricas_no_limiar(y, prob, 0.5)

        total = (
            m["verdadeiros_positivos"]
            + m["falsos_positivos"]
            + m["falsos_negativos"]
            + m["verdadeiros_negativos"]
        )
        assert total == len(y)

    def test_reporta_as_quatro_metricas_exigidas(
        self, oof_sintetico: pd.DataFrame
    ) -> None:
        tabela = evaluate.avaliar_no_limiar(oof_sintetico, 0.5)
        assert set(tabela["metrica"]) == set(evaluate.METRICAS_EXIGIDAS)
        assert (tabela["ic_inferior"] <= tabela["valor"]).all()
        assert (tabela["valor"] <= tabela["ic_superior"]).all()


class TestSensibilidade:
    def test_recall_cresce_com_o_custo_do_falso_negativo(
        self, oof_sintetico: pd.DataFrame
    ) -> None:
        curva = evaluate.curva_sensibilidade(
            oof_sintetico, razoes=(1.0, 3.0, 10.0, 50.0)
        )
        assert curva["recall"].is_monotonic_increasing
        assert curva["limiar"].is_monotonic_decreasing

    def test_marca_exatamente_uma_ancora(self, oof_sintetico: pd.DataFrame) -> None:
        curva = evaluate.curva_sensibilidade(oof_sintetico)
        assert curva["ancora"].sum() == 1
        assert (
            curva.loc[curva["ancora"], "razao_custo"].iloc[0]
            == config.RAZAO_CUSTO_ANCORA
        )

    def test_mais_sinalizados_significa_menos_falsos_negativos(
        self, oof_sintetico: pd.DataFrame
    ) -> None:
        curva = evaluate.curva_sensibilidade(
            oof_sintetico, razoes=(1.0, 3.0, 10.0, 50.0)
        )
        assert curva["n_sinalizados"].is_monotonic_increasing
        assert curva["falsos_negativos"].is_monotonic_decreasing


class TestCurvaGanho:
    def test_captura_e_monotonica_e_completa(self, oof_sintetico: pd.DataFrame) -> None:
        ganho = evaluate.curva_ganho(oof_sintetico)
        assert ganho["pct_recalls_capturados"].is_monotonic_increasing
        assert ganho["pct_recalls_capturados"].iloc[-1] == pytest.approx(1.0)

    def test_lift_termina_em_um(self, oof_sintetico: pd.DataFrame) -> None:
        """Inspecionar a frota inteira nao pode ser melhor que o acaso."""
        ganho = evaluate.curva_ganho(oof_sintetico)
        assert ganho["lift"].iloc[-1] == pytest.approx(1.0)

    def test_modelo_util_tem_lift_acima_de_um_no_topo(
        self, oof_sintetico: pd.DataFrame
    ) -> None:
        ganho = evaluate.curva_ganho(oof_sintetico)
        assert ganho["lift"].iloc[0] > 1.0

    def test_modelo_sem_sinal_nao_tem_ganho(self) -> None:
        """Probabilidade aleatoria: o lift precisa ficar em torno de 1."""
        gerador = np.random.default_rng(config.SEED)
        n = 400
        oof = pd.DataFrame(
            {
                "repeticao": 0,
                "indice": np.arange(n),
                "y_verdadeiro": np.repeat([0, 1], n // 2),
                "probabilidade": gerador.uniform(size=n),
            }
        )
        ganho = evaluate.curva_ganho(oof)
        assert abs(ganho["lift"].iloc[0] - 1.0) < 0.5


class TestCalibracao:
    def test_modelo_perfeitamente_calibrado_tem_desvio_nulo(self) -> None:
        """Se 30% de risco significa 30% de recalls, o desvio precisa zerar."""
        gerador = np.random.default_rng(config.SEED)
        prob = gerador.uniform(0.05, 0.95, 4_000)
        y = (gerador.uniform(size=4_000) < prob).astype(int)
        oof = pd.DataFrame(
            {
                "repeticao": 0,
                "indice": np.arange(len(y)),
                "y_verdadeiro": y,
                "probabilidade": prob,
            }
        )
        tabela = evaluate.calibracao(oof)
        assert tabela["desvio"].abs().max() < 0.06

    def test_modelo_descalibrado_e_detectado(self) -> None:
        gerador = np.random.default_rng(config.SEED)
        prob = gerador.uniform(0.05, 0.95, 4_000)
        # Frequencia real e o dobro da prevista: descalibragem grosseira.
        y = (gerador.uniform(size=4_000) < np.clip(prob * 2, 0, 1)).astype(int)
        oof = pd.DataFrame(
            {
                "repeticao": 0,
                "indice": np.arange(len(y)),
                "y_verdadeiro": y,
                "probabilidade": prob,
            }
        )
        tabela = evaluate.calibracao(oof)
        assert tabela["desvio"].abs().max() > 0.15
