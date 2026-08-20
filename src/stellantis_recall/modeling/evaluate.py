"""Parte 3: avaliacao do modelo e definicao do ponto de corte.

O enunciado pede Accuracy, Precision, Recall e F1. Essas quatro metricas nao
existem sem um limiar, e o limiar convencional de 0,5 nao e neutro -- ele e o
otimo apenas quando os dois tipos de erro custam o mesmo. Este modulo trata o
limiar como o que ele e: uma decisao de negocio, derivada de uma razao de custo
declarada, e acompanhada da analise de sensibilidade que mostra o que aconteceria
sob outras razoes.

Tudo aqui parte das probabilidades **fora da amostra** persistidas pela Parte 2.
Nenhuma metrica e calculada sobre dado que o modelo viu no treino, e a estrutura
por repeticao -- cada observacao recebe uma predicao em cada uma das 5
repeticoes -- permite estimar o intervalo de confianca tanto das metricas quanto
do proprio limiar otimo.

Uso pelo terminal:

    uv run python -m stellantis_recall.modeling.evaluate
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from .. import config
from . import train

#: Grade de limiares varrida na otimizacao. Fina o bastante para que o limiar
#: reportado tenha tres casas uteis, sem depender dos valores exatos de
#: probabilidade observados numa repeticao especifica.
GRADE_LIMIARES = np.linspace(0.01, 0.99, 197)

METRICAS_EXIGIDAS = ("accuracy", "precision", "recall", "f1")


# --------------------------------------------------------------------------- #
# Custo e limiar
# --------------------------------------------------------------------------- #


def custo_esperado(
    y: np.ndarray,
    prob: np.ndarray,
    limiar: float,
    razao: float,
) -> float:
    """Custo total de decidir com um dado limiar, em unidades de falso positivo.

    So a razao entre os custos importa: fixar o custo do falso positivo em 1 e
    exprimir o do falso negativo como multiplo dele nao perde generalidade e
    elimina uma unidade monetaria que o desafio nao fornece.
    """
    predito = prob >= limiar
    falsos_negativos = int(((y == 1) & ~predito).sum())
    falsos_positivos = int(((y == 0) & predito).sum())
    return falsos_negativos * razao + falsos_positivos * config.CUSTO_FALSO_POSITIVO


def limiar_otimo(y: np.ndarray, prob: np.ndarray, razao: float) -> float:
    """Limiar que minimiza o custo esperado sob a razao informada.

    Havendo empate, prevalece o maior limiar. O desempate nao e arbitrario: em
    caso de custo igual, o limiar mais alto sinaliza menos veiculos e portanto
    consome menos capacidade de inspecao.
    """
    custos = np.array(
        [custo_esperado(y, prob, limiar, razao) for limiar in GRADE_LIMIARES]
    )
    return float(GRADE_LIMIARES[np.flatnonzero(custos == custos.min())[-1]])


def metricas_no_limiar(
    y: np.ndarray,
    prob: np.ndarray,
    limiar: float,
) -> dict[str, float]:
    """As quatro metricas exigidas, mais a contagem que traduz cada uma."""
    predito = (prob >= limiar).astype(int)
    verdadeiros_positivos = int(((y == 1) & (predito == 1)).sum())
    falsos_positivos = int(((y == 0) & (predito == 1)).sum())
    falsos_negativos = int(((y == 1) & (predito == 0)).sum())
    verdadeiros_negativos = int(((y == 0) & (predito == 0)).sum())

    return {
        "limiar": limiar,
        "accuracy": accuracy_score(y, predito),
        "precision": precision_score(y, predito, zero_division=0),
        "recall": recall_score(y, predito, zero_division=0),
        "f1": f1_score(y, predito, zero_division=0),
        "verdadeiros_positivos": verdadeiros_positivos,
        "falsos_positivos": falsos_positivos,
        "falsos_negativos": falsos_negativos,
        "verdadeiros_negativos": verdadeiros_negativos,
        "n_sinalizados": verdadeiros_positivos + falsos_positivos,
        "taxa_sinalizacao": (verdadeiros_positivos + falsos_positivos) / len(y),
    }


# --------------------------------------------------------------------------- #
# Agregacao por repeticao
# --------------------------------------------------------------------------- #


def _por_repeticao(oof: pd.DataFrame):
    """Itera sobre as repeticoes, devolvendo (rotulo, probabilidade)."""
    for repeticao, grupo in oof.groupby("repeticao", sort=True):
        ordenado = grupo.sort_values("indice")
        yield (
            int(repeticao),
            ordenado["y_verdadeiro"].to_numpy(dtype=int),
            ordenado["probabilidade"].to_numpy(dtype=float),
        )


def _resumir(valores: np.ndarray) -> tuple[float, float, float]:
    """Media e intervalo de confianca a partir das repeticoes."""
    media = float(valores.mean())
    if len(valores) < 2:
        return media, media, media
    erro_padrao = float(valores.std(ddof=1) / np.sqrt(len(valores)))
    margem = config.Z_NIVEL_CONFIANCA * erro_padrao
    return media, media - margem, media + margem


def avaliar_no_limiar(oof: pd.DataFrame, limiar: float) -> pd.DataFrame:
    """Metricas no limiar dado, com intervalo estimado sobre as repeticoes."""
    por_repeticao = pd.DataFrame(
        [metricas_no_limiar(y, prob, limiar) for _, y, prob in _por_repeticao(oof)]
    )

    linhas = []
    for metrica in METRICAS_EXIGIDAS:
        media, inferior, superior = _resumir(por_repeticao[metrica].to_numpy())
        linhas.append(
            {
                "metrica": metrica,
                "valor": media,
                "ic_inferior": inferior,
                "ic_superior": superior,
                "limiar": limiar,
            }
        )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Analise de sensibilidade ao custo
# --------------------------------------------------------------------------- #


def curva_sensibilidade(
    oof: pd.DataFrame,
    razoes: tuple[float, ...] = config.RAZOES_CUSTO_SENSIBILIDADE,
) -> pd.DataFrame:
    """Como a decisao muda conforme a razao de custo.

    Este e o entregavel que impede o limiar de parecer arbitrario. Para cada
    razao, o limiar otimo e recalculado *dentro de cada repeticao* -- o que
    fornece nao so a media, mas a estabilidade da propria decisao. Um limiar com
    intervalo largo e um sinal de alerta: significa que a recomendacao muda
    conforme a particao dos dados.
    """
    linhas = []
    for razao in razoes:
        limiares = np.array(
            [limiar_otimo(y, prob, razao) for _, y, prob in _por_repeticao(oof)]
        )
        limiar_medio, limiar_inferior, limiar_superior = _resumir(limiares)

        metricas = pd.DataFrame(
            [
                metricas_no_limiar(y, prob, limiar)
                for (_, y, prob), limiar in zip(_por_repeticao(oof), limiares)
            ]
        )

        registro: dict[str, object] = {
            "razao_custo": razao,
            "limiar": limiar_medio,
            "limiar_ic_inferior": limiar_inferior,
            "limiar_ic_superior": limiar_superior,
            "ancora": razao == config.RAZAO_CUSTO_ANCORA,
        }
        for coluna in (
            *METRICAS_EXIGIDAS,
            "falsos_negativos",
            "falsos_positivos",
            "n_sinalizados",
            "taxa_sinalizacao",
        ):
            media, inferior, superior = _resumir(metricas[coluna].to_numpy())
            registro[coluna] = media
            registro[f"{coluna}_ic_inferior"] = inferior
            registro[f"{coluna}_ic_superior"] = superior

        linhas.append(registro)

    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Curvas
# --------------------------------------------------------------------------- #


def probabilidade_media(oof: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Media das probabilidades das 5 repeticoes, por observacao.

    Usada apenas nas curvas e na calibracao, onde o objetivo e uma leitura
    unica e estavel do comportamento do modelo. As metricas com intervalo
    continuam saindo repeticao a repeticao -- promediar antes achataria
    justamente a variabilidade que se quer medir.
    """
    agrupado = (
        oof.groupby("indice", sort=True)
        .agg(y=("y_verdadeiro", "first"), prob=("probabilidade", "mean"))
        .reset_index()
    )
    return agrupado["y"].to_numpy(dtype=int), agrupado["prob"].to_numpy(dtype=float)


def curva_roc(oof: pd.DataFrame) -> pd.DataFrame:
    y, prob = probabilidade_media(oof)
    fpr, tpr, limiares = roc_curve(y, prob)
    return pd.DataFrame(
        {"falso_positivo": fpr, "verdadeiro_positivo": tpr, "limiar": limiares}
    )


def curva_precisao_revocacao(oof: pd.DataFrame) -> pd.DataFrame:
    """Curva Precision-Recall.

    Mais informativa que a ROC quando a decisao pende para a classe positiva,
    que e exatamente o caso aqui: com o custo do falso negativo acima do custo
    do falso positivo, o regime de interesse e o de alto Recall, onde a ROC
    comprime as diferencas.
    """
    y, prob = probabilidade_media(oof)
    precisao, revocacao, limiares = precision_recall_curve(y, prob)
    # `precision_recall_curve` devolve um ponto a mais que limiares.
    return pd.DataFrame(
        {
            "precisao": precisao[:-1],
            "revocacao": revocacao[:-1],
            "limiar": limiares,
        }
    )


def calibracao(oof: pd.DataFrame, n_faixas: int = 10) -> pd.DataFrame:
    """Confronta probabilidade prevista com frequencia observada.

    Um modelo bem calibrado entrega, entre os veiculos aos quais atribuiu 30% de
    risco, cerca de 30% de recalls. Isso importa aqui mais do que o usual: o
    limiar e escolhido por custo, e a otimizacao so faz sentido se a
    probabilidade de fato significar o que diz.
    """
    y, prob = probabilidade_media(oof)
    faixas = pd.cut(prob, bins=np.linspace(0.0, 1.0, n_faixas + 1), include_lowest=True)

    tabela = (
        pd.DataFrame({"y": y, "prob": prob, "faixa": faixas})
        .groupby("faixa", observed=True)
        .agg(
            n=("y", "size"),
            probabilidade_media=("prob", "mean"),
            frequencia_observada=("y", "mean"),
        )
        .reset_index()
    )
    tabela["faixa"] = tabela["faixa"].astype(str)
    tabela["desvio"] = tabela["frequencia_observada"] - tabela["probabilidade_media"]
    return tabela


def curva_ganho(oof: pd.DataFrame, n_faixas: int = 10) -> pd.DataFrame:
    """Quantos recalls sao capturados ao inspecionar os N% de maior risco.

    Reformula o problema de "classificar" para "priorizar", e essa mudanca de
    enquadramento e o resultado mais util da Parte 3. Com discriminacao moderada
    (ROC AUC 0,79), qualquer limiar que alcance Recall alto acaba sinalizando a
    maior parte da frota -- o modelo nao consegue *excluir* veiculos com
    seguranca. O que ele consegue e *ordenar*: dizer quais inspecionar primeiro
    quando a capacidade de oficina e limitada, que e a restricao real de uma
    operacao de pos-vendas.

    O lift responde a pergunta executiva direta: inspecionando esta fatia da
    frota, quantas vezes mais recalls sao encontrados do que ao sortear veiculos
    ao acaso.
    """
    y, prob = probabilidade_media(oof)
    ordenado = y[np.argsort(-prob)]
    n_total = len(y)
    recalls_totais = int(y.sum())
    taxa_base = recalls_totais / n_total

    linhas = []
    for faixa in range(1, n_faixas + 1):
        n_inspecionados = int(round(n_total * faixa / n_faixas))
        capturados = int(ordenado[:n_inspecionados].sum())
        linhas.append(
            {
                "faixa": faixa,
                "pct_frota": faixa / n_faixas,
                "n_inspecionados": n_inspecionados,
                "recalls_capturados": capturados,
                "pct_recalls_capturados": capturados / recalls_totais,
                "precisao_na_faixa": capturados / n_inspecionados,
                "lift": (capturados / n_inspecionados) / taxa_base,
            }
        )
    return pd.DataFrame(linhas)


def matriz_confusao(oof: pd.DataFrame, limiar: float) -> pd.DataFrame:
    """Matriz de confusao media por repeticao, em formato longo."""
    por_repeticao = pd.DataFrame(
        [metricas_no_limiar(y, prob, limiar) for _, y, prob in _por_repeticao(oof)]
    )
    celulas = {
        ("Sem recall", "Nao sinalizado"): "verdadeiros_negativos",
        ("Sem recall", "Sinalizado"): "falsos_positivos",
        ("Com recall", "Nao sinalizado"): "falsos_negativos",
        ("Com recall", "Sinalizado"): "verdadeiros_positivos",
    }
    return pd.DataFrame(
        [
            {
                "real": real,
                "predito": predito,
                "n": float(por_repeticao[coluna].mean()),
                "celula": coluna,
            }
            for (real, predito), coluna in celulas.items()
        ]
    )


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResultadoAvaliacao:
    """Evidencia completa da Parte 3."""

    metricas: pd.DataFrame
    sensibilidade: pd.DataFrame
    roc: pd.DataFrame
    precisao_revocacao: pd.DataFrame
    calibracao: pd.DataFrame
    ganho: pd.DataFrame
    confusao: pd.DataFrame
    decisao: dict[str, object]


def executar(*, persistir: bool = True) -> ResultadoAvaliacao:
    """Avalia o modelo escolhido e define o ponto de corte operacional."""
    _, metadados = train.carregar_modelo()
    oof_completo = train.carregar_predicoes_oof()

    oof = oof_completo[
        (oof_completo["modelo"] == metadados["modelo"])
        & (oof_completo["conjunto"] == metadados["conjunto_features"])
    ]
    if oof.empty:
        raise ValueError(
            f"Nao ha predicoes fora da amostra para "
            f"{metadados['modelo']}/{metadados['conjunto_features']}."
        )

    sensibilidade = curva_sensibilidade(oof)
    ancora = sensibilidade[sensibilidade["ancora"]].iloc[0]
    limiar_operacional = round(float(ancora["limiar"]), 3)

    # As quatro metricas sao reportadas nos dois pontos: no limiar convencional,
    # que e o que a maioria dos trabalhos apresenta sem declarar, e no limiar
    # derivado do custo, que e o recomendado. A comparacao entre os dois e o
    # argumento.
    #
    # Distincao sutil e proposital: aqui o limiar unico ja arredondado e
    # aplicado a todas as repeticoes, porque e assim que ele opera em producao.
    # A curva de sensibilidade, por outro lado, reotimiza o limiar dentro de
    # cada repeticao -- ela mede a *estabilidade da decisao*, nao o desempenho
    # do limiar implantado. Dai as pequenas diferencas entre as duas tabelas.
    metricas = pd.concat(
        [
            avaliar_no_limiar(oof, 0.5).assign(cenario="limiar convencional (0,5)"),
            avaliar_no_limiar(oof, limiar_operacional).assign(
                cenario=f"limiar por custo {config.RAZAO_CUSTO_ANCORA:.0f}:1"
            ),
        ],
        ignore_index=True,
    )

    confusao = matriz_confusao(oof, limiar_operacional)
    roc = curva_roc(oof)
    pr = curva_precisao_revocacao(oof)
    calib = calibracao(oof)
    ganho = curva_ganho(oof)

    terco_superior = ganho[ganho["faixa"] == 3].iloc[0]
    metade_superior = ganho[ganho["faixa"] == 5].iloc[0]

    decisao = {
        "modelo": metadados["modelo"],
        "conjunto_features": metadados["conjunto_features"],
        "razao_custo": config.RAZAO_CUSTO_ANCORA,
        "limiar_operacional": limiar_operacional,
        "limiar_ic_inferior": round(float(ancora["limiar_ic_inferior"]), 3),
        "limiar_ic_superior": round(float(ancora["limiar_ic_superior"]), 3),
        "taxa_sinalizacao": round(float(ancora["taxa_sinalizacao"]), 4),
        "falsos_negativos_medios": round(float(ancora["falsos_negativos"]), 1),
        "falsos_positivos_medios": round(float(ancora["falsos_positivos"]), 1),
        "erro_absoluto_medio_calibracao": round(float(calib["desvio"].abs().mean()), 4),
        "captura_top30": round(float(terco_superior["pct_recalls_capturados"]), 4),
        "lift_top30": round(float(terco_superior["lift"]), 2),
        "captura_top50": round(float(metade_superior["pct_recalls_capturados"]), 4),
        "lift_top50": round(float(metade_superior["lift"]), 2),
    }

    if persistir:
        config.garantir_diretorios()
        metricas.to_parquet(config.METRICAS_FINAIS, index=False)
        sensibilidade.to_parquet(config.SENSIBILIDADE_CUSTO, index=False)
        roc.to_parquet(config.CURVA_ROC, index=False)
        pr.to_parquet(config.CURVA_PR, index=False)
        calib.to_parquet(config.CALIBRACAO, index=False)
        ganho.to_parquet(config.CURVA_GANHO, index=False)
        confusao.to_parquet(config.MATRIZ_CONFUSAO, index=False)
        config.DECISAO_OPERACIONAL.write_text(
            json.dumps(decisao, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return ResultadoAvaliacao(
        metricas=metricas,
        sensibilidade=sensibilidade,
        roc=roc,
        precisao_revocacao=pr,
        calibracao=calib,
        ganho=ganho,
        confusao=confusao,
        decisao=decisao,
    )


def carregar_decisao() -> dict[str, object]:
    """Le o ponto de corte operacional definido por esta etapa."""
    if not config.DECISAO_OPERACIONAL.is_file():
        raise FileNotFoundError(
            f"{config.DECISAO_OPERACIONAL} nao existe. Rode "
            "`uv run python -m stellantis_recall.modeling.evaluate`."
        )
    return json.loads(config.DECISAO_OPERACIONAL.read_text(encoding="utf-8"))


def main() -> int:
    """Avalia, persiste e imprime o resumo da Parte 3."""
    resultado = executar()

    print("Metricas (predicoes fora da amostra, IC sobre 5 repeticoes):")
    print(
        resultado.metricas.pivot(index="metrica", columns="cenario", values="valor")
        .round(4)
        .to_string()
    )

    print()
    print("Sensibilidade ao custo:")
    colunas = [
        "razao_custo",
        "limiar",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "falsos_negativos",
        "n_sinalizados",
        "ancora",
    ]
    print(resultado.sensibilidade[colunas].round(4).to_string(index=False))

    print()
    print("Ganho por priorizacao (ordenando a frota por risco):")
    print(
        resultado.ganho[
            ["pct_frota", "n_inspecionados", "recalls_capturados",
             "pct_recalls_capturados", "lift"]
        ]
        .round(3)
        .to_string(index=False)
    )

    print()
    print("Matriz de confusao no limiar operacional:")
    print(
        resultado.confusao.pivot(index="real", columns="predito", values="n")
        .round(1)
        .to_string()
    )

    print()
    print("Decisao operacional:")
    for chave, valor in resultado.decisao.items():
        print(f"  {chave}: {valor}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
