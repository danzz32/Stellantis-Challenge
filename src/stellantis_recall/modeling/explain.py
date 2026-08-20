"""Parte 4: interpretacao do modelo.

Quatro leituras complementares da mesma pergunta -- quais variaveis importam --
porque nenhuma delas sozinha responde de forma confiavel neste dataset.

**Coeficientes.** Diretos e exatos para um modelo linear, mas sob colinearidade
de 0,947 a repartição entre `idade_veiculo` e `km` e arbitraria: a penalizacao
L2 distribui o peso entre as duas de um modo que depende da amostra, nao do
fenomeno. Ler o coeficiente de cada uma isoladamente induz a erro.

**Importancia por permutacao, variavel a variavel.** Mede a perda de desempenho
ao destruir a relacao de uma variavel com o alvo. Sofre do mesmo problema, em
outra direcao: quando duas variaveis carregam a mesma informacao, permutar uma
delas quase nao degrada o modelo -- a outra cobre a ausencia -- e ambas parecem
irrelevantes.

**Importancia por grupo.** A correcao para o problema acima, e o motivo pelo
qual este modulo nao se limita a chamar `permutation_importance`. Permutando
`idade_veiculo` e `km` em bloco, com a mesma reordenacao de linhas para as duas,
a informacao compartilhada e removida de uma vez e a contribuicao conjunta
aparece.

**SHAP.** Decompoe a predicao *por veiculo*, e nao por variavel no agregado. E o
que sustenta a pergunta operacional do dashboard -- por que este veiculo esta no
topo da lista de inspecao.

Uso pelo terminal:

    uv run python -m stellantis_recall.modeling.explain
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import build_mart, config
from . import train

#: Permutacoes por dobra. Cinco bastam: a media sobre 25 dobras ja soma 125
#: reordenacoes por variavel, e o ruido residual fica bem abaixo do efeito.
N_PERMUTACOES = 5

#: Grupos de variaveis avaliadas em bloco. A chave e o nome do grupo.
#: `idade_veiculo` e `km` andam juntas porque correlacionam 0,947 -- avalia-las
#: separadamente esconde a contribuicao das duas.
GRUPOS_COLINEARES: dict[str, tuple[str, ...]] = {
    "tempo_e_uso": ("idade_veiculo", "km"),
}

#: Unidade em que cada variavel e comunicavel, e o rotulo correspondente.
#: A razao de chances "por quilometro" e 1,00001 -- verdadeira e inutil. Por
#: 10 mil quilometros ela vira um numero que cabe numa frase.
UNIDADES_COMUNICAVEIS: dict[str, tuple[float, str]] = {
    "idade_veiculo": (1.0, "por ano de idade"),
    "km": (10_000.0, "por 10 mil km rodados"),
    "reclamacoes": (1.0, "por reclamação registrada"),
    "km_por_ano": (1_000.0, "por mil km/ano"),
    "reclamacoes_por_ano": (1.0, "por reclamação/ano"),
    "reclamacoes_por_10k_km": (1.0, "por reclamação a cada 10 mil km"),
    "muitas_reclamacoes": (1.0, "com 3 ou mais reclamações"),
}


# --------------------------------------------------------------------------- #
# Importancia por permutacao
# --------------------------------------------------------------------------- #


def _permutar(
    X: pd.DataFrame,
    colunas: tuple[str, ...],
    gerador: np.random.Generator,
) -> pd.DataFrame:
    """Reordena um bloco de colunas mantendo a associacao interna do bloco.

    A mesma permutacao de linhas e aplicada a todas as colunas do grupo. Isso e
    o que diferencia importancia de grupo de "permutar cada uma separadamente":
    a relacao entre `idade_veiculo` e `km` e preservada, e o que se destroi e
    apenas a ligacao do bloco com o alvo.
    """
    embaralhado = X.copy()
    ordem = gerador.permutation(len(X))
    embaralhado[list(colunas)] = X[list(colunas)].to_numpy()[ordem]
    return embaralhado


def importancia_por_permutacao(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: dict[str, tuple[str, ...]],
    *,
    n_permutacoes: int = N_PERMUTACOES,
) -> pd.DataFrame:
    """Queda de ROC AUC ao permutar cada grupo, medida fora da amostra.

    A permutacao acontece na particao de *teste* de cada dobra, com o modelo
    ajustado na particao de treino. Permutar sobre o conjunto de treino mediria
    o quanto o modelo se apoiou na variavel, e nao o quanto ela contribui para
    generalizar -- que e a pergunta da Parte 4.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=config.N_DOBRAS,
        n_repeats=config.N_REPETICOES,
        random_state=config.SEED,
    )
    gerador = np.random.default_rng(config.SEED)

    linhas = []
    for i, (treino, teste) in enumerate(cv.split(X, y)):
        ajustado = clone(pipeline).fit(X.iloc[treino], y[treino])
        X_teste, y_teste = X.iloc[teste], y[teste]
        referencia = roc_auc_score(y_teste, ajustado.predict_proba(X_teste)[:, 1])

        for nome, colunas in grupos.items():
            quedas = []
            for _ in range(n_permutacoes):
                embaralhado = _permutar(X_teste, colunas, gerador)
                auc = roc_auc_score(
                    y_teste, ajustado.predict_proba(embaralhado)[:, 1]
                )
                quedas.append(referencia - auc)
            linhas.append(
                {
                    "dobra": i,
                    "grupo": nome,
                    "n_variaveis": len(colunas),
                    "auc_referencia": referencia,
                    "queda_auc": float(np.mean(quedas)),
                }
            )

    return pd.DataFrame(linhas)


def resumir_importancia(detalhado: pd.DataFrame) -> pd.DataFrame:
    """Media e intervalo de confianca da queda de AUC, por grupo."""
    linhas = []
    for nome, grupo in detalhado.groupby("grupo", sort=False):
        valores = grupo["queda_auc"].to_numpy()
        media = float(valores.mean())
        erro_padrao = float(valores.std(ddof=1) / np.sqrt(len(valores)))
        margem = config.Z_NIVEL_CONFIANCA * erro_padrao
        linhas.append(
            {
                "grupo": nome,
                "n_variaveis": int(grupo["n_variaveis"].iloc[0]),
                "queda_auc": media,
                "ic_inferior": media - margem,
                "ic_superior": media + margem,
                # Uma queda cujo intervalo cruza zero nao e distinguivel de
                # ruido de permutacao.
                "relevante": bool(media - margem > 0),
            }
        )
    return pd.DataFrame(linhas).sort_values("queda_auc", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Coeficientes
# --------------------------------------------------------------------------- #


def _escalas(pipeline: Pipeline) -> dict[str, float]:
    """Desvio-padrao usado na padronizacao, por variavel.

    Necessario para converter o coeficiente -- que o modelo estima na escala
    padronizada -- de volta para a unidade original, que e a unica em que a
    razao de chances pode ser comunicada a area de negocio.
    """
    preproc = pipeline.named_steps["preproc"]
    escalas: dict[str, float] = {}

    for nome, transformador, colunas in preproc.transformers_:
        alvo = transformador
        if isinstance(alvo, Pipeline):
            alvo = alvo.steps[-1][1]
        if isinstance(alvo, StandardScaler):
            escalas.update(dict(zip(colunas, alvo.scale_)))

    return escalas


def coeficientes(pipeline: Pipeline) -> pd.DataFrame:
    """Coeficientes da Regressao Logistica, em escala padronizada e natural.

    A leitura padronizada permite comparar variaveis medidas em unidades
    diferentes (anos, quilometros, contagem). A leitura natural traduz o
    coeficiente em razao de chances por unidade real -- por ano de idade, por
    mil quilometros -- que e a forma comunicavel.

    Ressalva que precisa acompanhar qualquer leitura destes numeros: sob
    colinearidade de 0,947, a divisao do efeito entre `idade_veiculo` e `km` e
    instavel, e o coeficiente individual de cada uma nao deve ser interpretado
    isoladamente. A leitura confiavel e a do grupo.
    """
    modelo = pipeline.named_steps["modelo"]
    if not hasattr(modelo, "coef_"):
        raise TypeError(
            f"{type(modelo).__name__} nao expoe coeficientes. "
            "Esta leitura so se aplica a modelos lineares."
        )

    nomes = list(pipeline.named_steps["preproc"].get_feature_names_out())
    coefs = modelo.coef_.ravel()
    escalas = _escalas(pipeline)

    tabela = pd.DataFrame(
        {
            "variavel": nomes,
            "coeficiente_padronizado": coefs,
            "razao_chances_por_desvio": np.exp(coefs),
            "escala": [escalas.get(nome, np.nan) for nome in nomes],
        }
    )
    tabela["coeficiente_natural"] = tabela["coeficiente_padronizado"] / tabela["escala"]
    tabela["razao_chances_por_unidade"] = np.exp(tabela["coeficiente_natural"])

    unidades = [UNIDADES_COMUNICAVEIS.get(nome, (1.0, "por unidade")) for nome in nomes]
    tabela["fator_unidade"] = [fator for fator, _ in unidades]
    tabela["unidade"] = [rotulo for _, rotulo in unidades]
    tabela["razao_chances_comunicavel"] = np.exp(
        tabela["coeficiente_natural"] * tabela["fator_unidade"]
    )

    tabela["magnitude"] = tabela["coeficiente_padronizado"].abs()
    return tabela.sort_values("magnitude", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #


def valores_shap(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Contribuicao de cada variavel para cada predicao.

    Para um modelo linear, os valores SHAP tem forma fechada -- sao o desvio
    padronizado multiplicado pelo coeficiente -- entao a decomposicao e exata,
    e nao uma aproximacao amostral.

    Returns:
        Valores por observacao e o resumo por variavel (media do valor absoluto).
    """
    import shap

    preproc = pipeline.named_steps["preproc"]
    modelo = pipeline.named_steps["modelo"]
    nomes = list(preproc.get_feature_names_out())
    transformado = preproc.transform(X)

    # O masker explicito evita a subamostragem padrao do SHAP em 100 linhas de
    # fundo. Com 499 observacoes o custo e irrelevante, e usar a base inteira
    # torna o valor de referencia exato em vez de amostral.
    fundo = shap.maskers.Independent(transformado, max_samples=len(transformado))
    explicador = shap.LinearExplainer(modelo, fundo)
    valores = np.asarray(explicador.shap_values(transformado))

    por_observacao = pd.DataFrame(valores, columns=nomes)
    por_observacao.insert(0, "indice", np.arange(len(X)))

    resumo = (
        pd.DataFrame(
            {
                "variavel": nomes,
                "shap_medio_absoluto": np.abs(valores).mean(axis=0),
                "shap_medio": valores.mean(axis=0),
            }
        )
        .sort_values("shap_medio_absoluto", ascending=False, ignore_index=True)
    )
    return por_observacao, resumo


def contribuicoes_individuais(
    pipeline: Pipeline, X: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Decompoe a predicao de cada observacao em contribuicoes por variavel.

    Para um modelo linear a decomposicao e exata e tem forma fechada: a
    contribuicao de uma variavel e o seu valor padronizado multiplicado pelo
    coeficiente. E a mesma quantidade que o SHAP calcula, sem custo de
    amostragem -- o que permite recalcula-la ao vivo para um veiculo digitado
    pelo usuario.

    A soma das contribuicoes mais o valor de referencia devolve o log-odds da
    predicao. Contribuicao positiva empurra o veiculo para o risco; negativa,
    para longe dele.

    Returns:
        Contribuicoes por observacao e o valor de referencia (intercepto), que
        corresponde ao log-odds de um veiculo medio.
    """
    preproc = pipeline.named_steps["preproc"]
    modelo = pipeline.named_steps["modelo"]
    if not hasattr(modelo, "coef_"):
        raise TypeError(
            f"{type(modelo).__name__} nao expoe coeficientes. "
            "A decomposicao exata so se aplica a modelos lineares."
        )

    nomes = list(preproc.get_feature_names_out())
    transformado = np.asarray(preproc.transform(X), dtype=float)
    contribuicoes = transformado * modelo.coef_.ravel()

    return (
        pd.DataFrame(contribuicoes, columns=nomes, index=X.index),
        float(modelo.intercept_[0]),
    )


# --------------------------------------------------------------------------- #
# Dependencia parcial
# --------------------------------------------------------------------------- #


def dependencia_parcial(
    pipeline: Pipeline,
    X: pd.DataFrame,
    variaveis: tuple[str, ...],
    n_pontos: int = 30,
) -> pd.DataFrame:
    """Risco previsto em funcao de cada variavel, com as demais promediadas.

    Traduz o modelo para a linguagem da area de negocio: "a partir de quantas
    reclamacoes o risco passa de 50%". Sob colinearidade a curva inclui um
    aviso implicito -- ela avalia combinacoes de idade e quilometragem que
    praticamente nao ocorrem na frota real, como um veiculo de 8 anos com 10 mil
    km. A leitura vale na regiao densa dos dados, nao nas pontas.
    """
    # `partial_dependence` recusa colunas inteiras: a grade que ela constroi e
    # continua, e o arredondamento implicito distorceria a curva.
    continuo = X.astype(
        {coluna: "float64" for coluna in X.select_dtypes("number").columns}
    )

    linhas = []
    for variavel in variaveis:
        resultado = partial_dependence(
            pipeline,
            continuo,
            features=[variavel],
            grid_resolution=n_pontos,
            response_method="predict_proba",
        )
        valores = np.asarray(resultado["grid_values"][0])
        risco = np.asarray(resultado["average"][0])
        linhas.append(
            pd.DataFrame(
                {"variavel": variavel, "valor": valores, "risco_previsto": risco}
            )
        )
    return pd.concat(linhas, ignore_index=True)


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResultadoExplicacao:
    """Evidencia completa da Parte 4."""

    importancia_individual: pd.DataFrame
    importancia_grupos: pd.DataFrame
    coeficientes: pd.DataFrame
    shap_resumo: pd.DataFrame
    shap_valores: pd.DataFrame
    dependencia: pd.DataFrame


def executar(*, persistir: bool = True) -> ResultadoExplicacao:
    """Produz as quatro leituras de interpretacao do modelo escolhido."""
    pipeline, metadados = train.carregar_modelo()
    colunas = tuple(metadados["colunas"])

    dados = build_mart.carregar_features()
    X = dados.loc[:, list(colunas)]
    y = dados[config.COLUNA_ALVO].to_numpy(dtype=int)

    # Leitura 1: cada variavel isolada.
    individuais = {coluna: (coluna,) for coluna in colunas}
    detalhado_individual = importancia_por_permutacao(pipeline, X, y, individuais)
    importancia_individual = resumir_importancia(detalhado_individual)

    # Leitura 2: variaveis colineares em bloco. So os grupos cujos membros
    # estao todos presentes no conjunto de atributos escolhido.
    grupos = {
        nome: membros
        for nome, membros in GRUPOS_COLINEARES.items()
        if set(membros) <= set(colunas)
    }
    restantes = set(colunas) - {m for membros in grupos.values() for m in membros}
    grupos.update({coluna: (coluna,) for coluna in sorted(restantes)})

    detalhado_grupos = importancia_por_permutacao(pipeline, X, y, grupos)
    importancia_grupos = resumir_importancia(detalhado_grupos)

    tabela_coeficientes = coeficientes(pipeline)
    shap_valores, shap_resumo = valores_shap(pipeline, X)
    dependencia = dependencia_parcial(pipeline, X, colunas)

    if persistir:
        config.garantir_diretorios()
        importancia_individual.to_parquet(config.IMPORTANCIA_PERMUTACAO, index=False)
        importancia_grupos.to_parquet(config.IMPORTANCIA_GRUPOS, index=False)
        tabela_coeficientes.to_parquet(config.COEFICIENTES, index=False)
        shap_resumo.to_parquet(config.SHAP_RESUMO, index=False)
        shap_valores.to_parquet(config.SHAP_VALORES, index=False)
        dependencia.to_parquet(config.DEPENDENCIA_PARCIAL, index=False)

    return ResultadoExplicacao(
        importancia_individual=importancia_individual,
        importancia_grupos=importancia_grupos,
        coeficientes=tabela_coeficientes,
        shap_resumo=shap_resumo,
        shap_valores=shap_valores,
        dependencia=dependencia,
    )


def main() -> int:
    """Executa a interpretacao e imprime as quatro leituras."""
    resultado = executar()

    print("Importancia por permutacao -- variavel a variavel (queda de ROC AUC):")
    print(resultado.importancia_individual.round(4).to_string(index=False))

    print()
    print("Importancia por permutacao -- colineares em bloco:")
    print(resultado.importancia_grupos.round(4).to_string(index=False))

    print()
    print("Coeficientes da Regressao Logistica:")
    print(
        resultado.coeficientes[
            [
                "variavel",
                "coeficiente_padronizado",
                "razao_chances_por_desvio",
                "razao_chances_comunicavel",
                "unidade",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("SHAP -- contribuicao media absoluta:")
    print(resultado.shap_resumo.round(4).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
