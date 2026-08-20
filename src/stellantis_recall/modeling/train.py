"""Parte 2: treino e comparacao de modelos.

Compara quatro classificadores sobre tres conjuntos de atributos, todos sob o
mesmo protocolo de validacao (`RepeatedStratifiedKFold`, 5 dobras x 5
repeticoes). O modulo nao escolhe "o melhor numero": ele produz a evidencia
necessaria para justificar a escolha, e a selecao segue um criterio declarado em
`config.METRICA_SELECAO`.

Tres decisoes de projeto merecem registro.

**Toda transformacao que aprende parametro vive dentro do `Pipeline`.** Escala,
log1p e codificacao de categoria sao ajustadas apenas na particao de treino de
cada dobra. Aplicadas antes da validacao cruzada, veriam o conjunto inteiro e
inflariam as metricas de forma invisivel.

**Cada modelo recebe o pre-processamento de que precisa, e nao um comum.** A
Regressao Logistica exige escala comparavel e sofre com a assimetria de 11,03 de
`reclamacoes_por_10k_km`; arvores sao indiferentes a ambos, por operarem sobre
ordenacao. Um pre-processador unico penalizaria um dos dois lados sem motivo.

**A selecao usa ROC AUC, nao Accuracy.** O limiar sera deslocado por custo na
Parte 3; escolher o modelo por uma metrica medida em 0,5 selecionaria pelo
desempenho num ponto de corte que nao sera usado.

Uso pelo terminal:

    uv run python -m stellantis_recall.modeling.train
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from scipy import stats

from .. import build_mart, config

# --------------------------------------------------------------------------- #
# Conjuntos de atributos
#
# Os tres conjuntos nao sao variacoes arbitrarias: cada um testa uma hipotese
# levantada na analise exploratoria.
# --------------------------------------------------------------------------- #

CONJUNTOS_FEATURES: dict[str, tuple[str, ...]] = {
    # As quatro colunas como vieram. E a referencia honesta -- o que o dataset
    # do enunciado oferece sem intervencao.
    "originais": ("modelo", "idade_veiculo", "km", "reclamacoes"),
    # Sem `modelo`. Testa diretamente o achado da Parte 1: chi2(8) = 6,69,
    # p = 0,570, V de Cramer = 0,116. Se o modelo do veiculo nao discrimina
    # risco, remover as 9 dummies deve manter o desempenho e reduzir ruido.
    "sem_modelo": ("idade_veiculo", "km", "reclamacoes"),
    # Com as derivadas. Testa se normalizar por exposicao acrescenta algo -- a
    # correlacao marginal das razoes com o alvo e proxima de zero (-0,021,
    # +0,062, -0,078), entao a expectativa e de ganho nulo. Um resultado
    # negativo aqui e reportavel, nao desperdicio.
    "com_derivadas": (
        "modelo",
        "idade_veiculo",
        "km",
        "reclamacoes",
        "km_por_ano",
        "reclamacoes_por_ano",
        "reclamacoes_por_10k_km",
        "muitas_reclamacoes",
    ),
}

CATEGORICAS = ("modelo",)
BOOLEANAS = ("muitas_reclamacoes",)
NUMERICAS_SIMETRICAS = ("idade_veiculo", "km", "reclamacoes")
NUMERICAS_ASSIMETRICAS = (
    "km_por_ano",
    "reclamacoes_por_ano",
    "reclamacoes_por_10k_km",
)


# --------------------------------------------------------------------------- #
# Pre-processamento
# --------------------------------------------------------------------------- #


def _presentes(candidatas: tuple[str, ...], colunas: tuple[str, ...]) -> list[str]:
    return [coluna for coluna in candidatas if coluna in colunas]


def construir_preprocessador(
    colunas: tuple[str, ...],
    *,
    sensivel_a_escala: bool,
) -> ColumnTransformer:
    """Monta o pre-processamento adequado ao tipo de modelo.

    Args:
        colunas: atributos do conjunto em uso.
        sensivel_a_escala: True para modelos lineares. Aplica padronizacao e
            comprime a cauda das razoes com log1p; alem disso descarta a
            primeira dummy de cada categorica, que sob multicolinearidade ja
            severa evita agravar a matriz de projeto. False para arvores, que
            dispensam ambos.
    """
    categoricas = _presentes(CATEGORICAS, colunas)
    booleanas = _presentes(BOOLEANAS, colunas)
    simetricas = _presentes(NUMERICAS_SIMETRICAS, colunas)
    assimetricas = _presentes(NUMERICAS_ASSIMETRICAS, colunas)

    codificador = OneHotEncoder(
        drop="first" if sensivel_a_escala else None,
        handle_unknown="ignore",
        sparse_output=False,
    )

    if sensivel_a_escala:
        trilha_simetrica = StandardScaler()
        trilha_assimetrica = Pipeline(
            [
                # log1p e seguro com zeros, que existem em reclamacoes.
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("escala", StandardScaler()),
            ]
        )
    else:
        trilha_simetrica = "passthrough"
        trilha_assimetrica = "passthrough"

    transformacoes = [
        ("categoricas", codificador, categoricas),
        ("booleanas", "passthrough", booleanas),
        ("simetricas", trilha_simetrica, simetricas),
        ("assimetricas", trilha_assimetrica, assimetricas),
    ]
    # ColumnTransformer aceita lista vazia de colunas, mas manter a etapa
    # atrapalha a leitura dos nomes de atributo na Parte 4.
    transformacoes = [t for t in transformacoes if t[2]]

    return ColumnTransformer(transformacoes, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------- #
# Modelos
#
# Hiperparametros fixados em valores regularizados, sem busca. Com 499 linhas e
# 25 dobras ja em uso, uma busca aninhada multiplicaria o custo para capturar
# ganho marginal, e ajustar hiperparametro na mesma particao que reporta a
# metrica e a forma mais comum de inflar resultado em amostra pequena. A busca
# fica registrada como melhoria proposta na Parte 4.
# --------------------------------------------------------------------------- #


def construir_modelos() -> dict[str, tuple[object, bool]]:
    """Devolve `nome -> (estimador, sensivel_a_escala)`.

    O XGBoost e importado aqui dentro, e nao no topo do modulo, pelo mesmo
    motivo que o SHAP em `explain.py`: ele so e necessario para *treinar*. Quem
    apenas carrega o modelo ja ajustado -- o painel interativo, por exemplo --
    passa a nao depender de um pacote de centenas de megabytes, o que importa em
    ambientes de publicacao com limite de recursos.
    """
    from xgboost import XGBClassifier

    return {
        # Piso de referencia: prediz sempre a classe majoritaria. Qualquer
        # modelo precisa superar 52,1% de Accuracy para ter serventia.
        "baseline": (DummyClassifier(strategy="most_frequent"), False),
        # Regularizacao L2 nao e refinamento opcional aqui: com VIF de 10,4 e
        # 9,7 entre idade e km, a solucao sem penalizacao tem coeficientes
        # instaveis e de sinal potencialmente invertido.
        # `l1_ratio=0` e a forma de pedir L2 pura no sklearn >= 1.8, onde o
        # argumento `penalty` foi depreciado em favor da parametrizacao
        # continua entre L1 e L2.
        "regressao_logistica": (
            LogisticRegression(
                l1_ratio=0.0,
                C=1.0,
                max_iter=2_000,
                random_state=config.SEED,
            ),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                # Folha minima de 5 em 400 linhas de treino: sem isso a arvore
                # memoriza observacoes isoladas da cauda de reclamacoes.
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=config.SEED,
                n_jobs=-1,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=config.SEED,
                n_jobs=-1,
            ),
            False,
        ),
    }


def montar_pipeline(nome_modelo: str, colunas: tuple[str, ...]) -> Pipeline:
    """Monta o `Pipeline` completo de um modelo para um conjunto de atributos."""
    estimador, sensivel = construir_modelos()[nome_modelo]
    return Pipeline(
        [
            ("preproc", construir_preprocessador(colunas, sensivel_a_escala=sensivel)),
            ("modelo", estimador),
        ]
    )


# --------------------------------------------------------------------------- #
# Validacao cruzada
# --------------------------------------------------------------------------- #

METRICAS_LIMIAR_FIXO = ("accuracy", "precision", "recall", "f1")
METRICAS_SEM_LIMIAR = ("roc_auc", "average_precision", "brier")


def _metricas_da_dobra(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Metricas de uma dobra.

    As quatro exigidas pelo enunciado sao medidas no limiar convencional de 0,5.
    Isso e proposital: elas servem de comparacao entre modelos sob condicao
    identica. O deslocamento do limiar por custo acontece na Parte 3, sobre as
    probabilidades fora da amostra que este modulo persiste.
    """
    pred = (prob >= 0.5).astype(int)
    # `zero_division=0` porque o baseline nunca prediz a classe positiva, e a
    # precisao indefinida dele deve aparecer como zero, nao como erro.
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "average_precision": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
    }


def avaliar(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda a validacao cruzada repetida.

    Devolve as metricas por dobra e as probabilidades fora da amostra. As
    probabilidades sao guardadas por repeticao -- cada observacao recebe uma
    predicao em cada uma das 5 repeticoes -- para que a Parte 3 possa estimar
    tambem a incerteza do limiar otimo, e nao so a das metricas.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=config.N_DOBRAS,
        n_repeats=config.N_REPETICOES,
        random_state=config.SEED,
    )

    linhas: list[dict[str, float]] = []
    predicoes: list[pd.DataFrame] = []

    for i, (treino, teste) in enumerate(cv.split(X, y)):
        repeticao, dobra = divmod(i, config.N_DOBRAS)

        ajustado = clone(pipeline).fit(X.iloc[treino], y[treino])
        prob = ajustado.predict_proba(X.iloc[teste])[:, 1]

        linhas.append(
            {"repeticao": repeticao, "dobra": dobra, **_metricas_da_dobra(y[teste], prob)}
        )
        predicoes.append(
            pd.DataFrame(
                {
                    "repeticao": repeticao,
                    "indice": teste,
                    "y_verdadeiro": y[teste],
                    "probabilidade": prob,
                }
            )
        )

    return pd.DataFrame(linhas), pd.concat(predicoes, ignore_index=True)


def resumir(dobras: pd.DataFrame) -> dict[str, float]:
    """Media e intervalo de confianca de cada metrica sobre as 25 dobras.

    O intervalo sai do erro padrao das estimativas por dobra. Ele descreve a
    variabilidade do procedimento de estimacao -- as dobras compartilham dados e
    nao sao independentes, entao a leitura correta e "quanto esta metrica oscila
    conforme a particao", nao um intervalo populacional estrito.
    """
    resumo: dict[str, float] = {}
    for metrica in (*METRICAS_LIMIAR_FIXO, *METRICAS_SEM_LIMIAR):
        valores = dobras[metrica].to_numpy()
        media = float(valores.mean())
        erro_padrao = float(valores.std(ddof=1) / np.sqrt(len(valores)))
        margem = config.Z_NIVEL_CONFIANCA * erro_padrao
        resumo[metrica] = media
        resumo[f"{metrica}_ic_inferior"] = media - margem
        resumo[f"{metrica}_ic_superior"] = media + margem
        resumo[f"{metrica}_desvio"] = float(valores.std(ddof=1))
    return resumo


# --------------------------------------------------------------------------- #
# Comparacao entre modelos
# --------------------------------------------------------------------------- #


def teste_pareado_corrigido(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_dobras: int = config.N_DOBRAS,
) -> tuple[float, float]:
    """Teste t pareado com a correcao de Nadeau-Bengio.

    O t pareado convencional pressupoe estimativas independentes. As 25 dobras
    da validacao cruzada repetida nao sao: elas compartilham dados de treino, e
    a correlacao resultante subestima a variancia. Aplicado sem correcao, o
    teste declara diferencas significantes que sao artefato do protocolo -- o
    erro mais comum ao comparar modelos por validacao cruzada.

    A correcao infla a variancia pelo fator (1/k + n_teste/n_treino), que para
    5 dobras repetidas 5 vezes leva o divisor de 1/25 para 1/25 + 1/4. O erro
    padrao resultante e cerca de 2,7 vezes maior, e a leitura fica conservadora.

    Referencia: Nadeau & Bengio (2003), *Inference for the Generalization Error*.

    Returns:
        Estatistica t e p-valor bilateral.
    """
    diferenca = a - b
    n = len(diferenca)
    proporcao_teste = 1.0 / (n_dobras - 1)  # n_teste / n_treino em k dobras
    variancia_corrigida = diferenca.var(ddof=1) * (1.0 / n + proporcao_teste)

    if variancia_corrigida <= 0:
        # Variancia nula: a diferenca e identica em todas as dobras. Os dois
        # casos precisam ser distinguidos -- tratar ambos como "indistinguivel"
        # esconderia uma diferenca deterministica, que e o erro na direcao mais
        # perigosa.
        if np.isclose(diferenca.mean(), 0.0):
            return 0.0, 1.0
        return float(np.sign(diferenca.mean()) * np.inf), 0.0

    t = float(diferenca.mean() / np.sqrt(variancia_corrigida))
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    return t, p


def testes_pareados(
    dobras: pd.DataFrame,
    referencia: tuple[str, str],
    metrica: str = config.METRICA_SELECAO,
) -> pd.DataFrame:
    """Compara a combinacao de referencia com todas as demais, dobra a dobra.

    O pareamento por dobra e o que torna a comparacao informativa: os dois
    modelos sao medidos exatamente sobre as mesmas particoes, entao a variacao
    de particionamento -- que domina em 499 linhas -- se cancela.
    """
    chave = ["repeticao", "dobra"]

    def serie(modelo: str, conjunto: str) -> np.ndarray:
        recorte = dobras[(dobras["modelo"] == modelo) & (dobras["conjunto"] == conjunto)]
        return recorte.sort_values(chave)[metrica].to_numpy()

    base = serie(*referencia)
    combinacoes = dobras[["modelo", "conjunto"]].drop_duplicates()

    linhas = []
    for modelo, conjunto in combinacoes.itertuples(index=False):
        if (modelo, conjunto) == referencia:
            continue
        outro = serie(modelo, conjunto)
        t_corrigido, p_corrigido = teste_pareado_corrigido(base, outro)
        _, p_ingenuo = stats.ttest_rel(base, outro)
        linhas.append(
            {
                "modelo": modelo,
                "conjunto": conjunto,
                "diferenca_media": float((base - outro).mean()),
                "t_corrigido": t_corrigido,
                "p_corrigido": p_corrigido,
                "p_ingenuo": float(p_ingenuo),
                "distinguivel": bool(p_corrigido < config.ALFA),
            }
        )

    return pd.DataFrame(linhas).sort_values("diferenca_media", ignore_index=True)


def selecionar(comparacao: pd.DataFrame) -> pd.Series:
    """Aplica a regra de um erro padrao, com desempate por parcimonia.

    Escolher simplesmente a maior media seria selecionar ruido: as combinacoes
    de topo diferem por milesimos, muito abaixo do desvio de ~0,045 entre
    dobras. A regra adotada e a convencional em validacao cruzada -- entre todas
    as combinacoes cuja media esta a menos de um erro padrao da melhor, fica a
    de menor numero de atributos.

    Parcimonia como criterio de desempate nao e preferencia estetica: menos
    atributos significa menos parametros estimados sobre 499 linhas, coeficiente
    mais estavel e explicacao mais simples para a area de negocio.
    """
    metrica = config.METRICA_SELECAO
    candidatos = comparacao[comparacao["modelo"] != "baseline"]

    melhor = candidatos.loc[candidatos[metrica].idxmax()]
    erro_padrao = melhor[f"{metrica}_desvio"] / np.sqrt(
        config.N_DOBRAS * config.N_REPETICOES
    )
    limite = melhor[metrica] - erro_padrao

    equivalentes = candidatos[candidatos[metrica] >= limite]
    return equivalentes.sort_values(
        ["n_features", metrica], ascending=[True, False]
    ).iloc[0]


# --------------------------------------------------------------------------- #
# Execucao
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResultadoTreino:
    """Evidencia completa da Parte 2."""

    comparacao: pd.DataFrame
    dobras: pd.DataFrame
    predicoes_oof: pd.DataFrame
    testes: pd.DataFrame
    nome_escolhido: str
    conjunto_escolhido: str
    pipeline_final: Pipeline


def executar(*, persistir: bool = True) -> ResultadoTreino:
    """Compara todos os modelos em todos os conjuntos e ajusta o escolhido."""
    dados = build_mart.carregar_features()
    y = dados[config.COLUNA_ALVO].to_numpy(dtype=int)

    comparacao: list[dict[str, object]] = []
    todas_dobras: list[pd.DataFrame] = []
    todas_predicoes: list[pd.DataFrame] = []

    for nome_conjunto, colunas in CONJUNTOS_FEATURES.items():
        X = dados.loc[:, list(colunas)]
        for nome_modelo in construir_modelos():
            pipeline = montar_pipeline(nome_modelo, colunas)
            dobras, predicoes = avaliar(pipeline, X, y)

            identificacao = {"modelo": nome_modelo, "conjunto": nome_conjunto}
            comparacao.append({**identificacao, "n_features": len(colunas), **resumir(dobras)})
            todas_dobras.append(dobras.assign(**identificacao))
            todas_predicoes.append(predicoes.assign(**identificacao))

    tabela = pd.DataFrame(comparacao).sort_values(
        config.METRICA_SELECAO, ascending=False, ignore_index=True
    )

    # O baseline nunca entra na selecao: ele existe como piso de comparacao, e
    # o seu ROC AUC de 0,5 e artefato de probabilidade constante.
    escolhido = selecionar(tabela)
    nome_escolhido = str(escolhido["modelo"])
    conjunto_escolhido = str(escolhido["conjunto"])

    colunas_escolhidas = CONJUNTOS_FEATURES[conjunto_escolhido]
    pipeline_final = montar_pipeline(nome_escolhido, colunas_escolhidas)
    pipeline_final.fit(dados.loc[:, list(colunas_escolhidas)], y)

    dobras_completas = pd.concat(todas_dobras, ignore_index=True)
    predicoes_completas = pd.concat(todas_predicoes, ignore_index=True)
    testes = testes_pareados(
        dobras_completas, referencia=(nome_escolhido, conjunto_escolhido)
    )

    if persistir:
        config.garantir_diretorios()
        tabela.to_parquet(config.COMPARACAO_MODELOS, index=False)
        dobras_completas.to_parquet(config.METRICAS_DOBRAS, index=False)
        predicoes_completas.to_parquet(config.PREDICOES_OOF, index=False)
        testes.to_parquet(config.TESTES_PAREADOS, index=False)
        joblib.dump(pipeline_final, config.MODELO_FINAL)
        config.MODELO_METADADOS.write_text(
            json.dumps(
                {
                    "modelo": nome_escolhido,
                    "conjunto_features": conjunto_escolhido,
                    "colunas": list(colunas_escolhidas),
                    "metrica_selecao": config.METRICA_SELECAO,
                    "valor_metrica_selecao": float(escolhido[config.METRICA_SELECAO]),
                    "seed": config.SEED,
                    "n_dobras": config.N_DOBRAS,
                    "n_repeticoes": config.N_REPETICOES,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return ResultadoTreino(
        comparacao=tabela,
        dobras=dobras_completas,
        predicoes_oof=predicoes_completas,
        testes=testes,
        nome_escolhido=nome_escolhido,
        conjunto_escolhido=conjunto_escolhido,
        pipeline_final=pipeline_final,
    )


def carregar_modelo() -> tuple[Pipeline, dict[str, object]]:
    """Le o modelo final e os seus metadados."""
    if not config.MODELO_FINAL.is_file():
        raise FileNotFoundError(
            f"{config.MODELO_FINAL} nao existe. Rode "
            "`uv run python -m stellantis_recall.modeling.train`."
        )
    metadados = json.loads(config.MODELO_METADADOS.read_text(encoding="utf-8"))
    return joblib.load(config.MODELO_FINAL), metadados


def carregar_predicoes_oof() -> pd.DataFrame:
    """Le as probabilidades fora da amostra usadas pela Parte 3."""
    if not config.PREDICOES_OOF.is_file():
        raise FileNotFoundError(
            f"{config.PREDICOES_OOF} nao existe. Rode "
            "`uv run python -m stellantis_recall.modeling.train`."
        )
    return pd.read_parquet(config.PREDICOES_OOF)


def main() -> int:
    """Treina, compara e persiste; imprime a tabela de comparacao."""
    resultado = executar()

    colunas_exibidas = [
        "modelo",
        "conjunto",
        "n_features",
        "roc_auc",
        "average_precision",
        "accuracy",
        "f1",
        "brier",
    ]
    print(resultado.comparacao[colunas_exibidas].round(4).to_string(index=False))
    print()
    print(
        f"Escolhido ({config.METRICA_SELECAO}, regra de um erro padrao + parcimonia): "
        f"{resultado.nome_escolhido} / {resultado.conjunto_escolhido}"
    )
    print()
    print("Comparacao pareada contra o escolhido (Nadeau-Bengio):")
    print(resultado.testes.round(4).to_string(index=False))
    print()
    print(f"Gravado em: {config.MODELO_FINAL}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
