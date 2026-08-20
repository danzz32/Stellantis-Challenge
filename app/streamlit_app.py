"""Painel interativo de risco de recall.

Espelha as cinco visoes do painel Quarto (`qmd/dashboard.qmd`) e acrescenta o
que um documento estatico nao pode oferecer:

* **Planejador de capacidade** -- o usuario informa quantos veiculos consegue
  inspecionar e le quantos recalls aquilo encontra.
* **Simulador de custo** -- a razao entre o custo de um recall nao antecipado e
  o de uma inspecao desnecessaria vira um controle. O limiar otimo, as quatro
  metricas e a matriz de confusao se recalculam ao vivo sobre as predicoes fora
  da amostra.
* **Score de veiculo** -- idade, quilometragem e reclamacoes digitados devolvem
  o risco previsto e a decomposicao exata de como cada fator contribuiu.

O aplicativo nao treina nem recalcula nada de pesado: le os mesmos Parquet de
`data/mart/` e `outputs/metrics/` que o relatorio e o painel Quarto consomem.
Isso garante que os tres entregaveis nunca discordem entre si.

Sobre a camada de apresentacao
------------------------------
O layout usa `streamlit-shadcn-ui` para cartoes, abas, distintivos e tabelas. A
troca resolve tambem uma inconsistencia de paleta: `st.warning` e `st.error`
pintam em amarelo e vermelho, que nao pertencem a identidade institucional. Os
alertas do shadcn em variante neutra respeitam a decisao registrada em
`viz.py` -- sem cor quente, com enfase por peso de texto e rotulo explicito.

Uso:

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_shadcn_ui as ui

from stellantis_recall import build_mart, config, eda, transform, viz
from stellantis_recall.modeling import evaluate, explain, train

# --------------------------------------------------------------------------- #
# Configuracao
# --------------------------------------------------------------------------- #

DIRETORIO_APP = Path(__file__).resolve().parent
LOGO = DIRETORIO_APP / "assets" / "logo.png"

st.set_page_config(
    page_title="Risco de Recall — Painel de Garantia",
    page_icon=str(LOGO) if LOGO.is_file() else "🔧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

viz.aplicar_tema()

#: Abaixo deste numero de veiculos, a taxa de um grupo e ruido e nao estimativa.
MINIMO_AMOSTRA = 20

PAGINAS = (
    "Visão executiva",
    "Perfil da frota",
    "Qualidade dos dados",
    "Desempenho do modelo",
    "Interpretação",
)

ROTULOS_MODELO = {
    "baseline": "Baseline trivial",
    "regressao_logistica": "Reg. logística L2",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
ROTULOS_CONJUNTO = {
    "originais": "originais",
    "sem_modelo": "sem modelo",
    "com_derivadas": "com derivadas",
}

ORDEM_FAIXA_IDADE = (
    "0-1 anos (novo)",
    "2-3 anos (garantia)",
    "4-5 anos (pos-garantia)",
    "6+ anos (maduro)",
)
ORDEM_FAIXA_KM = (
    "ate 25 mil km",
    "25-75 mil km",
    "75-125 mil km",
    "acima de 125 mil km",
)


#: Alinha os componentes shadcn a paleta institucional.
#:
#: Os componentes renderizam em *shadow DOM* com um tema Tailwind proprio, cujo
#: `:host` define `--primary` como um preto neutro. Variavel definida em `:root`
#: nao alcanca esse escopo -- regra de `:host` vence valor herdado. O que vence e
#: uma regra do documento externo mirando o proprio elemento hospedeiro: pela
#: cascata, declaracoes da arvore externa tem precedencia sobre `:host`.
#:
#: `[data-ssui-v2-host]` e o atributo que a propria biblioteca marca no host --
#: estavel, ao contrario das classes hash do Streamlit, que mudam a cada versao.
#:
#: Os valores vem de `viz.py` para que exista uma unica fonte da verdade da
#: identidade visual entre figuras, documentos e painel.
CSS_INSTITUCIONAL = f"""
<style>
[data-ssui-v2-host] {{
    --primary: {viz.AZUL};
    --primary-foreground: {viz.BRANCO};
    --ring: {viz.AZUL};
    --foreground: {viz.GRAFITE};
    --card-foreground: {viz.GRAFITE};
    --popover-foreground: {viz.GRAFITE};
    --muted: #F4F5F7;
    --muted-foreground: {viz.TINTA_SUAVE};
    --secondary: {viz.FUNDO_DESTAQUE};
    --secondary-foreground: {viz.AZUL_PROFUNDO};
    --accent: {viz.FUNDO_DESTAQUE};
    --accent-foreground: {viz.AZUL_PROFUNDO};
    --border: {viz.GRADE};
    --input: {viz.GRADE};
}}
/* Controles nativos do Streamlit que convivem com os componentes shadcn. */
[data-testid="stSlider"] [role="slider"] {{
    border-color: {viz.AZUL_PROFUNDO};
}}
</style>
"""

st.markdown(CSS_INSTITUCIONAL, unsafe_allow_html=True)


def br(valor: float, casas: int = 2) -> str:
    """Numero no formato brasileiro: virgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


def pct(valor: float, casas: int = 0) -> str:
    """Proporcao como percentual, no formato brasileiro."""
    return f"{br(valor * 100, casas)}%"


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Carregando camada mart…")
def carregar_dados() -> dict[str, pd.DataFrame]:
    """Le os Parquet do mart e das metricas. Cacheado por sessao."""
    return {
        "features": build_mart.carregar_features(),
        "ranking": build_mart.carregar_agregado("ranking_risco"),
        "evolucao": build_mart.carregar_agregado("evolucao_por_idade"),
        "perfil": build_mart.carregar_agregado("perfil_por_modelo"),
        "segmentos": build_mart.carregar_agregado("matriz_segmentos"),
        "qualidade": pd.read_parquet(config.RELATORIO_QUALIDADE),
        "comparacao": pd.read_parquet(config.COMPARACAO_MODELOS),
        "testes": pd.read_parquet(config.TESTES_PAREADOS),
        "metricas": pd.read_parquet(config.METRICAS_FINAIS),
        "sensibilidade": pd.read_parquet(config.SENSIBILIDADE_CUSTO),
        "roc": pd.read_parquet(config.CURVA_ROC),
        "pr": pd.read_parquet(config.CURVA_PR),
        "calibracao": pd.read_parquet(config.CALIBRACAO),
        "ganho": pd.read_parquet(config.CURVA_GANHO),
        "confusao": pd.read_parquet(config.MATRIZ_CONFUSAO),
        "imp_individual": pd.read_parquet(config.IMPORTANCIA_PERMUTACAO),
        "imp_grupos": pd.read_parquet(config.IMPORTANCIA_GRUPOS),
        "coeficientes": pd.read_parquet(config.COEFICIENTES),
        "dependencia": pd.read_parquet(config.DEPENDENCIA_PARCIAL),
    }


@st.cache_data(show_spinner=False)
def carregar_oof() -> pd.DataFrame:
    """Predicoes fora da amostra do modelo escolhido."""
    metadados = json.loads(config.MODELO_METADADOS.read_text(encoding="utf-8"))
    oof = train.carregar_predicoes_oof()
    return oof[
        (oof["modelo"] == metadados["modelo"])
        & (oof["conjunto"] == metadados["conjunto_features"])
    ]


@st.cache_resource(show_spinner=False)
def carregar_modelo():
    """Pipeline treinado e seus metadados."""
    return train.carregar_modelo()


@st.cache_data(show_spinner=False)
def simular_custo(razao: float) -> dict[str, float]:
    """Recalcula limiar otimo e metricas para uma razao de custo arbitraria.

    O limiar e reotimizado dentro de cada repeticao e depois promediado -- o
    mesmo procedimento de `evaluate.curva_sensibilidade`, aplicado a um valor
    que nao esta na grade pre-calculada. E o que permite o controle deslizante
    responder a qualquer razao, e nao apenas as sete tabeladas.
    """
    oof = carregar_oof()
    limiares, metricas = [], []
    for _, y, prob in evaluate._por_repeticao(oof):
        limiar = evaluate.limiar_otimo(y, prob, razao)
        limiares.append(limiar)
        metricas.append(evaluate.metricas_no_limiar(y, prob, limiar))

    tabela = pd.DataFrame(metricas)
    return {
        "limiar": float(np.mean(limiares)),
        **{coluna: float(tabela[coluna].mean()) for coluna in tabela.columns},
    }


@st.cache_data(show_spinner=False)
def ler_entregavel(caminho: Path) -> bytes | None:
    """Le um documento de `reports/` para oferecer download, se ele existir."""
    return caminho.read_bytes() if caminho.is_file() else None


# --------------------------------------------------------------------------- #
# Blocos de apresentacao
# --------------------------------------------------------------------------- #


def mostrar(figura) -> None:
    """Renderiza e libera a figura, evitando acumulo de memoria entre reruns."""
    st.pyplot(figura, width="stretch")
    plt.close(figura)


def nota(texto: str) -> None:
    """Ressalva metodologica discreta, junto do grafico a que se refere."""
    st.caption(texto)


def ressalva(titulo: str, descricao: str, chave: str) -> None:
    """Ressalva estatistica que precisa ser lida antes da conclusao.

    Usa a variante neutra do shadcn de proposito. A paleta institucional nao tem
    cor de alerta, e `st.warning` traria um amarelo que nao pertence a
    identidade -- a enfase fica no rotulo e no peso do texto.
    """
    ui.alert(title=titulo, description=descricao, key=chave)


def titulo_secao(texto: str, apoio: str | None = None) -> None:
    """Cabecalho de secao com subtitulo opcional."""
    st.markdown(f"#### {texto}")
    if apoio:
        st.caption(apoio)


dados = carregar_dados()
pipeline, metadados = carregar_modelo()
decisao = evaluate.carregar_decisao()

features = dados["features"]
ganho = dados["ganho"]
n_veiculos = len(features)
taxa_base = float(ganho["precisao_na_faixa"].iloc[-1])
limiar_vigente = float(decisao["limiar_operacional"])

# --------------------------------------------------------------------------- #
# Cabecalho institucional
# --------------------------------------------------------------------------- #

marca, identificacao, acoes = st.columns([0.7, 6.3, 3.0], vertical_alignment="center")

with marca:
    if LOGO.is_file():
        st.image(str(LOGO), width=68)

with identificacao:
    st.markdown(
        "<div style='line-height:1.25'>"
        "<div style='font-size:1.55rem;font-weight:650;color:#00133B'>"
        "Risco de Recall</div>"
        "<div style='font-size:0.92rem;color:#686B71'>"
        "Painel de Garantia · Qualidade e Pós-Vendas</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with acoes:
    ui.badges(
        [
            ui.BadgeItem(text="dados sintéticos", variant="secondary"),
            ui.BadgeItem(text=ROTULOS_MODELO[metadados["modelo"]], variant="outline"),
            ui.BadgeItem(
                text=f"ROC AUC {br(metadados['valor_metrica_selecao'], 3)}",
                variant="default",
            ),
        ],
        key="badges-cabecalho",
    )

ui.separator(key="sep-cabecalho")

pagina = ui.tabs(options=PAGINAS, value=PAGINAS[0], key="navegacao")

# --------------------------------------------------------------------------- #
# Barra lateral — procedência e entregáveis
# --------------------------------------------------------------------------- #

with st.sidebar:
    if LOGO.is_file():
        st.image(str(LOGO), width=52)
    st.markdown("**Procedência dos números**")
    st.caption(
        "Todos os valores vêm dos mesmos arquivos Parquet que alimentam o "
        "relatório técnico e o PDF executivo. Este painel não treina nada."
    )

    ui.separator(key="sep-lateral-1")

    ui.metric_card(
        label="Modelo em uso",
        value=ROTULOS_MODELO[metadados["modelo"]],
        description=", ".join(metadados["colunas"]),
        key="cartao-modelo",
    )
    ui.metric_card(
        label="Protocolo de validação",
        value=f"{metadados['n_dobras']}×{metadados['n_repeticoes']}",
        description="dobras estratificadas × repetições",
        key="cartao-validacao",
    )
    ui.metric_card(
        label="Limiar de decisão",
        value=br(limiar_vigente, 3),
        description=f"razão de custo {config.RAZAO_CUSTO_ANCORA:.0f}:1",
        key="cartao-limiar",
    )

    ui.separator(key="sep-lateral-2")

    st.markdown("**Entregáveis**")
    for caminho, rotulo, tipo in (
        (config.PROJECT_ROOT / "reports" / "02-executivo.pdf",
         "Sumário executivo (PDF)", "application/pdf"),
        (config.PROJECT_ROOT / "reports" / "01-analise.html",
         "Relatório técnico (HTML)", "text/html"),
    ):
        conteudo = ler_entregavel(caminho)
        if conteudo is None:
            st.caption(f"{rotulo} — ainda não renderizado")
            continue
        st.download_button(
            rotulo,
            data=conteudo,
            file_name=caminho.name,
            mime=tipo,
            width="stretch",
            key=f"baixar-{caminho.stem}",
        )

# --------------------------------------------------------------------------- #
# Visão executiva
# --------------------------------------------------------------------------- #

if pagina == "Visão executiva":
    top30 = ganho[ganho["faixa"] == 3].iloc[0]

    colunas = st.columns(4)
    with colunas[0]:
        ui.metric_card(label="Veículos analisados", value=f"{n_veiculos}",
                       description="base de garantia", key="kpi-veiculos")
    with colunas[1]:
        ui.metric_card(label="Tiveram recall", value=pct(taxa_base, 1),
                       description="taxa observada na base", key="kpi-taxa")
    with colunas[2]:
        ui.metric_card(
            label="Recalls nos 30% mais críticos",
            value=pct(top30["pct_recalls_capturados"]),
            description="inspecionando em ordem de risco",
            key="kpi-captura",
        )
    with colunas[3]:
        ui.metric_card(
            label="Ganho sobre inspeção aleatória",
            value=f"{br(top30['lift'])}×",
            description="mais recalls, mesma capacidade",
            key="kpi-lift",
        )

    ui.separator(key="sep-exec-1")
    titulo_secao(
        "Planejador de capacidade de inspeção",
        "A restrição real de pós-vendas não é *quais* veículos inspecionar — é "
        "quantos cabem na capacidade de oficina.",
    )

    esquerda, direita = st.columns([1, 2.2])

    with esquerda:
        capacidade = st.slider(
            "Veículos que consigo inspecionar",
            min_value=int(n_veiculos * 0.05),
            max_value=n_veiculos,
            value=int(n_veiculos * 0.30),
            step=5,
        )
        fracao = capacidade / n_veiculos

        # Interpolação sobre a curva de ganho: a capacidade escolhida raramente
        # cai exatamente num dos decis pré-calculados.
        captura = float(
            np.interp(
                fracao,
                np.concatenate([[0.0], ganho["pct_frota"].to_numpy()]),
                np.concatenate([[0.0], ganho["pct_recalls_capturados"].to_numpy()]),
            )
        )
        recalls_totais = int(ganho["recalls_capturados"].iloc[-1])
        encontrados = int(round(captura * recalls_totais))
        lift = captura / fracao if fracao > 0 else 0.0

        ui.metric_card(label="Fração da frota", value=pct(fracao),
                       description=f"{capacidade} de {n_veiculos} veículos",
                       key="cap-fracao")
        ui.metric_card(label="Recalls encontrados",
                       value=f"{encontrados}",
                       description=f"de {recalls_totais} existentes na base",
                       key="cap-encontrados")
        ui.metric_card(label="Cobertura", value=pct(captura),
                       description=f"{br(lift)}× melhor que sortear ao acaso",
                       key="cap-cobertura")

    with direita:
        com_origem = pd.concat(
            [
                pd.DataFrame([{"pct_frota": 0.0, "pct_recalls_capturados": 0.0}]),
                ganho[["pct_frota", "pct_recalls_capturados"]],
            ],
            ignore_index=True,
        )

        fig, eixo = plt.subplots(figsize=(8.0, 4.2))
        eixo.plot(
            com_origem["pct_frota"], com_origem["pct_recalls_capturados"],
            "o-", color=viz.AZUL, markersize=5, linewidth=2.4,
            label="priorização pelo modelo",
        )
        eixo.plot([0, 1], [0, 1], color=viz.COR_REFERENCIA, linestyle="--",
                  linewidth=1.4, label="inspeção aleatória")
        eixo.fill_between(
            com_origem["pct_frota"], com_origem["pct_frota"],
            com_origem["pct_recalls_capturados"], color=viz.AZUL, alpha=0.14,
        )
        eixo.scatter([fracao], [captura], s=190, color=viz.AZUL_PROFUNDO,
                     zorder=5, marker="D", label="capacidade escolhida")
        eixo.vlines(fracao, 0, captura, color=viz.AZUL_PROFUNDO,
                    linestyle=":", linewidth=1.6)
        eixo.hlines(captura, 0, fracao, color=viz.AZUL_PROFUNDO,
                    linestyle=":", linewidth=1.6)

        eixo.set_xlabel("fração da frota inspecionada")
        eixo.set_ylabel("recalls encontrados")
        eixo.set_xlim(0, 1)
        eixo.set_ylim(0, 1.02)
        viz.rotular_percentual(eixo)
        viz.rotular_percentual_x(eixo)
        viz.apenas_grade_horizontal(eixo)
        eixo.legend(loc="lower right")
        fig.tight_layout()
        mostrar(fig)

    ui.separator(key="sep-exec-2")
    titulo_secao("Onde o risco se concentra")

    esquerda, direita = st.columns([3, 2])

    with esquerda:
        segmentos = dados["segmentos"]
        pivo = segmentos.pivot(
            index="faixa_idade", columns="faixa_km", values="taxa_recall"
        )
        contagem = segmentos.pivot(
            index="faixa_idade", columns="faixa_km", values="n_veiculos"
        )
        indices = [i for i in ORDEM_FAIXA_IDADE if i in pivo.index]
        colunas_km = [k for k in ORDEM_FAIXA_KM if k in pivo.columns]
        pivo = pivo.reindex(index=indices, columns=colunas_km)
        contagem = contagem.reindex(index=indices, columns=colunas_km)
        exibida = pivo.mask(contagem.fillna(0) < MINIMO_AMOSTRA)

        fig, eixo = plt.subplots(figsize=(7.4, 3.8))
        imagem = eixo.imshow(exibida, cmap=viz.MAPA_RISCO, vmin=0, vmax=1,
                             aspect="auto")
        for i in range(len(pivo.index)):
            for j in range(len(pivo.columns)):
                n = contagem.iloc[i, j]
                if pd.isna(n) or n < MINIMO_AMOSTRA:
                    eixo.text(j, i, "n/d", ha="center", va="center",
                              color=viz.TINTA_SUAVE, fontsize=9, style="italic")
                    continue
                taxa = pivo.iloc[i, j]
                eixo.text(
                    j, i, f"{pct(taxa)}\nn={int(n)}", ha="center", va="center",
                    color=viz.BRANCO if taxa > 0.45 else viz.AZUL_PROFUNDO,
                    fontsize=9, fontweight="semibold",
                )
        eixo.set_xticks(
            range(len(pivo.columns)),
            [c.replace(" mil km", "k").replace("acima de ", ">").replace("ate ", "≤")
             for c in pivo.columns],
        )
        eixo.set_yticks(range(len(pivo.index)),
                        [i.split(" (")[0] for i in pivo.index])
        eixo.grid(False)
        fig.colorbar(imagem, ax=eixo, shrink=0.9, label="taxa de recall")
        fig.tight_layout()
        mostrar(fig)
        nota(
            f"Células com menos de {MINIMO_AMOSTRA} veículos aparecem como "
            "**n/d** — a taxa existe, mas não é estimativa confiável."
        )

    with direita:
        ranking = dados["ranking"].sort_values("taxa_recall")
        posicoes = np.arange(len(ranking))
        erro = np.vstack([
            ranking["taxa_recall"] - ranking["ic_inferior"],
            ranking["ic_superior"] - ranking["taxa_recall"],
        ])

        fig, eixo = plt.subplots(figsize=(5.6, 3.8))
        eixo.barh(
            posicoes, ranking["taxa_recall"],
            color=[viz.cor_por_risco(t, 0.2, 0.8) for t in ranking["taxa_recall"]],
            height=0.6, edgecolor=viz.GRADE,
        )
        eixo.errorbar(ranking["taxa_recall"], posicoes, xerr=erro, fmt="none",
                      ecolor=viz.AZUL_PROFUNDO, elinewidth=1.5, capsize=3)
        eixo.axvline(taxa_base, color=viz.COR_REFERENCIA, linestyle="--",
                     linewidth=1.4)
        eixo.set_yticks(posicoes, ranking["modelo"])
        eixo.set_xlabel("taxa de recall")
        viz.rotular_percentual_x(eixo)
        eixo.grid(axis="y", visible=False)
        eixo.set_title("Ranking por modelo")
        fig.tight_layout()
        mostrar(fig)

        ressalva(
            "⚠ Este ranking não é estatisticamente sustentável",
            "O intervalo do primeiro colocado contém integralmente o do último, "
            "e o teste de independência não rejeita a hipótese de que o modelo "
            "do veículo é irrelevante (χ²(8) = 6,69; p = 0,570). Priorizar por "
            "ele seria priorizar por ruído amostral.",
            "ressalva-ranking",
        )

# --------------------------------------------------------------------------- #
# Perfil da frota
# --------------------------------------------------------------------------- #

elif pagina == "Perfil da frota":
    with st.expander("Filtros", expanded=True):
        colunas = st.columns(3)
        modelos = colunas[0].multiselect(
            "Modelo", sorted(features["modelo"].unique()), default=[]
        )
        faixas_idade = colunas[1].multiselect(
            "Faixa de idade",
            [f for f in ORDEM_FAIXA_IDADE if f in set(features["faixa_idade"])],
            default=[],
        )
        faixas_km = colunas[2].multiselect(
            "Faixa de quilometragem",
            [f for f in ORDEM_FAIXA_KM if f in set(features["faixa_km"])],
            default=[],
        )

    recorte = features
    if modelos:
        recorte = recorte[recorte["modelo"].isin(modelos)]
    if faixas_idade:
        recorte = recorte[recorte["faixa_idade"].isin(faixas_idade)]
    if faixas_km:
        recorte = recorte[recorte["faixa_km"].isin(faixas_km)]

    if recorte.empty:
        ressalva(
            "Nenhum veículo no recorte",
            "A combinação de filtros selecionada não retorna registros. "
            "Remova ao menos um critério.",
            "ressalva-vazio",
        )
        st.stop()

    alvo = recorte[config.COLUNA_ALVO].astype(int)
    n_recorte = len(recorte)
    taxa_recorte = float(alvo.mean())
    ic_inf, ic_sup = eda.intervalo_wilson(
        np.array([int(alvo.sum())]), np.array([n_recorte])
    )

    colunas = st.columns(4)
    with colunas[0]:
        ui.metric_card(label="Veículos no recorte", value=f"{n_recorte}",
                       description=f"{pct(n_recorte / n_veiculos)} da base",
                       key="perfil-n")
    with colunas[1]:
        ui.metric_card(label="Taxa de recall", value=pct(taxa_recorte, 1),
                       description=f"base: {pct(taxa_base, 1)}", key="perfil-taxa")
    with colunas[2]:
        ui.metric_card(label="IC 95% da taxa",
                       value=f"{pct(ic_inf[0])}–{pct(ic_sup[0])}",
                       description="intervalo de Wilson", key="perfil-ic")
    with colunas[3]:
        ui.metric_card(label="Reclamações (média)",
                       value=br(float(recorte["reclamacoes"].mean())),
                       description="por veículo no recorte", key="perfil-recl")

    if n_recorte < MINIMO_AMOSTRA:
        ressalva(
            "⚠ Amostra insuficiente para decisão",
            f"O recorte tem apenas {n_recorte} veículos, e o intervalo de "
            f"confiança da taxa tem {pct(ic_sup[0] - ic_inf[0])} de amplitude — "
            "largo demais para sustentar priorização.",
            "ressalva-amostra",
        )

    ui.separator(key="sep-perfil-1")
    esquerda, direita = st.columns(2)

    with esquerda:
        titulo_secao("Evolução ao longo da vida do veículo")
        evolucao = dados["evolucao"]
        ic_inf_e, ic_sup_e = eda.intervalo_wilson(
            evolucao["n_recalls"], evolucao["n_veiculos"]
        )

        fig, eixo = plt.subplots(figsize=(6.6, 3.8))
        eixo.plot(evolucao["idade_veiculo"], evolucao["taxa_recall"], "o-",
                  color=viz.AZUL, markersize=6, linewidth=2.2,
                  label="taxa por idade")
        eixo.fill_between(evolucao["idade_veiculo"], ic_inf_e, ic_sup_e,
                          color=viz.AZUL, alpha=0.16, label="IC 95%")
        eixo.plot(evolucao["idade_veiculo"], evolucao["taxa_recall_acumulada"],
                  "s--", color=viz.TINTA_SUAVE, markersize=4, linewidth=1.5,
                  label="acumulado")
        eixo.set_xlabel("idade (anos)")
        eixo.set_ylabel("taxa de recall")
        viz.rotular_percentual(eixo)
        viz.apenas_grade_horizontal(eixo)
        eixo.legend(loc="upper left")
        fig.tight_layout()
        mostrar(fig)
        nota(
            "`idade_veiculo` é a única dimensão temporal do dataset. Leitura de "
            "coorte — perfil da frota por idade — e não série temporal."
        )

    with direita:
        titulo_secao("Reclamações acumuladas e risco")
        por_reclamacoes = (
            features.assign(alvo=features[config.COLUNA_ALVO].astype(int))
            .groupby("reclamacoes", as_index=False)
            .agg(n=("alvo", "size"), n_recalls=("alvo", "sum"))
        )
        por_reclamacoes["taxa"] = por_reclamacoes["n_recalls"] / por_reclamacoes["n"]
        robustos = por_reclamacoes[por_reclamacoes["n"] >= 10]
        ic_inf_r, ic_sup_r = eda.intervalo_wilson(
            robustos["n_recalls"], robustos["n"]
        )

        fig, eixo = plt.subplots(figsize=(6.6, 3.8))
        eixo.errorbar(
            robustos["reclamacoes"], robustos["taxa"],
            yerr=np.vstack([robustos["taxa"] - ic_inf_r,
                            ic_sup_r - robustos["taxa"]]),
            fmt="o-", color=viz.AZUL, ecolor=viz.TINTA_SUAVE,
            elinewidth=1.4, capsize=4, markersize=6, linewidth=2.2,
        )
        eixo.axvline(config.LIMIAR_RECLAMACOES - 0.5, color=viz.AZUL_PROFUNDO,
                     linestyle="--", linewidth=1.5)
        eixo.annotate(
            f"degrau em {config.LIMIAR_RECLAMACOES}\nreclamações",
            xy=(config.LIMIAR_RECLAMACOES - 0.5, 0.42), xytext=(4.4, 0.22),
            fontsize=9, color=viz.AZUL_PROFUNDO, fontweight="semibold",
            arrowprops=dict(arrowstyle="->", color=viz.AZUL_PROFUNDO,
                            linewidth=1.2),
        )
        eixo.set_xlabel("reclamações registradas")
        eixo.set_ylabel("taxa de recall")
        viz.rotular_percentual(eixo)
        viz.apenas_grade_horizontal(eixo)
        fig.tight_layout()
        mostrar(fig)

    ui.separator(key="sep-perfil-2")
    titulo_secao("Perfil descritivo por modelo")

    perfil = dados["perfil"]
    ui.table(
        pd.DataFrame({
            "Modelo": perfil["modelo"],
            "Veículos": perfil["n_veiculos"],
            "Recalls": perfil["n_recalls"].astype(int),
            "Taxa": [pct(t, 1) for t in perfil["taxa_recall"]],
            "Idade média": [br(v, 1) for v in perfil["idade_media"]],
            "Km médio": [f"{int(v):,}".replace(",", ".") for v in perfil["km_medio"]],
            "Reclamações (média)": [br(v) for v in perfil["reclamacoes_media"]],
        }),
        key="tabela-perfil",
    )

# --------------------------------------------------------------------------- #
# Qualidade dos dados
# --------------------------------------------------------------------------- #

elif pagina == "Qualidade dos dados":
    titulo_secao(
        "Contrato de dados",
        "A avaliação de qualidade não é inspeção manual: é um contrato declarado "
        "em `pandera` que passa ou falha, persistido como artefato auditável.",
    )

    esquerda, direita = st.columns([2, 3])

    with esquerda:
        ui.table(
            dados["qualidade"].rename(columns={
                "verificacao": "Verificação", "valor": "Valor",
                "detalhe": "Detalhe",
            }),
            caption="Resultado da transformação raw → trusted",
            key="tabela-qualidade",
        )

    with direita:
        ui.table(
            pd.DataFrame(
                [
                    ("Duplicata (1 linha)", "Removida",
                     "Evita a mesma observação em treino e teste na validação cruzada."),
                    ("Outliers em reclamações (10)", "Mantidos",
                     "Cauda legítima de contagem; são os veículos de maior interesse."),
                    ("Valores ausentes", "Nenhuma ação",
                     "Não há nulos em nenhuma coluna."),
                    ("Alvo Sim/Não", "Convertido para booleano",
                     "Evita comparação por string — inclusive contra 'Não' com til."),
                    ("Coerência km × idade", "Aprovada",
                     "Rodagem anual implícita 100% dentro da faixa plausível."),
                    ("Domínio de `modelo`", "Validado",
                     "9 níveis fixos, verificados pelo contrato."),
                ],
                columns=["Achado", "Decisão", "Justificativa"],
            ),
            caption="Cada regra de limpeza responde a um achado da análise exploratória",
            key="tabela-decisoes",
        )

    ui.separator(key="sep-qualidade")
    titulo_secao("Distribuição das variáveis, por desfecho")

    rotulado = features.assign(
        alvo=transform.rotular_alvo(features[config.COLUNA_ALVO])
    )
    fig, eixos = plt.subplots(1, 3, figsize=(13.0, 3.6))

    for rotulo, sub in rotulado.groupby("alvo"):
        eixos[0].hist(sub["km"], bins=22, alpha=0.8,
                      color=viz.CORES_ALVO[rotulo], label=rotulo)
    eixos[0].set_title("Quilometragem")
    eixos[0].set_xlabel("km")
    eixos[0].set_ylabel("veículos")
    viz.rotular_milhares(eixos[0])
    eixos[0].legend(title="Recall")

    for eixo, coluna, titulo, unidade in (
        (eixos[1], "idade_veiculo", "Idade do veículo", "anos"),
        (eixos[2], "reclamacoes", "Reclamações", "ocorrências"),
    ):
        tabela = pd.crosstab(rotulado[coluna], rotulado["alvo"])
        posicoes = np.arange(len(tabela))
        largura = 0.42
        for deslocamento, rotulo in zip((-largura / 2, largura / 2), tabela.columns):
            eixo.bar(posicoes + deslocamento, tabela[rotulo], largura,
                     color=viz.CORES_ALVO[rotulo], label=rotulo)
        eixo.set_xticks(posicoes, tabela.index)
        eixo.set_title(titulo)
        eixo.set_xlabel(unidade)

    for eixo in eixos:
        viz.apenas_grade_horizontal(eixo)
    fig.tight_layout()
    mostrar(fig)

# --------------------------------------------------------------------------- #
# Desempenho do modelo
# --------------------------------------------------------------------------- #

elif pagina == "Desempenho do modelo":
    titulo_secao(
        "Simulador de custo — onde colocar o ponto de corte",
        "O limiar de 0,5 não é neutro: ele é o ótimo apenas quando os dois erros "
        "custam o mesmo.",
    )

    esquerda, direita = st.columns([1, 2.4])

    with esquerda:
        razao = st.slider(
            "Um recall não antecipado custa … inspeções desnecessárias",
            min_value=1.0,
            max_value=20.0,
            value=float(config.RAZAO_CUSTO_ANCORA),
            step=0.5,
            format="%.1f",
        )
        resultado = simular_custo(razao)

        ui.metric_card(label="Limiar de decisão", value=br(resultado["limiar"], 3),
                       description=f"razão {br(razao, 1)}:1", key="sim-limiar")
        ui.metric_card(
            label="Veículos sinalizados",
            value=f"{resultado['n_sinalizados']:.0f}",
            description=f"{pct(resultado['taxa_sinalizacao'])} da frota "
                        f"({n_veiculos} veículos)",
            key="sim-sinalizados",
        )

        if razao == config.RAZAO_CUSTO_ANCORA:
            ressalva(
                f"✓ Âncora adotada na entrega ({razao:.0f}:1)",
                "É o último ponto antes da degeneração: a Precision permanece "
                "acima da taxa base e o Recall segue executável.",
                "sim-ancora",
            )
        elif resultado["taxa_sinalizacao"] > 0.85:
            ressalva(
                "⚠ A recomendação deixou de selecionar",
                f"Com {br(razao, 1)}:1 o modelo sinaliza "
                f"{pct(resultado['taxa_sinalizacao'])} da frota e a Precision cai "
                f"para {br(resultado['precision'], 3)}, contra uma taxa base de "
                f"{br(taxa_base, 3)}. Inspecionar quase tudo não é priorizar.",
                "sim-degenerado",
            )

    with direita:
        # A referência é recalculada pelo mesmo caminho do valor exibido. Usar
        # `metricas_finais.parquet` aqui produziria variação não nula mesmo com
        # o controle parado na âncora, porque aquele artefato aplica o limiar
        # único já arredondado a todas as repetições, enquanto o simulador
        # reotimiza dentro de cada uma.
        ancora = simular_custo(float(config.RAZAO_CUSTO_ANCORA))

        colunas = st.columns(4)
        for coluna, metrica, rotulo in zip(
            colunas,
            ("accuracy", "precision", "recall", "f1"),
            ("Accuracy", "Precision", "Recall", "F1"),
        ):
            diferenca = resultado[metrica] - ancora[metrica]
            with coluna:
                ui.metric_card(
                    label=rotulo,
                    value=br(resultado[metrica], 3),
                    delta=f"{'+' if diferenca >= 0 else ''}{br(diferenca, 3)}",
                    description="vs. âncora 3:1",
                    key=f"sim-{metrica}",
                )

        ui.table(
            pd.DataFrame(
                [
                    ["Com recall", f"{resultado['verdadeiros_positivos']:.0f}",
                     f"{resultado['falsos_negativos']:.0f}"],
                    ["Sem recall", f"{resultado['falsos_positivos']:.0f}",
                     f"{resultado['verdadeiros_negativos']:.0f}"],
                ],
                columns=["Situação real", "Sinalizado", "Não sinalizado"],
            ),
            caption="Matriz de confusão no limiar simulado (média por repetição)",
            key="sim-confusao",
        )

        sensibilidade = dados["sensibilidade"]
        fig, eixo = plt.subplots(figsize=(8.4, 3.0))
        eixo.plot(sensibilidade["razao_custo"], sensibilidade["recall"], "o-",
                  color=viz.AZUL_PROFUNDO, markersize=5, label="Recall")
        eixo.plot(sensibilidade["razao_custo"], sensibilidade["precision"], "s-",
                  color=viz.AZUL, markersize=5, label="Precision")
        eixo.plot(sensibilidade["razao_custo"],
                  sensibilidade["taxa_sinalizacao"], "^--", color=viz.CINZA,
                  markersize=5, label="% da frota sinalizada")
        eixo.axhline(taxa_base, color=viz.COR_REFERENCIA, linestyle=":",
                     linewidth=1.3, label="taxa base")
        eixo.axvline(razao, color=viz.AZUL_PROFUNDO, linewidth=2.0, alpha=0.55)
        eixo.set_xscale("log")
        eixo.set_xticks(sensibilidade["razao_custo"],
                        [f"{int(r)}:1" for r in sensibilidade["razao_custo"]])
        eixo.set_xlabel("razão de custo FN:FP")
        viz.rotular_percentual(eixo)
        viz.apenas_grade_horizontal(eixo)
        eixo.legend(fontsize=8, loc="center right")
        fig.tight_layout()
        mostrar(fig)

    ui.separator(key="sep-desempenho-1")
    titulo_secao("Comparação das 12 combinações")

    esquerda, direita = st.columns([3, 2])

    with esquerda:
        comparacao = dados["comparacao"].sort_values("roc_auc").reset_index(drop=True)
        posicoes = np.arange(len(comparacao))
        erro = np.vstack([
            comparacao["roc_auc"] - comparacao["roc_auc_ic_inferior"],
            comparacao["roc_auc_ic_superior"] - comparacao["roc_auc"],
        ])
        cores = [viz.CINZA if m == "baseline" else viz.AZUL
                 for m in comparacao["modelo"]]
        melhor = dados["comparacao"].query("modelo != 'baseline'")["roc_auc"].max()
        erro_padrao = dados["comparacao"].loc[
            dados["comparacao"]["roc_auc"].idxmax(), "roc_auc_desvio"
        ] / np.sqrt(25)

        fig, eixo = plt.subplots(figsize=(8.0, 4.6))
        eixo.axvspan(melhor - erro_padrao, melhor, color=viz.AZUL, alpha=0.10,
                     label="1 erro padrão do melhor")
        eixo.errorbar(comparacao["roc_auc"], posicoes, xerr=erro, fmt="none",
                      ecolor=viz.TINTA_SUAVE, elinewidth=1.4, capsize=3)
        eixo.scatter(comparacao["roc_auc"], posicoes, color=cores, s=60, zorder=3)
        eixo.axvline(0.5, color=viz.COR_REFERENCIA, linestyle=":", linewidth=1.3,
                     label="acaso")
        eixo.set_yticks(
            posicoes,
            [f"{ROTULOS_MODELO[m]} · {ROTULOS_CONJUNTO[c]}"
             for m, c in zip(comparacao["modelo"], comparacao["conjunto"])],
        )
        eixo.set_xlabel("ROC AUC")
        eixo.grid(axis="y", visible=False)
        eixo.legend(loc="lower right")
        fig.tight_layout()
        mostrar(fig)

    with direita:
        testes = dados["testes"]
        ui.table(
            pd.DataFrame({
                "Combinação": [
                    f"{ROTULOS_MODELO[m]} · {ROTULOS_CONJUNTO[c]}"
                    for m, c in zip(testes["modelo"], testes["conjunto"])
                ],
                "Δ": [br(v, 4) for v in testes["diferenca_media"]],
                "p ingênuo": [br(v, 3) for v in testes["p_ingenuo"]],
                "p corrigido": [br(v, 3) for v in testes["p_corrigido"]],
                "Distinguível": ["sim" if d else "não" for d in testes["distinguivel"]],
            }),
            caption="Comparação pareada contra o modelo escolhido",
            max_height=300,
            key="tabela-testes",
        )
        ressalva(
            "As 25 dobras não são independentes",
            "Elas compartilham dados de treino. Com a correção de "
            "Nadeau-Bengio, o p ingênuo declarava 7 de 11 comparações "
            "significantes; o corrigido, apenas 2. Todas as combinações "
            "razoáveis são indistinguíveis entre si, e a escolha se justifica "
            "por parcimônia, não por superioridade.",
            "ressalva-nadeau",
        )

    ui.separator(key="sep-desempenho-2")
    esquerda, direita = st.columns(2)

    with esquerda:
        titulo_secao("Curvas de desempenho")
        roc, pr = dados["roc"], dados["pr"]
        fig, (cima, baixo) = plt.subplots(2, 1, figsize=(6.4, 5.6))

        cima.plot(roc["falso_positivo"], roc["verdadeiro_positivo"],
                  color=viz.AZUL, linewidth=2.2)
        cima.plot([0, 1], [0, 1], color=viz.COR_REFERENCIA, linestyle=":",
                  linewidth=1.3)
        cima.fill_between(roc["falso_positivo"], roc["verdadeiro_positivo"],
                          roc["falso_positivo"], color=viz.AZUL, alpha=0.13)
        cima.set_title(f"ROC — AUC {br(metadados['valor_metrica_selecao'], 3)}")
        cima.set_xlabel("falso positivo")
        cima.set_ylabel("verdadeiro positivo")

        baixo.plot(pr["revocacao"], pr["precisao"], color=viz.AZUL_PROFUNDO,
                   linewidth=2.2)
        baixo.axhline(taxa_base, color=viz.COR_REFERENCIA, linestyle=":",
                      linewidth=1.3, label=f"taxa base ({pct(taxa_base)})")
        baixo.set_title("Precision-Recall")
        baixo.set_xlabel("revocação")
        baixo.set_ylabel("precisão")
        baixo.legend(loc="lower left", fontsize=8)

        for eixo in (cima, baixo):
            eixo.set_xlim(0, 1)
            eixo.set_ylim(0, 1.02)
        fig.tight_layout()
        mostrar(fig)

    with direita:
        titulo_secao("Calibração")
        calib = dados["calibracao"]
        fig, eixo = plt.subplots(figsize=(6.4, 5.6))
        eixo.plot([0, 1], [0, 1], color=viz.COR_REFERENCIA, linestyle="--",
                  linewidth=1.4, label="calibração perfeita")
        eixo.scatter(calib["probabilidade_media"], calib["frequencia_observada"],
                     s=calib["n"] * 2.4, color=viz.AZUL, alpha=0.8, zorder=3)
        pior = calib.loc[calib["desvio"].abs().idxmax()]
        eixo.scatter([pior["probabilidade_media"]], [pior["frequencia_observada"]],
                     s=pior["n"] * 2.4, facecolor="none",
                     edgecolor=viz.AZUL_PROFUNDO, linewidth=2.4, zorder=4)
        eixo.set_xlabel("probabilidade prevista")
        eixo.set_ylabel("frequência observada")
        eixo.set_xlim(0, 1)
        eixo.set_ylim(0, 1)
        eixo.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        mostrar(fig)

        ressalva(
            "⚠ Calibração é o ponto fraco declarado",
            f"Erro absoluto médio de "
            f"{br(decisao['erro_absoluto_medio_calibracao'], 3)}. A faixa "
            f"destacada prevê {pct(pior['probabilidade_media'])} e observa "
            f"{pct(pior['frequencia_observada'])}. A otimização do limiar por "
            "custo pressupõe probabilidades honestas — no meio da escala elas "
            "não são.",
            "ressalva-calibracao",
        )

# --------------------------------------------------------------------------- #
# Interpretação
# --------------------------------------------------------------------------- #

elif pagina == "Interpretação":
    titulo_secao(
        "Score de um veículo",
        "Para um modelo linear a decomposição da predição tem forma fechada — é "
        "a mesma quantidade que o SHAP calcula, sem custo de amostragem.",
    )

    # Os limites vêm do observado, e não do domínio de negócio. Um controle que
    # fosse até 30 anos convidaria a extrapolar para regiões onde o modelo nunca
    # viu um único veículo e a curva logística segue subindo por construção, não
    # por evidência.
    idade_max = int(features["idade_veiculo"].max())
    km_min = int(features["km"].min() // 1_000 * 1_000)
    km_max = int(np.ceil(features["km"].max() / 1_000) * 1_000)
    recl_max = int(features["reclamacoes"].max())

    entrada, saida = st.columns([1, 2.2])

    with entrada:
        idade = st.slider("Idade do veículo (anos)", 0, idade_max, min(5, idade_max))
        km = st.slider("Quilometragem", km_min, km_max,
                       int(features["km"].median() // 1_000 * 1_000), step=1_000)
        reclamacoes = st.slider("Reclamações registradas", 0, recl_max,
                                min(4, recl_max))
        # O separador de milhar é aplicado só ao número: aplicá-lo à frase
        # inteira trocaria também as vírgulas do texto por pontos.
        km_formatado = f"{km_max:,}".replace(",", ".")
        nota(
            f"Faixas limitadas ao observado na base (idade ≤ {idade_max} anos, "
            f"km ≤ {km_formatado}, reclamações ≤ {recl_max}). "
            "Fora dela o modelo extrapolaria."
        )

    veiculo = pd.DataFrame(
        [{"idade_veiculo": idade, "km": km, "reclamacoes": reclamacoes}]
    ).loc[:, list(metadados["colunas"])]

    risco = float(pipeline.predict_proba(veiculo)[0, 1])
    contribuicoes, referencia = explain.contribuicoes_individuais(pipeline, veiculo)

    with entrada:
        ui.metric_card(
            label="Risco previsto",
            value=pct(risco, 1),
            description=f"limiar vigente: {br(limiar_vigente, 3)}",
            key="score-risco",
        )
        ui.badges(
            [
                ui.BadgeItem(
                    text="Sinalizar para inspeção" if risco >= limiar_vigente
                    else "Não sinalizar",
                    variant="default" if risco >= limiar_vigente else "secondary",
                )
            ],
            key="score-decisao",
        )

    with saida:
        serie = contribuicoes.iloc[0].sort_values()
        rotulos = {
            "idade_veiculo": "Idade do veículo",
            "km": "Quilometragem",
            "reclamacoes": "Reclamações",
        }

        fig, eixo = plt.subplots(figsize=(8.0, 3.4))
        posicoes = np.arange(len(serie))
        cores = [viz.AZUL if valor > 0 else viz.CINZA for valor in serie]
        eixo.barh(posicoes, serie.to_numpy(), color=cores, height=0.55)
        eixo.axvline(0, color=viz.TINTA, linewidth=1.2)
        for posicao, valor in zip(posicoes, serie.to_numpy()):
            eixo.text(
                valor + (0.04 if valor >= 0 else -0.04), posicao,
                f"{'+' if valor >= 0 else ''}{br(valor)}", va="center",
                ha="left" if valor >= 0 else "right",
                fontsize=9, color=viz.TINTA, fontweight="semibold",
            )
        eixo.set_yticks(posicoes, [rotulos.get(i, i) for i in serie.index])
        eixo.set_xlabel("contribuição para o log-odds do risco")
        eixo.set_title("Por que este veículo recebeu esse score")
        eixo.grid(axis="y", visible=False)
        margem = max(abs(serie).max() * 1.45, 0.5)
        eixo.set_xlim(-margem, margem)
        fig.tight_layout()
        mostrar(fig)

        nota(
            f"Ponto de partida (veículo médio da frota): {br(referencia)} em "
            "log-odds. Barras à direita empurram o veículo para o risco; à "
            "esquerda, para longe dele."
        )

    ui.separator(key="sep-interp-1")
    esquerda, direita = st.columns([3, 2])

    with esquerda:
        titulo_secao("Importância das variáveis")
        fig, (um, dois) = plt.subplots(1, 2, figsize=(9.2, 3.6))
        for eixo, tabela, titulo in (
            (um, dados["imp_individual"], "Uma variável por vez"),
            (dois, dados["imp_grupos"], "Colineares em bloco"),
        ):
            ordem = tabela.sort_values("queda_auc")
            posicoes = np.arange(len(ordem))
            erro = np.vstack([
                ordem["queda_auc"] - ordem["ic_inferior"],
                ordem["ic_superior"] - ordem["queda_auc"],
            ])
            cores = [viz.AZUL_PROFUNDO if n > 1 else viz.AZUL
                     for n in ordem["n_variaveis"]]
            eixo.barh(posicoes, ordem["queda_auc"], color=cores, height=0.55)
            eixo.errorbar(ordem["queda_auc"], posicoes, xerr=erro, fmt="none",
                          ecolor=viz.TINTA_SUAVE, elinewidth=1.3, capsize=3)
            eixo.set_yticks(posicoes,
                            [g.replace("_", " ") for g in ordem["grupo"]])
            eixo.set_xlabel("queda de ROC AUC")
            eixo.set_title(titulo)
            eixo.set_xlim(0, 0.175)
            eixo.grid(axis="y", visible=False)

        soma = dados["imp_individual"].set_index("grupo").loc[
            ["idade_veiculo", "km"], "queda_auc"
        ].sum()
        bloco = dados["imp_grupos"].set_index("grupo").loc["tempo_e_uso", "queda_auc"]
        dois.annotate(f"{br(bloco / soma)}× a soma das partes",
                      xy=(0.5, 0.12), xycoords="axes fraction", fontsize=9.5,
                      color=viz.AZUL_PROFUNDO, fontweight="semibold")
        fig.tight_layout()
        mostrar(fig)
        nota(
            "Com `corr(idade, km) = 0,947`, permutar uma variável de cada vez "
            "**subestima as duas** — a outra cobre a ausência. A leitura "
            "individual, saída padrão de qualquer biblioteca, está errada aqui."
        )

    with direita:
        titulo_secao("Efeito em razão de chances")
        coeficientes = dados["coeficientes"]
        ui.table(
            pd.DataFrame({
                "Variável": coeficientes["variavel"],
                "Razão de chances": [
                    f"{br(v, 3)}×" for v in coeficientes["razao_chances_comunicavel"]
                ],
                "Unidade": coeficientes["unidade"],
            }),
            key="tabela-coeficientes",
        )
        ressalva(
            "Não ler idade e km isoladamente",
            "Sob colinearidade de 0,947 a repartição do efeito entre as duas "
            "depende da amostra, não do fenômeno. A leitura confiável é a do "
            "bloco tempo_e_uso.",
            "ressalva-coeficientes",
        )

    ui.separator(key="sep-interp-2")
    titulo_secao("Risco previsto em função de cada variável")

    dependencia = dados["dependencia"]
    variaveis = list(metadados["colunas"])
    titulos = {
        "idade_veiculo": ("Idade do veículo", "anos"),
        "km": ("Quilometragem", "km"),
        "reclamacoes": ("Reclamações", "ocorrências"),
    }

    fig, eixos = plt.subplots(1, len(variaveis), figsize=(13.0, 3.4))
    for eixo, variavel in zip(eixos, variaveis):
        curva = dependencia[dependencia["variavel"] == variavel]
        titulo, unidade = titulos[variavel]
        eixo.plot(curva["valor"], curva["risco_previsto"], color=viz.AZUL,
                  linewidth=2.4)
        eixo.fill_between(curva["valor"], 0, curva["risco_previsto"],
                          color=viz.AZUL, alpha=0.10)
        eixo.axhline(limiar_vigente, color=viz.AZUL_PROFUNDO, linestyle="--",
                     linewidth=1.4)
        eixo.set_title(titulo)
        eixo.set_xlabel(unidade)
        eixo.set_ylim(0, 1)
        viz.rotular_percentual(eixo)
        viz.apenas_grade_horizontal(eixo)
    eixos[0].set_ylabel("risco previsto")
    viz.rotular_milhares(eixos[1])
    fig.tight_layout()
    mostrar(fig)
    nota(
        "A dependência parcial avalia combinações que quase não ocorrem na "
        "frota — um veículo de 8 anos com 10 mil km, por exemplo. A leitura vale "
        "na região densa dos dados, não nas pontas."
    )

# --------------------------------------------------------------------------- #
# Rodapé
# --------------------------------------------------------------------------- #

ui.separator(key="sep-rodape")
st.caption(
    "Desafio técnico · Bolsista Cientista de Dados · dados sintéticos fornecidos "
    "no enunciado. Nenhum número descreve a frota real da Stellantis."
)
