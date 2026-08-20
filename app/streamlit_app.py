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
O layout usa `streamlit-shadcn-ui` para cartoes de metrica, abas, distintivos,
separadores e tabelas.

Os cartoes de ressalva sao componente proprio, e nao `ui.alert`, por dois
motivos: o shadcn so oferece as variantes `default` e `destructive` (vermelha),
e renderiza em shadow DOM, onde nao ha como mirar instancias especificas.

Sobre a cor desses cartoes: a paleta institucional nao tem cor de alerta, e a
primeira versao resolveu isso com enfase apenas por peso de texto. Na pratica o
aviso competia de menos pela atencao. As cores de estado -- ambar para aviso,
verde para confirmacao -- ficam separadas das institucionais de proposito: elas
comunicam *severidade*, nao identidade de marca. Os graficos seguem sem cor
quente, com risco codificado por valor, conforme `viz.py`.

Os graficos sao Plotly, e nao matplotlib. Os documentos continuam em matplotlib
por precisarem renderizar identicamente em HTML e PDF sem motor JavaScript; aqui
a leitura ganha com hover, zoom e legenda clicavel. Ambos os motores consomem as
mesmas constantes de `viz.py`, entao paleta e semantica de cor sao identicas --
muda so o mecanismo de desenho.

Uso:

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from plotly.subplots import make_subplots

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

viz.aplicar_tema_plotly()

#: Barra de ferramentas enxuta. `scrollZoom` fica desligado de proposito: com
#: varios graficos empilhados, capturar a rolagem da pagina dentro de um deles
#: torna a navegacao imprevisivel.
CONFIG_GRAFICO = {
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
}

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
[data-testid="stSlider"] [role="slider"] {{
    border-color: {viz.AZUL_PROFUNDO};
}}

/* Cartões de ressalva.
   O `ui.alert` do shadcn só oferece as variantes `default` e `destructive`
   (vermelha), e renderiza em shadow DOM -- não há como mirar instâncias
   específicas de fora. Daí o componente próprio: um aviso precisa da coloração
   convencional de aviso para ser lido como tal. */
.ressalva {{
    display: flex;
    align-items: flex-start;
    gap: 0.72rem;
    padding: 0.85rem 1.05rem;
    margin: 0.5rem 0 1.1rem;
    border-radius: 6px;
    border-left: 4px solid var(--ressalva-cor);
    background: var(--ressalva-fundo);
    font-size: 0.885rem;
    line-height: 1.55;
    color: #4A4E57;
}}
.ressalva__icone {{
    flex: 0 0 auto;
    margin-top: 0.12rem;
    line-height: 0;
}}
.ressalva__titulo {{
    display: block;
    margin-bottom: 0.18rem;
    font-weight: 650;
    color: #2A2E36;
}}
.ressalva code {{
    background: rgba(0, 0, 0, 0.05);
    padding: 0.05rem 0.28rem;
    border-radius: 3px;
    font-size: 0.86em;
}}

/* Frase de abertura da página: a conclusão antes da evidência. */
.abertura {{
    font-size: 1.02rem;
    line-height: 1.6;
    color: #4A4E57;
    margin: 0.2rem 0 1.1rem;
    padding-left: 0.9rem;
    border-left: 3px solid {viz.AZUL};
}}
.abertura strong {{
    color: {viz.AZUL_PROFUNDO};
    font-weight: 650;
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


def grafico(figura: go.Figure, altura: int = 380, chave: str | None = None) -> None:
    """Renderiza uma figura Plotly com a configuracao padrao do painel.

    `theme=None` e essencial, e nao um detalhe: o padrao do Streamlit e
    `theme="streamlit"`, que aplica o tema *dele* por cima do template da
    figura. Com o padrao, a cor de fonte volta para o cinza do Streamlit e a
    caixa de hover para o fundo branco -- ou seja, o tema institucional montado
    em `viz.py` seria silenciosamente descartado.
    """
    figura.update_layout(height=altura)
    st.plotly_chart(
        figura, width="stretch", theme=None, config=CONFIG_GRAFICO, key=chave
    )


def nota(texto: str) -> None:
    """Ressalva metodologica discreta, junto do grafico a que se refere."""
    st.caption(texto)


# --------------------------------------------------------------------------- #
# Cartoes de ressalva
#
# A paleta institucional nao traz cor de alerta, e a primeira versao do painel
# resolveu isso com enfase apenas por peso de texto. Na pratica o aviso competia
# de menos pela atencao -- risco que ficou registrado na epoca e se confirmou.
# Um bloco que existe para impedir uma leitura errada precisa da coloracao
# convencional de aviso.
#
# As cores de estado ficam separadas das institucionais de proposito: elas
# comunicam *severidade*, nao identidade de marca. O tipo `info` e a excecao --
# ali o azul institucional cabe, porque o conteudo e contexto e nao alerta.
# --------------------------------------------------------------------------- #

AMBAR_AVISO = "#F0B400"
FUNDO_AVISO = "#FEF9E7"
VERDE_OK = "#2E7D32"
FUNDO_OK = "#EDF7EE"

_ICONES = {
    "aviso": (
        '<circle cx="10" cy="10" r="9" fill="{cor}"/>'
        '<rect x="9" y="4.8" width="2" height="6.4" rx="1" fill="#fff"/>'
        '<circle cx="10" cy="14.4" r="1.2" fill="#fff"/>'
    ),
    "info": (
        '<circle cx="10" cy="10" r="9" fill="{cor}"/>'
        '<circle cx="10" cy="5.8" r="1.2" fill="#fff"/>'
        '<rect x="9" y="8.6" width="2" height="6.6" rx="1" fill="#fff"/>'
    ),
    "sucesso": (
        '<circle cx="10" cy="10" r="9" fill="{cor}"/>'
        '<path d="M6 10.3l2.6 2.6L14.2 7.3" stroke="#fff" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
    ),
}

ESTILOS_RESSALVA = {
    "aviso": (AMBAR_AVISO, FUNDO_AVISO),
    "info": (viz.AZUL, viz.FUNDO_DESTAQUE),
    "sucesso": (VERDE_OK, FUNDO_OK),
}


def ressalva(titulo: str, descricao: str, tipo: str = "aviso") -> None:
    """Cartao de ressalva com a aparencia convencional do seu tipo.

    Args:
        titulo: frase curta que resume o que precisa ser notado.
        descricao: o porque, em texto corrido.
        tipo: `aviso` (ambar) para o que pode induzir leitura errada,
            `info` (azul institucional) para contexto, `sucesso` (verde) para
            confirmacao.
    """
    cor, fundo = ESTILOS_RESSALVA[tipo]
    icone = _ICONES[tipo].format(cor=cor)

    st.markdown(
        f'<div class="ressalva" style="--ressalva-cor:{cor};'
        f'--ressalva-fundo:{fundo}">'
        f'<span class="ressalva__icone">'
        f'<svg width="19" height="19" viewBox="0 0 20 20">{icone}</svg>'
        f"</span>"
        f'<span><span class="ressalva__titulo">{titulo}</span>{descricao}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def abertura(mensagem: str) -> None:
    """Frase de abertura da pagina: a conclusao antes da evidencia.

    O painel e enviado sem acompanhamento -- quem o abre nao tem quem explique.
    Cada pagina precisa dizer na primeira linha o que ela conclui, e so entao
    mostrar como chegou la. Sem isso, o leitor tem que reconstruir a conclusao
    a partir dos graficos, e a maioria nao vai fazer isso.
    """
    st.markdown(f'<div class="abertura">{mensagem}</div>', unsafe_allow_html=True)


def titulo_secao(texto: str, apoio: str | None = None) -> None:
    """Cabecalho de secao.

    O titulo enuncia o *achado*, nao a categoria do grafico. "Ranking por
    modelo" obriga o leitor a descobrir sozinho o que o ranking diz; "O modelo
    do veiculo nao separa risco" ja entrega a leitura, e o grafico vira a prova.
    """
    st.markdown(f"#### {texto}")
    if apoio:
        st.caption(apoio)


def linha_referencia(
    figura: go.Figure,
    *,
    y: float | None = None,
    x: float | None = None,
    rotulo: str = "",
    **kwargs,
) -> None:
    """Linha tracejada de referencia (acaso, taxa base, limiar)."""
    comum = dict(
        line=dict(color=viz.COR_REFERENCIA, dash="dash", width=1.4),
        annotation_text=rotulo,
        annotation_position="top left",
        annotation_font=dict(color=viz.TINTA_SUAVE, size=10),
        **kwargs,
    )
    if y is not None:
        figura.add_hline(y=y, **comum)
    if x is not None:
        figura.add_vline(x=x, **comum)


#: Artefatos sem os quais o painel nao tem o que mostrar. Sao versionados
#: justamente porque o ambiente publicado nao roda o pipeline.
ARTEFATOS_OBRIGATORIOS = (
    config.MART_FEATURES,
    config.MART_RANKING_RISCO,
    config.MODELO_FINAL,
    config.MODELO_METADADOS,
    config.DECISAO_OPERACIONAL,
    config.CURVA_GANHO,
)


def exigir_artefatos() -> None:
    """Interrompe com diagnostico util se a camada de dados nao existir.

    Sem esta checagem, um artefato ausente produz um `FileNotFoundError` cru no
    meio da pagina -- pessima experiencia em ambiente publicado, onde quem abre
    o painel nao tem acesso ao terminal nem sabe o que e um Parquet.
    """
    ausentes = [caminho for caminho in ARTEFATOS_OBRIGATORIOS if not caminho.is_file()]
    if not ausentes:
        return

    st.error("Os artefatos de dados não foram encontrados.")
    st.markdown(
        "Este painel **lê** a camada `mart` e as métricas já calculadas; ele não "
        "treina nada. Para gerá-los a partir de `data/raw/`:\n\n"
        "```bash\n"
        "uv run recall-pipeline\n"
        "uv run python -m stellantis_recall.modeling.train\n"
        "uv run python -m stellantis_recall.modeling.evaluate\n"
        "uv run python -m stellantis_recall.modeling.explain\n"
        "```"
    )
    with st.expander("Arquivos ausentes"):
        for caminho in ausentes:
            st.code(str(caminho.relative_to(config.PROJECT_ROOT)), language="text")
    st.stop()


exigir_artefatos()

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
    st.markdown("**STELLANTIS**")
    st.caption(
        "Movidos pela nossa diversidade, desenvolvemos juntos nossas atividades,"
        " com respeito, ética, construindo e cuidando do futuro."
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

    abertura(
        "<strong>Inspecionar os 30% de maior risco encontra metade dos "
        "recalls.</strong> O modelo ordena a frota por idade, quilometragem "
        "e reclamações — três dados que a garantia já registra."
    )

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
        "Quanto a sua capacidade alcança",
        "A restrição de pós-vendas não é *quais* veículos inspecionar — é quantos "
        "cabem na oficina. Ajuste e veja o resultado.",
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
        ui.metric_card(label="Recalls encontrados", value=f"{encontrados}",
                       description=f"de {recalls_totais} existentes na base",
                       key="cap-encontrados")
        ui.metric_card(label="Cobertura", value=pct(captura),
                       description=f"{br(lift)}× melhor que sortear ao acaso",
                       key="cap-cobertura")

    with direita:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="inspeção aleatória",
            mode="lines", line=dict(color=viz.COR_REFERENCIA, dash="dash", width=1.6),
            hovertemplate="Ao acaso: %{x:.0%} da frota → %{y:.0%} dos recalls"
                          "<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0.0], ganho["pct_frota"]]),
            y=np.concatenate([[0.0], ganho["pct_recalls_capturados"]]),
            name="priorização pelo modelo",
            mode="lines+markers",
            line=dict(color=viz.AZUL, width=3),
            marker=dict(size=8),
            fill="tonexty", fillcolor="rgba(36,56,130,0.13)",
            customdata=np.column_stack([
                np.concatenate([[0], ganho["n_inspecionados"]]),
                np.concatenate([[0], ganho["recalls_capturados"]]),
                np.concatenate([[1.0], ganho["lift"]]),
            ]),
            hovertemplate="<b>Inspecionar %{x:.0%} da frota</b><br>"
                          "%{customdata[0]} veículos<br>"
                          "encontra %{y:.0%} dos recalls (%{customdata[1]})<br>"
                          "ganho de %{customdata[2]:.2f}× sobre o acaso"
                          "<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[fracao], y=[captura], name="capacidade escolhida",
            mode="markers",
            marker=dict(size=16, color=viz.AZUL_PROFUNDO, symbol="diamond"),
            hovertemplate=f"<b>Sua capacidade</b><br>{capacidade} veículos "
                          f"({pct(fracao)})<br>{encontrados} recalls "
                          f"({pct(captura)})<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title="fração da frota inspecionada", tickformat=".0%",
                       range=[0, 1]),
            yaxis=dict(title="recalls encontrados", tickformat=".0%",
                       range=[0, 1.02]),
            legend=dict(orientation="h", y=-0.22, x=0),
            hovermode="x unified",
        )
        grafico(fig, altura=420, chave="g-ganho")

    ui.separator(key="sep-exec-2")
    titulo_secao(
        "O risco se concentra em veículos maduros e rodados",
        "Veículos com 6+ anos e mais de 125 mil km: 84% de recall. Novos com até "
        "25 mil km: 6%.",
    )

    esquerda, direita = st.columns([3, 2])

    with esquerda:
        segmentos = dados["segmentos"]
        pivo = segmentos.pivot(index="faixa_idade", columns="faixa_km",
                               values="taxa_recall")
        contagem = segmentos.pivot(index="faixa_idade", columns="faixa_km",
                                   values="n_veiculos")
        reclamacoes = segmentos.pivot(index="faixa_idade", columns="faixa_km",
                                      values="reclamacoes_media")
        indices = [i for i in ORDEM_FAIXA_IDADE if i in pivo.index]
        colunas_km = [k for k in ORDEM_FAIXA_KM if k in pivo.columns]
        pivo = pivo.reindex(index=indices, columns=colunas_km)
        contagem = contagem.reindex(index=indices, columns=colunas_km)
        reclamacoes = reclamacoes.reindex(index=indices, columns=colunas_km)

        insuficiente = contagem.fillna(0) < MINIMO_AMOSTRA
        exibida = pivo.mask(insuficiente)
        rotulos = np.where(
            insuficiente.to_numpy() | pivo.isna().to_numpy(),
            "n/d",
            np.vectorize(lambda v: pct(v) if pd.notna(v) else "")(pivo.to_numpy()),
        )

        fig = go.Figure(go.Heatmap(
            z=exibida.to_numpy(),
            x=[c.replace(" mil km", "k").replace("acima de ", ">").replace("ate ", "≤")
               for c in pivo.columns],
            y=[i.split(" (")[0] for i in pivo.index],
            colorscale=viz.ESCALA_RISCO_PLOTLY,
            zmin=0, zmax=1,
            text=rotulos,
            texttemplate="%{text}",
            textfont=dict(size=13),
            customdata=np.dstack([
                contagem.fillna(0).to_numpy(),
                reclamacoes.fillna(0).to_numpy(),
            ]),
            hovertemplate="<b>%{y} · %{x}</b><br>"
                          "taxa de recall: %{z:.1%}<br>"
                          "veículos no segmento: %{customdata[0]:.0f}<br>"
                          "reclamações (média): %{customdata[1]:.2f}"
                          "<extra></extra>",
            colorbar=dict(title="taxa", tickformat=".0%", thickness=14),
            hoverongaps=False,
        ))
        fig.update_layout(
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="", showgrid=False, autorange="reversed"),
        )
        grafico(fig, altura=330, chave="g-segmentos")
        nota(
            f"**n/d** = menos de {MINIMO_AMOSTRA} veículos no segmento. A taxa "
            "existe, mas não sustenta decisão."
        )

    with direita:
        ranking = dados["ranking"].sort_values("taxa_recall")
        fig = go.Figure(go.Bar(
            x=ranking["taxa_recall"], y=ranking["modelo"], orientation="h",
            marker=dict(
                color=ranking["taxa_recall"], colorscale=viz.ESCALA_RISCO_PLOTLY,
                cmin=0.2, cmax=0.8, line=dict(color=viz.GRADE, width=1),
            ),
            error_x=dict(
                type="data", symmetric=False,
                array=ranking["ic_superior"] - ranking["taxa_recall"],
                arrayminus=ranking["taxa_recall"] - ranking["ic_inferior"],
                color=viz.AZUL_PROFUNDO, thickness=1.6, width=5,
            ),
            customdata=np.column_stack([
                ranking["n_veiculos"], ranking["n_recalls"],
                ranking["ic_inferior"], ranking["ic_superior"],
            ]),
            hovertemplate="<b>%{y}</b><br>"
                          "taxa: %{x:.1%}<br>"
                          "IC 95%: %{customdata[2]:.1%} – %{customdata[3]:.1%}<br>"
                          "%{customdata[1]:.0f} recalls em %{customdata[0]:.0f} "
                          "veículos<extra></extra>",
        ))
        linha_referencia(fig, x=taxa_base, rotulo=f"global {pct(taxa_base, 1)}")
        fig.update_layout(
            xaxis=dict(title="taxa de recall", tickformat=".0%"),
            yaxis=dict(title="", showgrid=False),
            title="Ranking por modelo",
        )
        grafico(fig, altura=330, chave="g-ranking")

        ressalva(
            "Não priorize por modelo de veículo",
            "As barras de erro se sobrepõem: o intervalo do primeiro colocado "
            "cobre o do último. O teste estatístico não encontra associação "
            "(p = 0,570). Esta ordem é ruído, não risco.",
            "aviso",
        )

# --------------------------------------------------------------------------- #
# Perfil da frota
# --------------------------------------------------------------------------- #

elif pagina == "Perfil da frota":
    abertura(
        "<strong>O risco não está no modelo do veículo. Está no tempo de uso "
        "e nas reclamações.</strong> Filtre abaixo para examinar qualquer "
        "recorte da frota."
    )

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
            "Nenhum veículo neste recorte",
            "Remova ao menos um filtro.",
            "aviso",
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
            "Amostra pequena demais para decidir",
            f"São {n_recorte} veículos, e a taxa varia em "
            f"{pct(ic_sup[0] - ic_inf[0])} para mais ou para menos. Largo demais "
            "para priorizar.",
            "aviso",
        )

    ui.separator(key="sep-perfil-1")
    esquerda, direita = st.columns(2)

    with esquerda:
        titulo_secao("O risco sobe de 5% para 85% ao longo da vida")
        evolucao = dados["evolucao"]
        ic_inf_e, ic_sup_e = eda.intervalo_wilson(
            evolucao["n_recalls"], evolucao["n_veiculos"]
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([evolucao["idade_veiculo"],
                              evolucao["idade_veiculo"][::-1]]),
            y=np.concatenate([ic_sup_e, ic_inf_e[::-1]]),
            fill="toself", fillcolor="rgba(36,56,130,0.16)",
            line=dict(width=0), name="IC 95% (Wilson)",
            hoverinfo="skip", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=evolucao["idade_veiculo"], y=evolucao["taxa_recall"],
            mode="lines+markers", name="taxa por idade",
            line=dict(color=viz.AZUL, width=3), marker=dict(size=9),
            customdata=np.column_stack([
                evolucao["n_veiculos"], evolucao["n_recalls"], ic_inf_e, ic_sup_e,
            ]),
            hovertemplate="<b>%{x} ano(s)</b><br>"
                          "taxa: %{y:.1%}<br>"
                          "IC 95%: %{customdata[2]:.1%} – %{customdata[3]:.1%}<br>"
                          "%{customdata[1]:.0f} recalls em %{customdata[0]:.0f} "
                          "veículos<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=evolucao["idade_veiculo"], y=evolucao["taxa_recall_acumulada"],
            mode="lines+markers", name="acumulado da frota",
            line=dict(color=viz.TINTA_SUAVE, width=1.8, dash="dash"),
            marker=dict(size=6, symbol="square"),
            hovertemplate="acumulado até %{x} ano(s): %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title="idade do veículo (anos)", dtick=1),
            yaxis=dict(title="taxa de recall", tickformat=".0%", range=[0, 1]),
            legend=dict(orientation="h", y=-0.25, x=0),
        )
        grafico(fig, altura=390, chave="g-evolucao")
        nota(
            "Leitura de coorte: a frota hoje, por idade. Não é tendência ao "
            "longo do tempo — a base não tem datas."
        )

    with direita:
        titulo_secao("A partir da terceira reclamação, o risco dobra")
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

        fig = go.Figure(go.Scatter(
            x=robustos["reclamacoes"], y=robustos["taxa"],
            mode="lines+markers", name="taxa de recall",
            line=dict(color=viz.AZUL, width=3), marker=dict(size=10),
            error_y=dict(
                type="data", symmetric=False,
                array=ic_sup_r - robustos["taxa"],
                arrayminus=robustos["taxa"] - ic_inf_r,
                color=viz.TINTA_SUAVE, thickness=1.5, width=5,
            ),
            customdata=np.column_stack([robustos["n"], robustos["n_recalls"]]),
            hovertemplate="<b>%{x} reclamação(ões)</b><br>"
                          "taxa: %{y:.1%}<br>"
                          "%{customdata[1]:.0f} recalls em %{customdata[0]:.0f} "
                          "veículos<extra></extra>",
        ))
        fig.add_vline(
            x=config.LIMIAR_RECLAMACOES - 0.5,
            line=dict(color=viz.AZUL_PROFUNDO, dash="dash", width=1.8),
            annotation_text=f"degrau em {config.LIMIAR_RECLAMACOES}",
            annotation_position="top right",
            annotation_font=dict(color=viz.AZUL_PROFUNDO, size=11),
        )
        fig.update_layout(
            xaxis=dict(title="reclamações registradas", dtick=1),
            yaxis=dict(title="taxa de recall", tickformat=".0%", range=[0, 1]),
            showlegend=False,
        )
        grafico(fig, altura=390, chave="g-reclamacoes")
        nota(
            "Exibidos apenas os níveis com 10 ou mais veículos."
        )

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
    abertura(
        "<strong>Os dados passam em todos os testes do contrato.</strong> "
        "500 registros, nenhum valor ausente, uma duplicata removida. Abaixo, a "
        "evidência e o porquê de cada decisão de limpeza."
    )

    titulo_secao(
        "O contrato passou",
        "Qualidade aqui não é inspeção manual: é um contrato declarado em "
        "`pandera`, que passa ou falha e deixa registro auditável.",
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
    titulo_secao(
        "Veículos com recall se concentram à direita nas três variáveis",
        "Clique na legenda para isolar um dos desfechos.",
    )

    rotulado = features.assign(
        alvo=transform.rotular_alvo(features[config.COLUNA_ALVO])
    )
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Quilometragem", "Idade do veículo", "Reclamações"),
        horizontal_spacing=0.07,
    )

    for rotulo, sub in rotulado.groupby("alvo"):
        fig.add_trace(
            go.Histogram(
                x=sub["km"], name=rotulo, nbinsx=22, opacity=0.85,
                marker_color=viz.CORES_ALVO[rotulo], legendgroup=rotulo,
                hovertemplate=f"<b>{rotulo}</b><br>km: %{{x}}<br>"
                              "veículos: %{y}<extra></extra>",
            ),
            row=1, col=1,
        )

    for coluna_dado, coluna_grafico in (("idade_veiculo", 2), ("reclamacoes", 3)):
        tabela = pd.crosstab(rotulado[coluna_dado], rotulado["alvo"])
        for rotulo in tabela.columns:
            fig.add_trace(
                go.Bar(
                    x=tabela.index, y=tabela[rotulo], name=rotulo,
                    marker_color=viz.CORES_ALVO[rotulo], legendgroup=rotulo,
                    showlegend=False,
                    hovertemplate=f"<b>{rotulo}</b><br>%{{x}}<br>"
                                  "veículos: %{y}<extra></extra>",
                ),
                row=1, col=coluna_grafico,
            )

    fig.update_layout(
        barmode="group", bargap=0.18,
        legend=dict(orientation="h", y=-0.18, x=0, title="Recall"),
    )
    fig.update_xaxes(title_text="km", row=1, col=1)
    fig.update_xaxes(title_text="anos", dtick=1, row=1, col=2)
    fig.update_xaxes(title_text="ocorrências", dtick=1, row=1, col=3)
    fig.update_yaxes(title_text="veículos", row=1, col=1)
    grafico(fig, altura=390, chave="g-distribuicoes")

# --------------------------------------------------------------------------- #
# Desempenho do modelo
# --------------------------------------------------------------------------- #

elif pagina == "Desempenho do modelo":
    abertura(
        "<strong>Dados dois veículos, o modelo acerta qual tem mais risco em 79% "
        "das vezes.</strong> Suficiente para priorizar a fila de inspeção, "
        "insuficiente para excluir veículos. Esta página mostra onde ele falha."
    )

    titulo_secao(
        "Onde colocar o ponto de corte",
        "O limiar de 0,5 só é ótimo quando os dois erros custam o mesmo — e aqui "
        "não custam. Ajuste a razão e veja a decisão mudar.",
    )

    esquerda, direita = st.columns([1, 2.4])

    with esquerda:
        razao = st.slider(
            "Um recall não antecipado custa … inspeções desnecessárias",
            min_value=1.0, max_value=20.0,
            value=float(config.RAZAO_CUSTO_ANCORA), step=0.5, format="%.1f",
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
                f"Este é o ponto adotado na entrega ({razao:.0f}:1)",
                "É o último corte antes de o modelo passar a sinalizar quase "
                "toda a frota.",
                "sucesso",
            )
        elif resultado["taxa_sinalizacao"] > 0.85:
            ressalva(
                "Aqui o modelo deixa de selecionar",
                f"Sinaliza {pct(resultado['taxa_sinalizacao'])} da frota com "
                f"precisão de {br(resultado['precision'], 2)} — contra "
                f"{br(taxa_base, 2)} de quem sorteia ao acaso. Inspecionar quase "
                "tudo não é priorizar.",
                "aviso",
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
                    label=rotulo, value=br(resultado[metrica], 3),
                    delta=f"{'+' if diferenca >= 0 else ''}{br(diferenca, 3)}",
                    description="vs. âncora 3:1", key=f"sim-{metrica}",
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
        fig = go.Figure()
        for coluna, nome, cor, tracejado in (
            ("recall", "Recall", viz.AZUL_PROFUNDO, None),
            ("precision", "Precision", viz.AZUL, None),
            ("taxa_sinalizacao", "% da frota sinalizada", viz.CINZA, "dash"),
        ):
            fig.add_trace(go.Scatter(
                x=sensibilidade["razao_custo"], y=sensibilidade[coluna],
                name=nome, mode="lines+markers",
                line=dict(color=cor, width=2.4, dash=tracejado),
                marker=dict(size=8),
                customdata=sensibilidade["limiar"],
                hovertemplate=f"<b>{nome}</b><br>razão %{{x}}:1<br>"
                              "valor: %{y:.1%}<br>"
                              "limiar: %{customdata:.3f}<extra></extra>",
            ))
        linha_referencia(fig, y=taxa_base, rotulo=f"taxa base {pct(taxa_base, 1)}")
        fig.add_vline(
            x=razao, line=dict(color=viz.AZUL_PROFUNDO, width=2.5),
            opacity=0.45,
        )
        fig.update_layout(
            xaxis=dict(title="razão de custo FN:FP", type="log",
                       tickvals=sensibilidade["razao_custo"],
                       ticktext=[f"{int(r)}:1" for r in sensibilidade["razao_custo"]]),
            yaxis=dict(title="", tickformat=".0%", range=[0, 1.05]),
            legend=dict(orientation="h", y=-0.28, x=0),
            hovermode="x unified",
        )
        grafico(fig, altura=330, chave="g-sensibilidade")

    ui.separator(key="sep-desempenho-1")
    titulo_secao(
        "Nenhuma combinação é melhor que as outras",
        "Doze combinações de modelo e atributos, todas com intervalos que se "
        "sobrepõem. Passe o cursor para ler as métricas de cada uma.",
    )

    esquerda, direita = st.columns([3, 2])

    with esquerda:
        comparacao = dados["comparacao"].sort_values("roc_auc").reset_index(drop=True)
        rotulos = [
            f"{ROTULOS_MODELO[m]} · {ROTULOS_CONJUNTO[c]}"
            for m, c in zip(comparacao["modelo"], comparacao["conjunto"])
        ]
        melhor = dados["comparacao"].query("modelo != 'baseline'")["roc_auc"].max()
        erro_padrao = dados["comparacao"].loc[
            dados["comparacao"]["roc_auc"].idxmax(), "roc_auc_desvio"
        ] / np.sqrt(25)

        fig = go.Figure()
        fig.add_vrect(
            x0=melhor - erro_padrao, x1=melhor,
            fillcolor=viz.AZUL, opacity=0.10, line_width=0,
            annotation_text="1 erro padrão do melhor",
            annotation_position="top left",
            annotation_font=dict(color=viz.TINTA_SUAVE, size=10),
        )
        fig.add_trace(go.Scatter(
            x=comparacao["roc_auc"], y=rotulos, mode="markers",
            marker=dict(
                size=13,
                color=[viz.CINZA if m == "baseline" else viz.AZUL
                       for m in comparacao["modelo"]],
            ),
            error_x=dict(
                type="data", symmetric=False,
                array=comparacao["roc_auc_ic_superior"] - comparacao["roc_auc"],
                arrayminus=comparacao["roc_auc"] - comparacao["roc_auc_ic_inferior"],
                color=viz.TINTA_SUAVE, thickness=1.5, width=5,
            ),
            customdata=np.column_stack([
                comparacao["n_features"], comparacao["accuracy"],
                comparacao["precision"], comparacao["recall"],
                comparacao["f1"], comparacao["brier"],
                comparacao["roc_auc_ic_inferior"], comparacao["roc_auc_ic_superior"],
            ]),
            hovertemplate="<b>%{y}</b><br>"
                          "ROC AUC: %{x:.4f}<br>"
                          "IC 95%: %{customdata[6]:.3f} – %{customdata[7]:.3f}<br>"
                          "atributos: %{customdata[0]:.0f}<br>"
                          "<br>Accuracy: %{customdata[1]:.3f}<br>"
                          "Precision: %{customdata[2]:.3f}<br>"
                          "Recall: %{customdata[3]:.3f}<br>"
                          "F1: %{customdata[4]:.3f}<br>"
                          "Brier: %{customdata[5]:.4f}<extra></extra>",
            showlegend=False,
        ))
        linha_referencia(fig, x=0.5, rotulo="acaso")
        fig.update_layout(
            xaxis=dict(title="ROC AUC"),
            yaxis=dict(title="", showgrid=False),
        )
        grafico(fig, altura=470, chave="g-comparacao")

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
            "Por que nenhuma diferença conta como real",
            "As 25 dobras compartilham dados de treino, então o teste comum "
            "exagera a significância. Corrigida essa dependência, 9 das 11 "
            "comparações deixam de ser significativas. Entre equivalentes, "
            "escolhemos o modelo mais simples.",
            "info",
        )

    ui.separator(key="sep-desempenho-2")
    esquerda, direita = st.columns(2)

    with esquerda:
        titulo_secao(
            "O preço de capturar mais recalls",
            "Para chegar a 90% de captura, a precisão cai para cerca de 60%. "
            "O cursor mostra o limiar de cada ponto.",
        )
        roc, pr = dados["roc"], dados["pr"]
        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.16,
            subplot_titles=(
                f"ROC — AUC {br(metadados['valor_metrica_selecao'], 3)}",
                "Precision-Recall",
            ),
        )
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
            line=dict(color=viz.COR_REFERENCIA, dash="dot", width=1.4),
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=roc["falso_positivo"], y=roc["verdadeiro_positivo"],
            mode="lines", showlegend=False, fill="tozeroy",
            fillcolor="rgba(36,56,130,0.12)",
            line=dict(color=viz.AZUL, width=2.6),
            customdata=roc["limiar"],
            hovertemplate="limiar %{customdata:.3f}<br>"
                          "falso positivo: %{x:.1%}<br>"
                          "verdadeiro positivo: %{y:.1%}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pr["revocacao"], y=pr["precisao"], mode="lines", showlegend=False,
            line=dict(color=viz.AZUL_PROFUNDO, width=2.6),
            customdata=pr["limiar"],
            hovertemplate="limiar %{customdata:.3f}<br>"
                          "revocação: %{x:.1%}<br>"
                          "precisão: %{y:.1%}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(
            y=taxa_base, row=2, col=1,
            line=dict(color=viz.COR_REFERENCIA, dash="dot", width=1.4),
        )
        fig.update_xaxes(title_text="falso positivo", tickformat=".0%",
                         range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="verdadeiro positivo", tickformat=".0%",
                         range=[0, 1.02], row=1, col=1)
        fig.update_xaxes(title_text="revocação", tickformat=".0%",
                         range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="precisão", tickformat=".0%",
                         range=[0, 1.02], row=2, col=1)
        grafico(fig, altura=560, chave="g-curvas")

    with direita:
        titulo_secao("A probabilidade acerta nos extremos, erra no meio")
        calib = dados["calibracao"]
        pior = calib.loc[calib["desvio"].abs().idxmax()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="calibração perfeita",
            line=dict(color=viz.COR_REFERENCIA, dash="dash", width=1.5),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=calib["probabilidade_media"], y=calib["frequencia_observada"],
            mode="markers", name="faixas de risco",
            marker=dict(
                size=calib["n"], sizemode="area",
                sizeref=2.0 * calib["n"].max() / (44.0**2), sizemin=6,
                color=viz.AZUL, opacity=0.8,
                line=dict(color=viz.AZUL_PROFUNDO, width=1),
            ),
            customdata=np.column_stack([calib["n"], calib["desvio"]]),
            hovertemplate="prevista: %{x:.1%}<br>"
                          "observada: %{y:.1%}<br>"
                          "desvio: %{customdata[1]:+.1%}<br>"
                          "veículos na faixa: %{customdata[0]:.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[pior["probabilidade_media"]], y=[pior["frequencia_observada"]],
            mode="markers", name="pior faixa",
            marker=dict(size=30, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color=viz.AZUL_PROFUNDO, width=2.6)),
            hoverinfo="skip",
        ))
        fig.update_layout(
            xaxis=dict(title="probabilidade prevista", tickformat=".0%",
                       range=[0, 1]),
            yaxis=dict(title="frequência observada", tickformat=".0%",
                       range=[0, 1]),
            legend=dict(orientation="h", y=-0.22, x=0),
        )
        grafico(fig, altura=560, chave="g-calibracao")

        ressalva(
            "Este é o ponto fraco do modelo",
            f"Na faixa destacada, ele prevê "
            f"{pct(pior['probabilidade_media'])} de risco e observa "
            f"{pct(pior['frequencia_observada'])}. O erro médio é de "
            f"{br(decisao['erro_absoluto_medio_calibracao'] * 100, 1)} pontos. "
            "O ponto de corte por custo supõe probabilidade honesta.",
            "aviso",
        )

# --------------------------------------------------------------------------- #
# Interpretação
# --------------------------------------------------------------------------- #

elif pagina == "Interpretação":
    abertura(
        "<strong>Cada reclamação registrada aumenta em 27% a chance de "
        "recall.</strong> Tempo de uso e quilometragem, lidos em conjunto, "
        "pesam quase o dobro disso."
    )

    titulo_secao(
        "Simule um veículo",
        "Informe as três características e veja o risco, a decisão e quanto cada "
        "fator contribuiu.",
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
            f"Limites do que a base contém: até {idade_max} anos, "
            f"{km_formatado} km e {recl_max} reclamações. Fora disso o modelo "
            "estaria adivinhando."
        )

    veiculo = pd.DataFrame(
        [{"idade_veiculo": idade, "km": km, "reclamacoes": reclamacoes}]
    ).loc[:, list(metadados["colunas"])]

    risco = float(pipeline.predict_proba(veiculo)[0, 1])
    contribuicoes, referencia = explain.contribuicoes_individuais(pipeline, veiculo)

    with entrada:
        ui.metric_card(
            label="Risco previsto", value=pct(risco, 1),
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
        valores_veiculo = {
            "idade_veiculo": f"{idade} ano(s)",
            "km": f"{km:,}".replace(",", ".") + " km",
            "reclamacoes": f"{reclamacoes} reclamação(ões)",
        }

        fig = go.Figure(go.Bar(
            x=serie.to_numpy(), y=[rotulos.get(i, i) for i in serie.index],
            orientation="h",
            marker_color=[viz.AZUL if v > 0 else viz.CINZA for v in serie],
            text=[f"{'+' if v >= 0 else ''}{br(v)}" for v in serie],
            textposition="outside",
            textfont=dict(color=viz.TINTA, size=12),
            customdata=[valores_veiculo.get(i, "") for i in serie.index],
            hovertemplate="<b>%{y}</b><br>%{customdata}<br>"
                          "contribuição: %{x:+.3f} em log-odds<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=viz.TINTA, width=1.4))
        margem = max(abs(serie).max() * 1.6, 0.5)
        fig.update_layout(
            xaxis=dict(title="contribuição para o log-odds do risco",
                       range=[-margem, margem]),
            yaxis=dict(title="", showgrid=False),
            title="Por que este veículo recebeu esse score",
            showlegend=False,
        )
        grafico(fig, altura=320, chave="g-contribuicoes")

        nota(
            "Barras à direita empurram o veículo para o risco; à esquerda, "
            f"para longe dele. O ponto de partida — um veículo médio — é "
            f"{br(referencia)}."
        )

    ui.separator(key="sep-interp-1")
    esquerda, direita = st.columns([3, 2])

    with esquerda:
        titulo_secao(
            "Medir uma variável por vez leva à conclusão errada",
            "À esquerda, o método padrão. À direita, o correto para variáveis "
            "que carregam a mesma informação.",
        )
        fig = make_subplots(
            rows=1, cols=2, horizontal_spacing=0.22,
            subplot_titles=("Uma variável por vez", "Colineares em bloco"),
        )
        for coluna, tabela in ((1, dados["imp_individual"]), (2, dados["imp_grupos"])):
            ordem = tabela.sort_values("queda_auc")
            fig.add_trace(
                go.Bar(
                    x=ordem["queda_auc"],
                    y=[g.replace("_", " ") for g in ordem["grupo"]],
                    orientation="h", showlegend=False,
                    marker_color=[viz.AZUL_PROFUNDO if n > 1 else viz.AZUL
                                  for n in ordem["n_variaveis"]],
                    error_x=dict(
                        type="data", symmetric=False,
                        array=ordem["ic_superior"] - ordem["queda_auc"],
                        arrayminus=ordem["queda_auc"] - ordem["ic_inferior"],
                        color=viz.TINTA_SUAVE, thickness=1.5, width=5,
                    ),
                    customdata=np.column_stack([
                        ordem["ic_inferior"], ordem["ic_superior"],
                        ordem["n_variaveis"],
                    ]),
                    hovertemplate="<b>%{y}</b><br>"
                                  "queda de ROC AUC: %{x:.4f}<br>"
                                  "IC 95%: %{customdata[0]:.4f} – "
                                  "%{customdata[1]:.4f}<br>"
                                  "variáveis no grupo: %{customdata[2]:.0f}"
                                  "<extra></extra>",
                ),
                row=1, col=coluna,
            )

        soma = dados["imp_individual"].set_index("grupo").loc[
            ["idade_veiculo", "km"], "queda_auc"
        ].sum()
        bloco = dados["imp_grupos"].set_index("grupo").loc["tempo_e_uso", "queda_auc"]
        fig.add_annotation(
            text=f"<b>{br(bloco / soma)}× a soma das partes</b>",
            xref="x2 domain", yref="y2 domain", x=0.5, y=0.08,
            showarrow=False, font=dict(color=viz.AZUL_PROFUNDO, size=12),
        )
        fig.update_xaxes(title_text="queda de ROC AUC", range=[0, 0.175])
        fig.update_yaxes(showgrid=False)
        grafico(fig, altura=340, chave="g-importancia")
        nota(
            "Ao remover só a idade, a quilometragem cobre a ausência — e "
            "vice-versa. Cada uma parece pouco importante. Juntas, valem 1,73× "
            "a soma das partes."
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
            "Não leia idade e quilometragem separadamente",
            "As duas correlacionam 0,947 — medem quase a mesma coisa. Como o "
            "efeito se divide entre elas muda conforme a amostra. Leia o bloco "
            "tempo e uso.",
            "aviso",
        )

    ui.separator(key="sep-interp-2")
    titulo_secao(
        "A partir de que ponto um veículo entra na fila",
        "Onde a curva cruza a linha tracejada, o veículo passa a ser sinalizado.",
    )

    dependencia = dados["dependencia"]
    variaveis = list(metadados["colunas"])
    titulos = {
        "idade_veiculo": ("Idade do veículo", "anos"),
        "km": ("Quilometragem", "km"),
        "reclamacoes": ("Reclamações", "ocorrências"),
    }

    fig = make_subplots(
        rows=1, cols=len(variaveis), horizontal_spacing=0.07,
        subplot_titles=[titulos[v][0] for v in variaveis],
    )
    for coluna, variavel in enumerate(variaveis, start=1):
        curva = dependencia[dependencia["variavel"] == variavel]
        _, unidade = titulos[variavel]
        fig.add_trace(
            go.Scatter(
                x=curva["valor"], y=curva["risco_previsto"], mode="lines",
                showlegend=False, fill="tozeroy",
                fillcolor="rgba(36,56,130,0.10)",
                line=dict(color=viz.AZUL, width=3),
                hovertemplate=f"%{{x:,.0f}} {unidade}<br>"
                              "risco previsto: %{y:.1%}<extra></extra>",
            ),
            row=1, col=coluna,
        )
        fig.add_hline(
            y=limiar_vigente, row=1, col=coluna,
            line=dict(color=viz.AZUL_PROFUNDO, dash="dash", width=1.5),
        )
        fig.update_xaxes(title_text=unidade, row=1, col=coluna)
        fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=coluna)
    fig.update_yaxes(title_text="risco previsto", row=1, col=1)
    grafico(fig, altura=340, chave="g-dependencia")
    nota(
        "As pontas das curvas descrevem combinações raras na frota — um "
        "veículo de 8 anos com 10 mil km, por exemplo. Leia o miolo."
    )

# --------------------------------------------------------------------------- #
# Rodapé
# --------------------------------------------------------------------------- #

ui.separator(key="sep-rodape")
st.caption(
    "Feito com dados sintéticos. "
    "Nenhum número descreve a frota real da Stellantis."
)
