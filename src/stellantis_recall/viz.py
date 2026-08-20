"""Identidade visual unica dos entregaveis.

Relatorio, PDF executivo e dashboard consomem daqui. Centralizar o tema evita o
efeito mais comum em entregas de analise -- tres documentos com tres paletas --
e faz com que a cor tenha significado consistente em todos eles.

Matplotlib foi escolhido por renderizar identicamente em HTML e em PDF sem
dependencia de exportador externo. A qualidade visual vem do tema, nao da
biblioteca.

Sobre a paleta institucional
----------------------------
Os cinco tons corporativos sao a base, e nenhum matiz novo foi introduzido. Duas
lacunas precisaram de tratamento, ambas resolvidas por *tint* dos proprios tons:

**Cinza medio para texto secundario.** O #BCBCBC tem razao de contraste de 1,9:1
sobre branco -- reprovado em qualquer criterio de legibilidade. Rotulos de eixo
usam um tint de 70% do grafite institucional, que sobe o contraste para ~5,3:1
sem sair da matiz.

**Cor de alerta.** A paleta nao tem vermelho, ambar ou qualquer cor quente. Em
vez de introduzir uma, a enfase de risco e feita por **valor**: quanto mais
escuro, maior o risco (#BCBCBC -> #243882 -> #00133B). Linhas de referencia
(acaso, taxa base) usam grafite tracejado em vez de vermelho. O custo dessa
decisao e real e vale registrar: um alerta em escala monocromatica compete menos
pela atencao do que um alerta colorido, entao a sinalizacao de problema se apoia
mais em anotacao textual e peso de traco.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from . import config

# --------------------------------------------------------------------------- #
# Paleta institucional
# --------------------------------------------------------------------------- #

AZUL_PROFUNDO = "#00133B"
AZUL = "#243882"
GRAFITE = "#282B34"
CINZA = "#BCBCBC"
BRANCO = "#FFFFFF"

PALETA_INSTITUCIONAL = (AZUL_PROFUNDO, AZUL, GRAFITE, CINZA, BRANCO)

# --------------------------------------------------------------------------- #
# Tints derivados
#
# Calculados por mistura linear com branco, preservando a matiz de origem. Nao
# sao cores novas -- sao os mesmos tons em outra intensidade.
# --------------------------------------------------------------------------- #

#: Grafite a 70% sobre branco. Contraste ~5,3:1 -- legivel como texto secundario.
TINTA_SUAVE = "#686B71"

#: Cinza institucional a 40% sobre branco. Linhas de grade discretas.
GRADE = "#E4E4E4"

#: Azul institucional a 45% sobre branco. Series secundarias e preenchimentos.
AZUL_CLARO = "#9BA4C6"

#: Azul profundo a 12% sobre branco. Fundo de destaque.
FUNDO_DESTAQUE = "#E6E8EE"

TINTA = GRAFITE

# --------------------------------------------------------------------------- #
# Papeis semanticos
#
# A cor carrega significado fixo em todos os documentos: azul institucional e
# sempre "com recall" / risco, cinza e sempre "sem recall" / neutro.
# --------------------------------------------------------------------------- #

COR_POSITIVO = AZUL
COR_NEGATIVO = CINZA
COR_DESTAQUE = AZUL_PROFUNDO
COR_REFERENCIA = GRAFITE  # linhas de acaso, taxa base, calibracao perfeita

CORES_ALVO: dict[str, str] = {
    config.ROTULO_NEGATIVO: COR_NEGATIVO,
    config.ROTULO_POSITIVO: COR_POSITIVO,
}

#: Ciclo para series categoricas. Alterna matiz e valor para manter distincao
#: mesmo em impressao monocromatica.
SEQUENCIA = (AZUL, CINZA, AZUL_PROFUNDO, TINTA_SUAVE, AZUL_CLARO, GRAFITE)

# --------------------------------------------------------------------------- #
# Escalas continuas
# --------------------------------------------------------------------------- #

#: Sequencial de risco: claro = baixo, escuro = alto.
MAPA_RISCO = LinearSegmentedColormap.from_list(
    "risco", [BRANCO, CINZA, AZUL, AZUL_PROFUNDO]
)

#: Divergente para correlacoes. Os extremos se distinguem por *saturacao*
#: (grafite dessaturado contra azul saturado), ja que a paleta nao oferece duas
#: matizes opostas. Menos imediato que um vermelho-azul convencional, porem
#: institucionalmente consistente.
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list(
    "divergente", [GRAFITE, CINZA, BRANCO, AZUL, AZUL_PROFUNDO]
)


# --------------------------------------------------------------------------- #
# Tema
# --------------------------------------------------------------------------- #


def aplicar_tema() -> None:
    """Aplica o tema do projeto ao matplotlib. Chamar uma vez por documento."""
    mpl.rcParams.update(
        {
            "figure.figsize": (8.0, 4.2),
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": BRANCO,
            "axes.facecolor": BRANCO,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": AZUL_PROFUNDO,
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": TINTA_SUAVE,
            "axes.edgecolor": GRADE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=list(SEQUENCIA)),
            "grid.color": GRADE,
            "grid.linewidth": 0.8,
            "text.color": TINTA,
            "xtick.color": TINTA_SUAVE,
            "ytick.color": TINTA_SUAVE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def apenas_grade_horizontal(eixo: plt.Axes) -> None:
    """Deixa so a grade do eixo Y: linhas verticais raramente ajudam a leitura."""
    eixo.grid(axis="x", visible=False)
    eixo.grid(axis="y", visible=True)


def rotular_percentual(eixo: plt.Axes, casas: int = 0) -> None:
    """Formata o eixo Y como percentual."""
    eixo.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=casas))


def rotular_percentual_x(eixo: plt.Axes, casas: int = 0) -> None:
    """Formata o eixo X como percentual."""
    eixo.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=casas))


def rotular_milhares(eixo: plt.Axes) -> None:
    """Abrevia o eixo X em milhares: 120000 -> 120k."""
    eixo.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda valor, _: f"{valor / 1000:.0f}k")
    )


def cor_por_risco(taxa: float, minimo: float = 0.0, maximo: float = 1.0) -> str:
    """Cor da escala sequencial correspondente a uma taxa de risco."""
    amplitude = maximo - minimo
    posicao = 0.0 if amplitude == 0 else (taxa - minimo) / amplitude
    return mpl.colors.to_hex(MAPA_RISCO(min(max(posicao, 0.0), 1.0)))


# --------------------------------------------------------------------------- #
# Tema Plotly
#
# Os documentos (relatorio, PDF, painel Quarto) continuam em matplotlib: eles
# precisam renderizar identicamente em HTML e em PDF, sem depender de motor
# JavaScript. O painel interativo usa Plotly, porque ali a leitura ganha com
# hover e zoom.
#
# Os dois motores compartilham *as mesmas constantes* deste modulo, entao a
# identidade visual e a semantica de cor -- azul e risco, cinza e neutro -- se
# mantem identicas. O que muda e apenas o mecanismo de desenho.
# --------------------------------------------------------------------------- #

#: Escala continua de risco, equivalente a `MAPA_RISCO` em formato Plotly.
ESCALA_RISCO_PLOTLY = [
    [0.00, BRANCO],
    [0.33, CINZA],
    [0.72, AZUL],
    [1.00, AZUL_PROFUNDO],
]


def tema_plotly():
    """Constroi o template Plotly do projeto.

    Importa o Plotly de forma preguicosa: os modulos de dados e de modelagem nao
    dependem dele, e so o painel interativo chama esta funcao.
    """
    import plotly.graph_objects as go

    eixo = dict(
        gridcolor=GRADE,
        linecolor=GRADE,
        zeroline=False,
        showline=True,
        ticks="outside",
        tickcolor=GRADE,
        tickfont=dict(color=TINTA_SUAVE, size=11),
        title=dict(font=dict(color=TINTA_SUAVE, size=12)),
        automargin=True,
    )

    return go.layout.Template(
        layout=go.Layout(
            # Formato numerico brasileiro: virgula decimal, ponto para milhar.
            # Vale para eixos, rotulos e caixas de hover de uma so vez.
            separators=",.",
            font=dict(
                family="Segoe UI, -apple-system, Helvetica Neue, Arial, sans-serif",
                color=TINTA,
                size=12,
            ),
            paper_bgcolor=BRANCO,
            plot_bgcolor=BRANCO,
            colorway=list(SEQUENCIA),
            colorscale=dict(sequential=ESCALA_RISCO_PLOTLY),
            title=dict(
                font=dict(color=AZUL_PROFUNDO, size=15),
                x=0.0,
                xanchor="left",
            ),
            xaxis=eixo,
            yaxis=eixo,
            hoverlabel=dict(
                bgcolor=AZUL_PROFUNDO,
                bordercolor=AZUL_PROFUNDO,
                font=dict(color=BRANCO, size=12),
                align="left",
            ),
            hovermode="closest",
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=11),
            ),
            margin=dict(l=10, r=10, t=44, b=10),
        )
    )


def aplicar_tema_plotly(nome: str = "stellantis") -> str:
    """Registra o template como padrao. Chamar uma vez por sessao do painel."""
    import plotly.io as pio

    pio.templates[nome] = tema_plotly()
    pio.templates.default = nome
    return nome


def salvar(figura: Figure, nome: str, diretorio: Path | None = None) -> Path:
    """Persiste a figura em `outputs/figures/` e devolve o caminho gravado."""
    diretorio = diretorio or config.FIGURES_DIR
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / f"{nome}.png"
    figura.savefig(caminho)
    return caminho
