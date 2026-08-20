"""Identidade visual unica dos entregaveis.

Relatorio, PDF executivo e dashboard consomem daqui. Centralizar o tema evita o
efeito mais comum em entregas de analise -- tres documentos com tres paletas --
e faz com que a cor tenha significado consistente: azul e sempre "sem recall",
ambar e sempre "com recall".

Matplotlib foi escolhido por renderizar identicamente em HTML e em PDF sem
dependencia de exportador externo. A qualidade visual vem do tema, nao da
biblioteca.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import config

# --------------------------------------------------------------------------- #
# Paleta
# --------------------------------------------------------------------------- #

TINTA = "#1b2430"  # texto e eixos
TINTA_SUAVE = "#5b6672"  # rotulos secundarios
GRADE = "#e3e7ec"

AZUL = "#1f4e79"  # classe negativa / neutro
AMBAR = "#c8760a"  # classe positiva / atencao
VERMELHO = "#a4243b"  # alerta
VERDE = "#3d6b52"  # confirmacao
CINZA = "#8a949e"

COR_NEGATIVO = AZUL
COR_POSITIVO = AMBAR

CORES_ALVO: dict[str, str] = {
    config.ROTULO_NEGATIVO: COR_NEGATIVO,
    config.ROTULO_POSITIVO: COR_POSITIVO,
}

SEQUENCIA = (AZUL, AMBAR, VERDE, VERMELHO, CINZA, "#6b4c9a", "#0f7c8c")


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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
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


def salvar(figura: Figure, nome: str, diretorio: Path | None = None) -> Path:
    """Persiste a figura em `outputs/figures/` e devolve o caminho gravado."""
    diretorio = diretorio or config.FIGURES_DIR
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / f"{nome}.png"
    figura.savefig(caminho)
    return caminho
