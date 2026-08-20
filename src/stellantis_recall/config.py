"""Configuracao central do projeto.

Concentra caminhos, semente aleatoria e premissas de negocio em um unico lugar,
para que nenhum caminho ou constante fique escondido dentro de um notebook.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #


def _encontrar_raiz(inicio: Path) -> Path:
    """Sobe a arvore de diretorios ate achar o pyproject.toml do projeto.

    Torna os caminhos independentes do diretorio de trabalho: o pacote e
    importado tanto pelo terminal (na raiz) quanto pelos .qmd (dentro de qmd/).
    """
    for candidato in (inicio, *inicio.parents):
        if (candidato / "pyproject.toml").is_file():
            return candidato
    raise RuntimeError(f"pyproject.toml nao encontrado a partir de {inicio}")


PACOTE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _encontrar_raiz(PACOTE_DIR)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRUSTED_DIR = DATA_DIR / "trusted"
MART_DIR = DATA_DIR / "mart"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

SQL_DIR = PACOTE_DIR / "sql"

# Arquivos nomeados
RAW_XLSX = RAW_DIR / "Dataset_Sintetico_Recall_500_Registros.xlsx"
TRUSTED_VEICULOS = TRUSTED_DIR / "veiculos.parquet"
MART_FEATURES = MART_DIR / "features.parquet"
MART_PERFIL_MODELO = MART_DIR / "perfil_por_modelo.parquet"
MART_EVOLUCAO_IDADE = MART_DIR / "evolucao_por_idade.parquet"
MART_RANKING_RISCO = MART_DIR / "ranking_risco.parquet"
RELATORIO_QUALIDADE = METRICS_DIR / "quality_report.parquet"

DIRETORIOS_GERADOS = (TRUSTED_DIR, MART_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR)


def garantir_diretorios() -> None:
    """Cria os diretorios de saida se ainda nao existirem."""
    for diretorio in DIRETORIOS_GERADOS:
        diretorio.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #

SEED = 42

# --------------------------------------------------------------------------- #
# Dominio de negocio
#
# Os limites abaixo sao *plausibilidade de negocio*, nao o intervalo observado
# na amostra. Um contrato colado no minimo/maximo dos 500 registros quebraria
# em qualquer carga futura; a validacao precisa distinguir "valor novo" de
# "valor impossivel".
# --------------------------------------------------------------------------- #

COLUNA_ALVO = "recall"
ROTULO_POSITIVO = "Sim"
# Escapado como ã de proposito: o valor na origem e "Nao" com til, e um
# arquivo salvo em codificacao errada transformaria isso num bug silencioso.
ROTULO_NEGATIVO = "Não"
ROTULOS_ALVO = (ROTULO_POSITIVO, ROTULO_NEGATIVO)

MODELOS = (
    "Argo",
    "Commander",
    "Compass",
    "Cronos",
    "Fastback",
    "Pulse",
    "Renegade",
    "Strada",
    "Toro",
)

IDADE_MIN, IDADE_MAX = 0, 30
KM_MIN, KM_MAX = 0, 500_000
RECLAMACOES_MIN, RECLAMACOES_MAX = 0, 100

# Rodagem anual plausivel (km/ano). Usado apenas como *alerta* de consistencia
# na analise exploratoria, nunca como criterio de exclusao automatica.
KM_ANO_MIN, KM_ANO_MAX = 2_000, 60_000

# Limiar de reclamacoes derivado da Parte 1: a taxa de recall salta de 28,0%
# (2 reclamacoes) para 58,3% (3 reclamacoes) e depois estabiliza. O degrau
# sugere um ponto de corte no processo gerador, e vira variavel indicadora.
LIMIAR_RECLAMACOES = 3

# A idade e registrada em anos completos (piso). Um veiculo marcado com N anos
# tem idade real esperada de N + 0,5. Somar meio ano antes de dividir evita a
# divisao por zero em veiculos novos e mantem a razao km/ano sem vies sistematico.
OFFSET_IDADE = 0.5

# --------------------------------------------------------------------------- #
# Premissas de decisao
#
# PENDENTE DE CONFIRMACAO com a area de negocio. A razao entre os custos e o
# unico parametro que define o ponto de corte do modelo; deixa-la explicita e
# editavel aqui e proposital -- o alternativo seria o limiar implicito de 0.5,
# que assume custos simetricos sem nunca declarar isso.
#
#   FN (falso negativo): recall nao antecipado -> campanha tardia, exposicao
#                        de seguranca, custo de imagem.
#   FP (falso positivo): inspecao preventiva desnecessaria -> custo de oficina.
# --------------------------------------------------------------------------- #

CUSTO_FALSO_NEGATIVO = 10.0
CUSTO_FALSO_POSITIVO = 1.0

# --------------------------------------------------------------------------- #
# Estatistica
# --------------------------------------------------------------------------- #

NIVEL_CONFIANCA = 0.95
Z_NIVEL_CONFIANCA = 1.959963984540054  # quantil normal para 95%
ALFA = 1 - NIVEL_CONFIANCA
