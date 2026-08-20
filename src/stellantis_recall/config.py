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

MODELO_FINAL = MODELS_DIR / "model.joblib"
MODELO_METADADOS = MODELS_DIR / "modelo.json"
METRICAS_DOBRAS = METRICS_DIR / "cv_folds.parquet"
COMPARACAO_MODELOS = METRICS_DIR / "comparacao_modelos.parquet"
PREDICOES_OOF = METRICS_DIR / "predicoes_oof.parquet"
TESTES_PAREADOS = METRICS_DIR / "testes_pareados.parquet"
METRICAS_FINAIS = METRICS_DIR / "metricas_finais.parquet"
SENSIBILIDADE_CUSTO = METRICS_DIR / "sensibilidade_custo.parquet"
CURVA_ROC = METRICS_DIR / "curva_roc.parquet"
CURVA_PR = METRICS_DIR / "curva_pr.parquet"
CALIBRACAO = METRICS_DIR / "calibracao.parquet"
CURVA_GANHO = METRICS_DIR / "curva_ganho.parquet"
MATRIZ_CONFUSAO = METRICS_DIR / "matriz_confusao.parquet"
DECISAO_OPERACIONAL = METRICS_DIR / "decisao_operacional.json"

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

# Offset somado a idade antes de qualquer divisao por tempo. Evita divisao por
# zero em veiculos novos -- mas o valor nao e arbitrario, e foi escolhido por
# medicao.
#
# O candidato natural era 0,5, pela idade real esperada de um veiculo registrado
# com N anos completos. Os dados rejeitam esse valor: a relacao ajustada e
# km ~ 15.753 + 14.685 * idade, e esse intercepto de quase 16 mil km equivale a
# um ano de rodagem ja acumulado no veiculo marcado como "0 anos". Com offset
# 0,5, a coorte de idade zero terminava com 36,5 mil km/ano contra ~16 mil das
# demais, e km_por_ano passava a correlacionar -0,389 com a idade -- ou seja,
# reencodava a dimensao colinear em vez de separar-se dela.
#
#   offset   corr(km_por_ano, idade)   assimetria   amplitude entre coortes
#     0,5            -0,389               3,61              20.924
#     1,0            -0,107               1,79               3.998
#     1,5            +0,196               0,00               2.435
#
# 1,0 absorve o intercepto e deixa km_por_ano praticamente ortogonal a idade,
# que e exatamente o papel pretendido para essa variavel.
OFFSET_IDADE = 1.0

# --------------------------------------------------------------------------- #
# Premissas de decisao
#
# O modelo devolve probabilidade; transforma-la em decisao exige um ponto de
# corte. O limiar default de 0,5 nao e neutro -- ele e o otimo apenas quando os
# dois erros custam o mesmo, premissa que ninguem declara e que aqui e falsa:
#
#   FN (falso negativo): recall nao antecipado -> campanha tardia, exposicao
#                        de seguranca, custo de imagem.
#   FP (falso positivo): inspecao preventiva desnecessaria -> custo de oficina.
#
# So a *razao* entre os custos e identificavel: multiplicar ambos por uma
# constante nao move o limiar otimo. Por isso a razao e o parametro primario e
# os custos derivam dela, e nao o contrario.
# --------------------------------------------------------------------------- #

#: Grade varrida pela analise de sensibilidade (`modeling/evaluate.py`). O
#: resultado e um entregavel por si: mostra em que faixa a decisao e estavel e
#: onde ela vira.
RAZOES_CUSTO_SENSIBILIDADE: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)

# Razao ancora. As quatro metricas exigidas pelo enunciado (Accuracy, Precision,
# Recall, F1) so existem depois de escolhido um limiar, entao uma ancora e
# obrigatoria -- a curva de sensibilidade a acompanha, nao a substitui.
#
# 3:1 foi escolhida por medicao, nao por convencao. Diagnostico preliminar
# (Random Forest, predicoes fora da amostra em 5-fold, 499 veiculos):
#
#   razao   limiar   Precision   Recall   F1      inspecoes exigidas
#    1:1    0,360      0,646      0,887   0,748      328  (66% da frota)
#    3:1    0,290      0,619      0,925   0,742      357  (72%)
#    5:1    0,130      0,546      0,975   0,700      427  (86%)
#   10:1    0,095      0,537      0,979   0,693      436  (87%)
#   20:1    0,025      0,495      0,996   0,661      481  (96%)
#
# 3:1 e o ultimo ponto antes da degeneracao: a partir de 5:1 a recomendacao
# converge para "inspecionar a frota inteira", e a Precision cai para a taxa
# base de 0,479 -- ou seja, o modelo deixa de selecionar qualquer coisa. Em 3:1
# a Precision de 0,619 ainda esta bem acima da taxa base, o Recall de 0,925 e
# operacionalmente util, e o F1 fica a 0,006 do maximo observado na faixa.
#
# Premissa declarada, nao medida: nao ha dado de custo real neste desafio. O
# valor e editavel, e a curva de sensibilidade existe justamente para que a
# area de negocio possa reposicionar a decisao sem refazer a analise.
RAZAO_CUSTO_ANCORA = 3.0

CUSTO_FALSO_POSITIVO = 1.0
CUSTO_FALSO_NEGATIVO = RAZAO_CUSTO_ANCORA * CUSTO_FALSO_POSITIVO

# --------------------------------------------------------------------------- #
# Protocolo de validacao
#
# 5 dobras repetidas 5 vezes, e nao um unico split treino/teste. Com 499 linhas,
# uma particao unica de 20% deixa cerca de 100 observacoes no teste, e a metrica
# resultante carrega +-5 pontos percentuais so de ruido de particionamento --
# suficiente para inverter a ordem de dois modelos por acaso. As 25 estimativas
# fornecem tanto a media quanto o intervalo de confianca reportado.
# --------------------------------------------------------------------------- #

N_DOBRAS = 5
N_REPETICOES = 5

# Metrica de selecao do modelo final. Deliberadamente independente de limiar:
# como o ponto de corte sera deslocado por custo (ver RAZAO_CUSTO_ANCORA),
# escolher o modelo por Accuracy ou F1 medidos em 0,5 selecionaria pelo
# desempenho num limiar que nao vamos usar. ROC AUC mede a capacidade de
# ordenar risco, que e o que sobrevive a mudanca de limiar.
METRICA_SELECAO = "roc_auc"

# --------------------------------------------------------------------------- #
# Estatistica
# --------------------------------------------------------------------------- #

NIVEL_CONFIANCA = 0.95
Z_NIVEL_CONFIANCA = 1.959963984540054  # quantil normal para 95%
ALFA = 1 - NIVEL_CONFIANCA
