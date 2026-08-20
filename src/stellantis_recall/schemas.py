"""Contratos de dados (pandera).

Cada camada da arquitetura tem o seu contrato. Este modulo define o contrato da
camada de ingestao; os contratos de `trusted` e `mart` sao adicionados junto dos
seus respectivos modulos produtores.

O ponto de projeto aqui: o esquema bruto valida *estrutura e dominio*, mas nao
exige que o dado ja esteja limpo. Duplicatas, por exemplo, sao permitidas neste
nivel de proposito -- elas precisam chegar ate a analise exploratoria para serem
diagnosticadas, e so entao a regra de remocao e escrita em `transform.py`.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing.pandas import Series

from . import config

# --------------------------------------------------------------------------- #
# Camada de ingestao
# --------------------------------------------------------------------------- #


class RawSchema(pa.DataFrameModel):
    """Contrato do dado recem-lido do Excel, ja com colunas normalizadas.

    Falhar aqui significa que o arquivo de origem mudou de forma incompativel
    com o que a analise pressupoe.
    """

    modelo: Series[str] = pa.Field(
        nullable=False,
        isin=list(config.MODELOS),
        description="Modelo do veiculo.",
    )
    idade_veiculo: Series[int] = pa.Field(
        nullable=False,
        ge=config.IDADE_MIN,
        le=config.IDADE_MAX,
        description="Idade do veiculo em anos completos.",
    )
    km: Series[int] = pa.Field(
        nullable=False,
        ge=config.KM_MIN,
        le=config.KM_MAX,
        description="Quilometragem acumulada.",
    )
    reclamacoes: Series[int] = pa.Field(
        nullable=False,
        ge=config.RECLAMACOES_MIN,
        le=config.RECLAMACOES_MAX,
        description="Numero de reclamacoes de garantia registradas.",
    )
    recall: Series[str] = pa.Field(
        nullable=False,
        isin=list(config.ROTULOS_ALVO),
        description="Alvo: houve recall associado ao veiculo.",
    )

    class Config:  # noqa: D106 - configuracao declarativa do pandera
        name = "RawSchema"
        strict = True  # nenhuma coluna inesperada
        ordered = False
        coerce = True


#: Ordem canonica das colunas apos a ingestao.
COLUNAS_RAW: tuple[str, ...] = (
    "modelo",
    "idade_veiculo",
    "km",
    "reclamacoes",
    "recall",
)


# --------------------------------------------------------------------------- #
# Camada trusted
# --------------------------------------------------------------------------- #


class TrustedSchema(pa.DataFrameModel):
    """Contrato do dado limpo e tipado, uma linha por veiculo.

    Diferencas em relacao ao `RawSchema`, todas derivadas da analise da Parte 1:
    o alvo e booleano em vez de texto, e nao ha linhas duplicadas. Este contrato
    e validado de forma estrita -- a camada nao deve produzir arquivo fora dele.
    """

    modelo: Series[str] = pa.Field(nullable=False, isin=list(config.MODELOS))
    idade_veiculo: Series[int] = pa.Field(
        nullable=False, ge=config.IDADE_MIN, le=config.IDADE_MAX
    )
    km: Series[int] = pa.Field(nullable=False, ge=config.KM_MIN, le=config.KM_MAX)
    reclamacoes: Series[int] = pa.Field(
        nullable=False, ge=config.RECLAMACOES_MIN, le=config.RECLAMACOES_MAX
    )
    recall: Series[bool] = pa.Field(
        nullable=False,
        description="Alvo binarizado: True equivale a 'Sim' na origem.",
    )

    class Config:  # noqa: D106 - configuracao declarativa do pandera
        name = "TrustedSchema"
        strict = True
        unique_column_names = True
        # Sem coercao, ao contrario do RawSchema. Aqui o contrato *verifica* que
        # `transform.tipar` fez o seu trabalho, em vez de consertar o tipo por
        # conta propria -- e a diferenca importa no alvo: com coercao, uma
        # coluna que chegasse como texto teria 'Sim' e 'Nao' convertidos ambos
        # para True (qualquer string nao vazia e verdadeira), corrompendo o alvo
        # sem emitir um unico erro.
        coerce = False


#: Ordem canonica das colunas da camada trusted.
COLUNAS_TRUSTED: tuple[str, ...] = COLUNAS_RAW


# --------------------------------------------------------------------------- #
# Camada mart
# --------------------------------------------------------------------------- #


class FeaturesSchema(TrustedSchema):
    """Contrato da tabela de atributos consumida pela modelagem.

    Herda o contrato `trusted` e acrescenta as derivadas. As razoes sao
    validadas como finitas: `km_por_ano` e `reclamacoes_por_ano` dividem por
    `idade + 0,5` e `reclamacoes_por_10k_km` divide por `km`, cujo minimo de
    dominio e maior que zero -- se algum infinito aparecer, o pressuposto
    mudou e a validacao precisa interromper o pipeline.
    """

    km_por_ano: Series[float] = pa.Field(nullable=False, gt=0, le=config.KM_MAX)
    reclamacoes_por_ano: Series[float] = pa.Field(nullable=False, ge=0)
    reclamacoes_por_10k_km: Series[float] = pa.Field(nullable=False, ge=0)
    muitas_reclamacoes: Series[bool] = pa.Field(nullable=False)
    faixa_idade: Series[str] = pa.Field(nullable=False)
    faixa_km: Series[str] = pa.Field(nullable=False)

    class Config:  # noqa: D106 - configuracao declarativa do pandera
        name = "FeaturesSchema"
        strict = True
        coerce = False  # mesma razao do TrustedSchema


class AgregadoRiscoSchema(pa.DataFrameModel):
    """Contrato comum das tabelas agregadas com taxa e intervalo de confianca.

    Vale para `ranking_risco`; o dashboard consome estas colunas diretamente e
    quebraria em silencio se uma taxa saisse de [0, 1] ou um intervalo viesse
    invertido.
    """

    n_veiculos: Series[int] = pa.Field(gt=0)
    n_recalls: Series[int] = pa.Field(ge=0)
    taxa_recall: Series[float] = pa.Field(ge=0.0, le=1.0)
    ic_inferior: Series[float] = pa.Field(ge=0.0, le=1.0)
    ic_superior: Series[float] = pa.Field(ge=0.0, le=1.0)

    @pa.dataframe_check
    def intervalo_contem_a_estimativa(cls, df: pd.DataFrame) -> pd.Series:
        """O IC de Wilson nao e simetrico, mas precisa conter a taxa observada."""
        return (df["ic_inferior"] <= df["taxa_recall"]) & (
            df["taxa_recall"] <= df["ic_superior"]
        )

    @pa.dataframe_check
    def contagem_coerente(cls, df: pd.DataFrame) -> pd.Series:
        """Nao pode haver mais recalls do que veiculos no grupo."""
        return df["n_recalls"] <= df["n_veiculos"]

    class Config:  # noqa: D106 - configuracao declarativa do pandera
        name = "AgregadoRiscoSchema"
        strict = False  # cada agregado traz colunas proprias alem destas
        coerce = True
