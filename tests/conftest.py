"""Fixtures compartilhadas.

Os testes rodam sobre a planilha real de `data/raw/` -- e a unica entrada do
projeto e nao muda. Onde o teste precisa de um caso que nao existe na base
(valor fora de dominio, coluna faltando), o DataFrame e construido a mao.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stellantis_recall import config, ingest, transform


@pytest.fixture(scope="session")
def bruto() -> pd.DataFrame:
    """Dado recem-ingerido, sem limpeza."""
    return ingest.carregar()


@pytest.fixture(scope="session")
def trusted(bruto: pd.DataFrame) -> pd.DataFrame:
    """Dado limpo e tipado, sem tocar em disco."""
    limpo, _ = transform.transformar(bruto)
    return limpo


@pytest.fixture
def amostra_valida() -> pd.DataFrame:
    """Quatro linhas sinteticas conformes ao contrato bruto."""
    return pd.DataFrame(
        {
            "modelo": ["Toro", "Argo", "Compass", "Pulse"],
            "idade_veiculo": [0, 3, 5, 8],
            "km": [1_000, 55_000, 90_000, 140_000],
            "reclamacoes": [0, 3, 6, 11],
            "recall": [
                config.ROTULO_NEGATIVO,
                config.ROTULO_POSITIVO,
                config.ROTULO_POSITIVO,
                config.ROTULO_POSITIVO,
            ],
        }
    )
