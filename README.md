# Predição de Recall a partir de Ocorrências de Garantia

Desafio técnico — vaga de Bolsista Cientista de Dados, Stellantis.

**[▶ Abrir o painel interativo](https://app-stellantis-challenge.streamlit.app/)**

A partir de 500 registros **sintéticos** de garantia — modelo, idade,
quilometragem e número de reclamações — o projeto avalia a qualidade dos dados,
treina um classificador para o desfecho de recall, mede o seu desempenho e
traduz o resultado em material de decisão.

> Todo o conteúdo deste repositório é sintético, fornecido no enunciado. Não há
> informação real de clientes, de fornecedores ou de operação da Stellantis.

---

## Abordagem

Com quatro preditoras e um alvo balanceado, qualquer classificador razoável
chega perto do teto. O trabalho está em outro lugar: **separar sinal de ruído
amostral em 500 observações**, e não deixar que uma métrica pontual vire
recomendação de negócio.

**Contratos de dados explícitos.** Cada camada declara o que aceita, em
`pandera`. A avaliação de qualidade não é inspeção manual no notebook — é um
contrato que passa ou falha e deixa registro auditável.

**Estimativas com incerteza, sempre.** Taxas por grupo vêm com intervalo de
Wilson; o desempenho vem de validação cruzada estratificada repetida, não de um
split único. Em amostra pequena, métrica sem intervalo é opinião com aparência
de número.

**O ponto de corte é decisão de negócio.** O limiar de 0,5 assume custos
simétricos entre um recall não antecipado e uma inspeção desnecessária. A razão
de custo fica declarada em `config.py`, e o limiar é derivado dela.

### Principais achados

| achado | consequência |
|---|---|
| `corr(idade, km) = 0,947` · VIF 10,4 e 9,7 | colinearidade severa: exige regularização, e a importância precisa ser lida por permutação **em bloco** — medida individual inverte a conclusão |
| `modelo` × `recall`: **p = 0,570** | o modelo do veículo não discrimina risco. O ranking pedido é entregue com barras de erro e ressalva explícita |
| taxa de recall: 4,9% (0 anos) → 85,2% (8 anos) | idade, quilometragem e reclamações concentram todo o sinal |
| razões normalizadas por exposição: corr ≈ 0 | normalizar por uso **destrói** o sinal; ele está nos níveis absolutos |
| 12 combinações de modelo × atributos indistinguíveis entre si | a escolha se justifica por parcimônia, não por desempenho |
| inspecionar 30% da frota → 49% dos recalls | o modelo serve para **priorizar**, não para excluir |

---

## Como executar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e
[Quarto](https://quarto.org/docs/download/).

```bash
uv sync
```

```bash
uv run recall-pipeline
```

```bash
cd qmd && uv run quarto render
```

```bash
uv run streamlit run app/streamlit_app.py
```

Reconstrói o ambiente a partir do `uv.lock`, regenera as camadas de dados a
partir de `data/raw/`, renderiza os documentos em `reports/` e sobe o painel em
`localhost:8501`. Os testes rodam com `uv run pytest`.

---

## Stack

| ferramenta | papel |
|---|---|
| **uv** | ambiente e dependências travadas no `uv.lock` |
| **pandera** | contratos de dados por camada |
| **Parquet** | formato das camadas intermediárias, preserva tipos |
| **DuckDB** | agregações em SQL versionado; a mesma consulta serve relatório e painel |
| **Quarto** | relatório HTML, PDF executivo e painel estático de uma só base |
| **scikit-learn · XGBoost · SHAP** | modelagem e interpretação |
| **Streamlit · shadcn-ui · Plotly** | painel interativo |

Sobre o DuckDB: 500 linhas não exigem um motor analítico por desempenho. Ele
está aqui porque mantém as agregações em SQL declarativo, evita reimplementar a
mesma lógica em pandas para cada consumidor, e preserva o desenho válido se a
base real tiver ordens de grandeza a mais.

---

## Arquitetura de dados

```
data/raw/*.xlsx            origem imutável, como recebida
        │
     ingest.py             normaliza cabeçalhos · valida RawSchema
        │                  (em memória — nada é escrito em disco)
   transform.py            deduplica · tipa · converte o alvo
        ▼
data/trusted/*.parquet     contrato validado, 1 linha = 1 veículo
        │
   build_mart.py           features (features.py) + agregações (sql/ via DuckDB)
        ▼
data/mart/*.parquet        pronto para consumo, sem transformação adicional
        │
        ├──▶ modeling/     train · evaluate · explain  →  outputs/
        ├──▶ qmd/          relatório · PDF · painel     →  reports/
        └──▶ app/          painel interativo
```

A ingestão deliberadamente **não** persiste dados. A análise exploratória
precisa enxergar o dado como ele chegou; as regras de limpeza são consequência
dessa análise, não premissa dela.

---

## Estrutura

```
├── data/            raw (imutável) · trusted (validado) · mart (pronto)
├── src/stellantis_recall/
│   ├── config.py    caminhos, semente, domínios, premissas de custo
│   ├── schemas.py   contratos pandera
│   ├── ingest.py    leitura e validação
│   ├── transform.py limpeza  →  trusted
│   ├── features.py  biblioteca pura de atributos derivados
│   ├── build_mart.py  orquestração  →  mart
│   ├── eda.py       consultas e estatísticas da Parte 1
│   ├── viz.py       paleta e temas (matplotlib e Plotly)
│   ├── pipeline.py  encadeia as etapas
│   ├── sql/         consultas DuckDB versionadas
│   └── modeling/    train · evaluate · explain
├── qmd/             fontes Quarto
├── reports/         documentos renderizados — a entrega
├── app/             painel Streamlit
├── outputs/         modelo, métricas e figuras
└── tests/           115 testes
```

Regra que sustenta o desenho: **os `.qmd` e o app não contêm lógica analítica.**
Importam de `src/` e narram o resultado. Cada número tem um único lugar de
manutenção e é testável isoladamente.

---

## Entregáveis

| item do enunciado | onde está |
|---|---|
| Notebook com análise e modelo | [`reports/01-analise.html`](reports/01-analise.html) — Partes 1 a 4 |
| PDF executivo, máx. 3 páginas | [`reports/02-executivo.pdf`](reports/02-executivo.pdf) |
| Dashboard | [painel interativo](https://app-stellantis-challenge.streamlit.app/) · [`reports/dashboard.html`](reports/dashboard.html) (estático) |

O painel estático e o interativo cobrem as mesmas cinco visões e leem os mesmos
arquivos. O estático abre sem servidor; o interativo acrescenta **planejador de
capacidade de inspeção**, **simulador de razão de custo** — que recalcula limiar
e métricas ao vivo sobre as predições fora da amostra — e **score de veículo
individual** com a decomposição exata da predição.

**Repositório:** <https://github.com/danzz32/Stellantis-Challenge>
