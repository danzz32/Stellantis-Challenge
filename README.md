# Predição de Recall a partir de Ocorrências de Garantia

Desafio técnico — vaga de Bolsista Cientista de Dados, Stellantis.

A partir de 500 registros **sintéticos** de garantia (modelo, idade, quilometragem
e número de reclamações), o projeto avalia a qualidade dos dados, treina um
classificador para o desfecho de recall, mede o seu desempenho e traduz o
resultado em material de decisão executiva.

> **Sobre os dados:** todo o conteúdo deste repositório é sintético, fornecido no
> enunciado do desafio. Não há informação real de clientes, de fornecedores ou de
> operação da Stellantis.

---

## Como executar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e
[Quarto](https://quarto.org/docs/download/) instalados.

```bash
uv sync
```

```bash
uv run recall-ingest
```

```bash
cd qmd && uv run quarto render
```

O primeiro comando reconstrói o ambiente a partir do `uv.lock` — mesma versão de
cada pacote, em qualquer máquina. O segundo valida o contrato de dados e reporta
o resultado. O terceiro renderiza os documentos para `reports/`.

---

## Abordagem

O eixo da solução não é a escolha do algoritmo — com quatro preditoras e um alvo
balanceado, qualquer classificador razoável chega perto do teto. O eixo é
**separar sinal de ruído amostral em 500 observações** e não deixar que uma
métrica pontual vire recomendação de negócio.

Três decisões estruturam o trabalho:

**Contratos de dados explícitos.** Cada camada declara o que aceita, em `pandera`.
A avaliação de qualidade não é uma inspeção manual no notebook — é um contrato
que passa ou falha, e cujo resultado entra no relatório como evidência.

**Estimativas com incerteza, sempre.** Taxas por grupo vêm com intervalo de
Wilson; o desempenho do modelo vem de validação cruzada estratificada repetida,
não de um split único. Em amostra pequena, uma métrica sem intervalo é uma
opinião com aparência de número.

**O ponto de corte é decisão de negócio, não default.** O limiar de 0,5 assume
custos simétricos entre um recall não antecipado e uma inspeção desnecessária.
A razão de custo fica declarada e editável em `config.py`, e o limiar é derivado
dela.

### Achados que orientaram o resto do projeto

| achado | consequência |
|---|---|
| `corr(idade, km) = 0,947`, VIF 10,4 e 9,7 | colinearidade severa: Regressão Logística exige regularização, e a importância de variáveis precisa ser lida por permutação, não pelo `feature_importances_` |
| `modelo` × `recall`: χ²(8)=6,69, **p = 0,570** | o modelo do veículo **não** discrimina risco. O ranking pedido no dashboard é entregue com barras de erro e ressalva explícita |
| taxa de recall: 4,9% (0 anos) → 85,2% (8 anos) | idade e reclamações concentram todo o sinal utilizável |
| classes em 47,8% / 52,2% | *Accuracy* é métrica honesta aqui; sem necessidade de reamostragem |
| curtose negativa em `idade` e `km` | distribuições quase uniformes — assinatura de geração sintética, registrada nas limitações |

---

## Stack

| ferramenta | papel | por quê |
|---|---|---|
| **uv** | ambiente e dependências | `uv.lock` torna a entrega reproduzível byte a byte |
| **pandera** | contratos de dados | transforma "avaliação de qualidade" em teste executável |
| **Parquet** | formato das camadas intermediárias | preserva tipos (categórica, booleano) entre etapas |
| **DuckDB** | agregações analíticas | SQL versionado em `sql/`, e a *mesma* consulta serve relatório e dashboard |
| **Quarto** | documentos | uma base de código, três saídas: HTML, PDF executivo e dashboard |
| **scikit-learn / XGBoost / SHAP** | modelagem e interpretação | — |
| **Streamlit** | simulador interativo de risco | complemento opcional ao dashboard estático |

Uma nota de honestidade sobre o DuckDB: 500 linhas não exigem um motor analítico
por desempenho. Ele está aqui porque mantém as agregações em SQL declarativo,
evita reimplementar a mesma lógica em pandas para cada consumidor, e preserva o
desenho válido caso a base real de ocorrências tenha ordens de grandeza a mais.

---

## Arquitetura de dados

```
data/raw/*.xlsx            origem imutável, como recebida
        │
     ingest.py             normaliza cabeçalhos · valida RawSchema
        │                  (em memória — nada é escrito em disco)
        │
   transform.py            deduplica · tipa · converte o alvo
        ▼
data/trusted/*.parquet     contrato validado, 1 linha = 1 veículo
        │
   build_mart.py           features (features.py) + agregações (sql/ via DuckDB)
        ▼
data/mart/*.parquet        pronto para consumo, sem transformação adicional
        │
        ├──▶ modeling/     train · evaluate · explain  →  outputs/
        └──▶ qmd/          relatório · PDF executivo · dashboard  →  reports/
```

A ingestão deliberadamente **não** persiste dados. A análise exploratória precisa
enxergar o dado como ele chegou; as regras de limpeza são consequência dessa
análise, não premissa dela. Por isso `data/trusted/` só passa a existir depois
que a Parte 1 produziu o backlog de tratamento.

---

## Estrutura

```
├── data/
│   ├── raw/          planilha de origem (imutável)
│   ├── trusted/      dado validado e limpo
│   └── mart/         tabelas prontas para consumo
├── src/stellantis_recall/
│   ├── config.py     caminhos, semente, domínios, premissas de custo
│   ├── schemas.py    contratos pandera
│   ├── ingest.py     leitura e validação
│   ├── transform.py  limpeza  →  trusted
│   ├── features.py   biblioteca pura de atributos derivados
│   ├── build_mart.py orquestração  →  mart
│   ├── eda.py        consultas e estatísticas da Parte 1
│   ├── viz.py        tema e paleta compartilhados
│   ├── pipeline.py   encadeia as etapas
│   ├── sql/          consultas DuckDB versionadas
│   └── modeling/     train · evaluate · explain
├── qmd/              fontes Quarto (_quarto.yml aponta para ../reports)
├── reports/          documentos renderizados — a entrega
├── docs/             decisões técnicas e dicionário de dados
├── app/              Streamlit
├── outputs/          modelo, métricas e figuras
└── tests/            pytest
```

Regra que sustenta o desenho: **os `.qmd` não contêm lógica analítica.** Eles
importam de `src/` e narram o resultado. Assim cada número do relatório tem um
único lugar de manutenção e é testável isoladamente.

---

## Entregáveis

| item do enunciado | arquivo |
|---|---|
| Notebook com análise e modelo | `reports/01-analise.html` (fonte: `qmd/01-analise.qmd`) |
| PDF executivo, máx. 3 páginas | `reports/02-executivo.pdf` |
| Dashboard | `reports/dashboard.html` · `app/streamlit_app.py` |

---

## Estado atual

- [x] **Parte 1 — Análise exploratória:** contrato de ingestão, avaliação de
      qualidade, descritivas, correlações, VIF, teste de associação e taxas com
      intervalo de confiança.
- [ ] **Parte 2 — Modelo preditivo:** baseline, Regressão Logística regularizada,
      Random Forest e XGBoost, com justificativa da escolha.
- [ ] **Parte 3 — Avaliação:** Accuracy, Precision, Recall e F1 por validação
      cruzada estratificada repetida, com intervalo de confiança.
- [ ] **Parte 4 — Interpretação:** importância por permutação, SHAP, limitações.
- [ ] **Parte 5 — Dashboard executivo** e PDF de três páginas.
