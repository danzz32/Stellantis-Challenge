-- Perfil de qualidade coluna a coluna.
--
-- Uma linha por coluna da tabela `veiculos`, com preenchimento, cardinalidade e
-- amplitude. O UNION ALL e proposital: mantem o resultado em formato longo, que
-- e o que a tabela do relatorio consome sem transformacao adicional.
--
-- Parametros: nenhum.

with base as (
    select * from veiculos
)

select
    'modelo'                                   as coluna,
    'categorica'                               as natureza,
    count(*)                                   as n_linhas,
    count(modelo)                              as n_preenchidos,
    count(*) - count(modelo)                   as n_nulos,
    count(distinct modelo)                     as n_distintos,
    cast(null as double)                       as minimo,
    cast(null as double)                       as maximo
from base

union all
select
    'idade_veiculo',
    'numerica',
    count(*),
    count(idade_veiculo),
    count(*) - count(idade_veiculo),
    count(distinct idade_veiculo),
    cast(min(idade_veiculo) as double),
    cast(max(idade_veiculo) as double)
from base

union all
select
    'km',
    'numerica',
    count(*),
    count(km),
    count(*) - count(km),
    count(distinct km),
    cast(min(km) as double),
    cast(max(km) as double)
from base

union all
select
    'reclamacoes',
    'numerica',
    count(*),
    count(reclamacoes),
    count(*) - count(reclamacoes),
    count(distinct reclamacoes),
    cast(min(reclamacoes) as double),
    cast(max(reclamacoes) as double)
from base

union all
select
    'recall',
    'alvo',
    count(*),
    count(recall),
    count(*) - count(recall),
    count(distinct recall),
    cast(null as double),
    cast(null as double)
from base
