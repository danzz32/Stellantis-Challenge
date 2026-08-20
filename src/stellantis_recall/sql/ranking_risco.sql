-- Ranking de risco por modelo, com intervalo de confianca.
--
-- O enunciado pede "ranking de modelos de maior risco". Um ranking de taxas
-- brutas sobre ~55 veiculos por modelo produz ordenacao que e majoritariamente
-- ruido amostral. O intervalo de Wilson entra aqui justamente para tornar essa
-- incerteza visivel no proprio ranking: se os intervalos se sobrepoem, as
-- posicoes nao sao distinguiveis.
--
-- Wilson e preferido ao intervalo normal (Wald) porque nao degenera quando a
-- taxa se aproxima de 0 ou 1 e se comporta melhor em amostras pequenas.
--
--   centro = (p + z^2/2n) / (1 + z^2/n)
--   margem = z * sqrt( p(1-p)/n + z^2/(4n^2) ) / (1 + z^2/n)
--
-- O mesmo calculo existe em `eda.intervalo_wilson`. A duplicacao e intencional
-- -- SQL serve o dashboard, Python serve a analise -- e `tests/test_eda.py`
-- verifica que as duas implementacoes concordam.
--
-- Pressupoe a visao canonica de `eda.conectar`: `recall` e booleano.
--
-- Parametros:
--   $z : quantil normal do nivel de confianca (1.96 para 95%).

with contagem as (
    select
        modelo,
        count(*)                       as n_veiculos,
        count(*) filter (where recall) as n_recalls
    from veiculos
    group by modelo
),

taxa as (
    select
        modelo,
        n_veiculos,
        n_recalls,
        n_recalls * 1.0 / n_veiculos as p,
        $z * $z                      as z2
    from contagem
),

wilson as (
    select
        modelo,
        n_veiculos,
        n_recalls,
        p as taxa_recall,
        (p + z2 / (2 * n_veiculos)) / (1 + z2 / n_veiculos) as centro,
        $z * sqrt(p * (1 - p) / n_veiculos + z2 / (4 * n_veiculos * n_veiculos))
            / (1 + z2 / n_veiculos)                        as margem
    from taxa
)

select
    row_number() over (order by taxa_recall desc, modelo) as posicao,
    modelo,
    n_veiculos,
    n_recalls,
    taxa_recall,
    greatest(centro - margem, 0.0) as ic_inferior,
    least(centro + margem, 1.0)    as ic_superior,
    2 * margem                     as amplitude_ic
from wilson
order by posicao
