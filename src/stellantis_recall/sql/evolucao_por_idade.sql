-- Evolucao dos casos ao longo da idade da frota.
--
-- O dataset nao tem coluna de data: `idade_veiculo` e a unica dimensao temporal
-- disponivel, e e ela que sustenta o item "evolucao dos casos" do dashboard.
-- Isso e uma limitacao a declarar no relatorio -- e uma leitura de coorte
-- (frota por idade), nao uma serie temporal de ocorrencias.
--
-- Pressupoe a visao canonica de `eda.conectar`: `recall` e booleano.
--
-- Parametros: nenhum.

with por_idade as (
    select
        idade_veiculo,
        count(*)                                    as n_veiculos,
        count(*) filter (where recall)              as n_recalls,
        avg(case when recall then 1.0 else 0.0 end) as taxa_recall,
        avg(km)                                     as km_medio,
        avg(reclamacoes)                            as reclamacoes_media
    from veiculos
    group by idade_veiculo
)

select
    idade_veiculo,
    n_veiculos,
    n_recalls,
    taxa_recall,
    km_medio,
    reclamacoes_media,
    sum(n_veiculos) over janela as n_veiculos_acumulado,
    sum(n_recalls)  over janela as n_recalls_acumulado,
    sum(n_recalls)  over janela * 1.0
        / nullif(sum(n_veiculos) over janela, 0) as taxa_recall_acumulada
from por_idade
window janela as (order by idade_veiculo rows between unbounded preceding and current row)
order by idade_veiculo
