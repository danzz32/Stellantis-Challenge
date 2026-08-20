-- Perfil descritivo por modelo de veiculo.
--
-- Alimenta tanto a Parte 1 (analise exploratoria) quanto o dashboard executivo.
-- Ter uma unica consulta servindo os dois garante que o numero da pagina do
-- relatorio e o numero do painel nao possam divergir.
--
-- Pressupoe a visao canonica de `eda.conectar`: `recall` e booleano.
--
-- Parametros: nenhum.

select
    modelo,
    count(*)                                          as n_veiculos,
    count(*) filter (where recall)                    as n_recalls,
    avg(case when recall then 1.0 else 0.0 end)       as taxa_recall,
    avg(idade_veiculo)                                as idade_media,
    avg(km)                                           as km_medio,
    median(km)                                        as km_mediano,
    avg(reclamacoes)                                  as reclamacoes_media,
    max(reclamacoes)                                  as reclamacoes_max
from veiculos
group by modelo
order by taxa_recall desc, modelo
