-- Matriz de risco por segmento de frota: faixa de idade x faixa de quilometragem.
--
-- Diferente das demais consultas, esta roda sobre a tabela de atributos do mart
-- (`features.parquet`), porque depende das faixas construidas em `features.py`.
--
-- E a visao que traduz o modelo em recorte operacional: em vez de "risco do
-- veiculo X", responde "qual segmento da frota concentra risco". Segmento e
-- acionavel por uma area de pos-vendas; veiculo individual, nem sempre.
--
-- A contagem por celula acompanha a taxa de proposito. Com 499 veiculos
-- distribuidos em ate 16 celulas, algumas ficam com pouquissimas observacoes, e
-- a taxa dessas celulas nao deve ser lida como estimativa. O dashboard usa
-- `n_veiculos` para suprimir a leitura onde a amostra nao sustenta.
--
-- Parametros: nenhum.

select
    faixa_idade,
    faixa_km,
    count(*)                                    as n_veiculos,
    count(*) filter (where recall)               as n_recalls,
    avg(case when recall then 1.0 else 0.0 end) as taxa_recall,
    avg(reclamacoes)                            as reclamacoes_media
from veiculos
group by faixa_idade, faixa_km
order by faixa_idade, faixa_km
