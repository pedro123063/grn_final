# BOLSA-GRN

tirar o tol do diff_ev
pasta para coonnfiguracao de varias execuções
config por file
diferença mennnor que atol ou menor igual que atol pode ser motivo para parada antes do limite de gens (seed 0)

analise com contador de hardware pode ser meio para explicar compoprtamento e perdas de eficiencia com nnumeros maiores de cores

speedup e eficiencia ( atual e cortado) : calcular speedup e eficiencia tanto nesse quanto no trunncado por tol/atol e comparar.

pegar calculo da funcao objetivo e ver se vale à penas fazer calculo em gpu
    algo na biblioteca que permita porte facil p gpu
    na parte calculo e func objetivo (integracao numerica)  - pode ser pior pois custo de chamadas vai ter ganho pequeno em relacao a rebound do os e custo de chamada
        +passar todas as integracoes de todos os ind de geracao de uma vez só

    procurar bibliotecas que integram numericamente em gpu com as mesmas funcionalidades que atual e comparar tempos e resultados ( alem de speedup)
        +avaliar resultados e performance com uma chamada p depois ir juntando ( garantir consistencia antes de focar em performance)
        +migracao aos poucos. p perceber impacto
        +atencao : combinacao de operacoes aritmeticas em gpu diferente de cpu e f32 com double 