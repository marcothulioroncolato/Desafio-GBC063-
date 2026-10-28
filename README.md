# Desafio-GBC063-
Modelagem de MDP e Aprendizagem por Reforço aplicada ao jogo Bomberman — Desafio 1 (GBC063)
Este repositório contém a apresentação e a documentação do Desafio 1 da disciplina, focado no desenvolvimento e modelagem de um ambiente baseado em Processo de Decisão de Markov (MDP) para o jogo Bomberman.
O ficheiro com a apresentação de slides está disponível no repositório:
Desafio 1 - Bomberman(MDP)
Resumo do Problema
- Ambiente: Grelha 11 \times 11 com blocos destrutíveis, paredes fixas e 2 inimigos.
- Espaço de Ações: 6 ações discretas (Mover em 4 direções, Colocar bomba, Ficar parado).
- Propriedade Markoviana: Garantida ao incluir o *timer* de contagem decrescente das bombas no estado.
- Recompensas:
  - Destruir bloco: +10
  - Eliminar inimigo: +50
  - Eliminar todos os inimigos (Vitória): +100
  - Morte (Bomba/Inimigo): -100
  - Custo por passo: -1
