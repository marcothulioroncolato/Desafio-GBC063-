
"""Treina o DQN no Bomberman. Uso: python train.py --episodes 3000
Depois gere os gráficos com: python plot_results.py"""
import argparse
import csv
from collections import Counter
import os
import random
import time

import numpy as np
import torch

from bomberman_env import BombermanEnv
from dqn import DQNAgent


def evaluate(agent, env, n=50, aleatorio=False):
    """Joga n episódios sem exploração (ou com agente aleatório).
    Retorna (recompensa média, taxa de vitória, passos médios)."""
    total, wins, steps = 0.0, 0, 0
    for i in range(n):
        obs, info = env.reset(seed=10_000 + i)
        done = False
        while not done:
            if aleatorio:  # sorteia entre as ações permitidas (todas, se a máscara estiver desligada)
                a = int(random.choice(list(np.flatnonzero(info["mask"]))))
            else:
                a = agent.act(obs, info["mask"], greedy=True)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        wins += int(info["win"])
        steps += env.steps
    return total / n, wins / n, steps / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enemies", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eps-decay", type=int, default=10_000,
                   help="passos até o epsilon chegar a 0,05")
    p.add_argument("--step-cost", type=float, default=0.1,
                   help="penalidade por passo (o slide original usa 1.0)")
    p.add_argument("--train-every", type=int, default=4,
                   help="treina a rede 1 vez a cada N passos do jogo (maior = mais rápido)")
    p.add_argument("--max-steps", type=int, default=200, help="limite de passos por episódio")
    p.add_argument("--double", action="store_true", help="usa Double DQN")
    p.add_argument("--saida", default=".", help="pasta onde salvar os resultados (ex.: dqn, double)")
    p.add_argument("--shaping", type=float, default=0.5,
                   help="bônus por unidade de distância ganha em direção ao inimigo (0 desliga)")
    p.add_argument("--mascara", type=int, default=0,
                   help="1 = máscara de ações (bloqueia ações suicidas ou inúteis)")
    p.add_argument("--enemy-bombs", type=int, default=1, help="1 = inimigos plantam bombas, 0 = só andam")
    p.add_argument("--bomb-timer", type=int, default=5,
                   help="passos até a bomba explodir (mais alto = mais tempo para fugir)")
    p.add_argument("--checkpoint-every", type=int, default=250,
                   help="salva o modelo a cada N episódios em checkpoints/")
    args, _ = p.parse_known_args()
    os.makedirs(args.saida, exist_ok=True)
    out = lambda nome: os.path.join(args.saida, nome)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BombermanEnv(n_enemies=args.enemies, step_cost=args.step_cost,
                       bomb_timer=args.bomb_timer, shaping=args.shaping,
                       max_steps=args.max_steps, enemy_bombs=bool(args.enemy_bombs),
                       mascara=bool(args.mascara))
    agent = DQNAgent(in_channels=6, size=env.size, n_actions=env.action_space.n,
                     eps_decay_steps=args.eps_decay, double=args.double,
                     train_every=args.train_every)

    os.makedirs(out("checkpoints"), exist_ok=True)
    agent.save(out("checkpoints/dqn_ep0000.pt"))  # agente ainda sem treino

    print(f"Algoritmo: {'Double DQN' if args.double else 'DQN clássico'} | episódios={args.episodes} "
          f"| inimigos={args.enemies} (bombas={bool(args.enemy_bombs)}) | shaping={args.shaping} "
          f"| timer={args.bomb_timer} | máscara={bool(args.mascara)} | saída='{args.saida}'")
    t0 = time.time()
    rewards, rows, evals, causas = [], [], [], []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        total, done, losses = 0.0, False, []
        while not done:
            a = agent.act(obs, info["mask"])
            obs2, r, term, trunc, info = env.step(a)
            # só 'terminated' corta o alvo; truncamento por passos ainda faz bootstrap
            loss = agent.update(obs, a, r, obs2, term, info["mask"])
            if loss is not None:
                losses.append(loss)
            obs, total, done = obs2, total + r, term or trunc
        rewards.append(total)
        causas.append(info["causa"] or ("vitória" if info["win"] else "limite"))
        perda = float(np.mean(losses)) if losses else float("nan")
        rows.append((ep, total, int(info["win"]), round(agent.epsilon(), 3), env.steps, perda))

        if ep % args.checkpoint_every == 0:
            agent.save(out(f"checkpoints/dqn_ep{ep:04d}.pt"))

        if ep % 50 == 0:
            fim = " ".join(f"{k}={v}" for k, v in Counter(causas[-50:]).most_common())
            print(f"ep {ep:5d} | média(50)={np.mean(rewards[-50:]):8.1f} | eps={agent.epsilon():.2f} "
                  f"| {(time.time() - t0) / 60:.1f} min | fim dos últimos 50: {fim}")
        if ep % args.eval_every == 0:
            avg, win, _ = evaluate(agent, env)
            evals.append((ep, avg, win))
            print(f"  >> avaliação: recompensa={avg:.1f}  vitórias={win:.0%}")

    agent.save(out("dqn_bomberman.pt"))

    with open(out("rewards.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "recompensa", "vitoria", "epsilon", "passos", "perda_media"])
        w.writerows(rows)
    with open(out("avaliacoes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "recompensa", "vitoria"])
        w.writerows(evals)

    # comparação final: DQN treinado x agente aleatório (mesmos 100 mapas)
    dqn = evaluate(agent, env, n=100)
    rnd = evaluate(agent, env, n=100, aleatorio=True)
    with open(out("resultados_finais.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agente", "recompensa", "vitoria", "passos"])
        w.writerow(["DQN", *dqn])
        w.writerow(["Aleatório", *rnd])
    print(f"Final | DQN: recompensa={dqn[0]:.1f}, vitórias={dqn[1]:.0%} | "
          f"Aleatório: recompensa={rnd[0]:.1f}, vitórias={rnd[1]:.0%}")
    print(f"Agora rode: python plot_results.py {args.saida}")


if __name__ == "__main__":
    main()
