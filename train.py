
import argparse
import csv
import random

import numpy as np
import torch

from bomberman_env import BombermanEnv
from dqn import DQNAgent


def evaluate(agent, env, n=50, aleatorio=False):
    """Joga n episódios sem exploração (ou com agente aleatório).
    Retorna (recompensa média, taxa de vitória, passos médios)."""
    total, wins, steps = 0.0, 0, 0
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i)
        done = False
        while not done:
            a = env.action_space.sample() if aleatorio else agent.act(obs, greedy=True)
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
    args, _ = p.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = BombermanEnv(n_enemies=args.enemies, step_cost=args.step_cost)
    agent = DQNAgent(in_channels=6, size=env.size, n_actions=env.action_space.n,
                     eps_decay_steps=args.eps_decay)

    rewards, rows, evals = [], [], []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        total, done, losses = 0.0, False, []
        while not done:
            a = agent.act(obs)
            obs2, r, term, trunc, info = env.step(a)
            # só 'terminated' corta o alvo; truncamento por passos ainda faz bootstrap
            loss = agent.update(obs, a, r, obs2, term)
            if loss is not None:
                losses.append(loss)
            obs, total, done = obs2, total + r, term or trunc
        rewards.append(total)
        perda = float(np.mean(losses)) if losses else float("nan")
        rows.append((ep, total, int(info["win"]), round(agent.epsilon(), 3), env.steps, perda))

        if ep % 50 == 0:
            print(f"ep {ep:5d} | média(50)={np.mean(rewards[-50:]):8.1f} | eps={agent.epsilon():.2f}")
        if ep % args.eval_every == 0:
            avg, win, _ = evaluate(agent, env)
            evals.append((ep, avg, win))
            print(f"  >> avaliação: recompensa={avg:.1f}  vitórias={win:.0%}")

    agent.save("dqn_bomberman.pt")

    with open("rewards.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "recompensa", "vitoria", "epsilon", "passos", "perda_media"])
        w.writerows(rows)
    with open("avaliacoes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episodio", "recompensa", "vitoria"])
        w.writerows(evals)

    # comparação final: DQN treinado x agente aleatório (mesmos 100 mapas)
    dqn = evaluate(agent, env, n=100)
    rnd = evaluate(agent, env, n=100, aleatorio=True)
    with open("resultados_finais.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agente", "recompensa", "vitoria", "passos"])
        w.writerow(["DQN", *dqn])
        w.writerow(["Aleatório", *rnd])
    print(f"Final | DQN: recompensa={dqn[0]:.1f}, vitórias={dqn[1]:.0%} | "
          f"Aleatório: recompensa={rnd[0]:.1f}, vitórias={rnd[1]:.0%}")
    print("Agora rode: python plot_results.py")


if __name__ == "__main__":
    main()
