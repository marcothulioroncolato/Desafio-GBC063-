
"""Grava partidas do agente em GIF e MP4, se possível.

Exemplo:
python gravar_video.py --modelo double_mascara/dqn_bomberman.pt --episodios 3

Para gravar um agente aleatório:
python gravar_video.py --aleatorio --episodios 3
"""

import argparse
import os
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bomberman_env import BombermanEnv


CELL = 48
TOPO = 32


def _fonte(tamanho=16):
    """Carrega uma fonte que suporta acentos."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", tamanho)
    except OSError:
        try:
            import matplotlib

            caminho = os.path.join(
                matplotlib.get_data_path(),
                "fonts/ttf/DejaVuSans.ttf"
            )
            return ImageFont.truetype(caminho, tamanho)
        except Exception:
            return ImageFont.load_default()


FONTE = _fonte(16)


def desenhar(env, titulo):
    """Desenha o estado atual do Bomberman."""

    n = env.size

    img = Image.new(
        "RGB",
        (n * CELL, n * CELL + TOPO),
        (25, 28, 45)
    )

    d = ImageDraw.Draw(img)

    d.text(
        (10, 6),
        titulo,
        fill=(244, 239, 230),
        font=FONTE
    )

    # Desenha o tabuleiro
    for i in range(n):
        for j in range(n):

            x = j * CELL
            y = i * CELL + TOPO

            if env.walls[i, j]:
                cor = (70, 72, 90)

            elif env.blocks[i, j]:
                cor = (176, 124, 72)

            elif env.flames[i, j]:
                cor = (255, 140, 0)

            else:
                cor = (236, 234, 224)

            d.rectangle(
                (x, y, x + CELL - 2, y + CELL - 2),
                fill=cor
            )

    # Desenha as bombas
    for (r, c, t, dono) in env.bombs:

        cx = c * CELL + CELL // 2
        cy = r * CELL + CELL // 2 + TOPO

        d.ellipse(
            (cx - 16, cy - 16, cx + 16, cy + 16),
            fill=(20, 20, 20),
            outline=(205, 45, 45) if dono else None,
            width=4
        )

        d.text(
            (cx, cy),
            str(t),
            fill=(255, 255, 255),
            font=FONTE,
            anchor="mm"
        )

    # Desenha os inimigos
    for (r, c) in env.enemies:

        cx = c * CELL + CELL // 2
        cy = r * CELL + CELL // 2 + TOPO

        d.ellipse(
            (cx - 17, cy - 17, cx + 17, cy + 17),
            fill=(205, 45, 45)
        )

    # Desenha o agente
    r, c = env.agent

    cx = c * CELL + CELL // 2
    cy = r * CELL + CELL // 2 + TOPO

    d.ellipse(
        (cx - 17, cy - 17, cx + 17, cy + 17),
        fill=(45, 115, 225)
    )

    return img


NOMES = [
    "cima",
    "baixo",
    "esquerda",
    "direita",
    "bomba",
    "parado"
]


def jogar(escolher, env, seed, rotulo):
    """Executa uma partida e armazena os frames."""

    obs, info = env.reset(seed=seed)

    total = 0.0
    done = False

    acoes = Counter()

    frames = [
        desenhar(
            env,
            f"{rotulo} | passo 0 | recompensa 0.0"
        )
    ]

    while not done:

        # Escolhe a ação
        a = escolher(obs, info["mask"])

        acoes[NOMES[a]] += 1

        # Executa a ação
        obs, r, term, trunc, info = env.step(a)

        total += r

        done = term or trunc

        # Salva o estado visual
        frames.append(
            desenhar(
                env,
                f"{rotulo} | passo {env.steps} | "
                f"recompensa {total:.1f}"
            )
        )

    # Pausa no final da partida
    frames += [frames[-1]] * 6

    fim = info["causa"] or (
        "vitória" if info["win"] else "limite de passos"
    )

    return frames, total, fim, acoes


def salvar(frames, nome, fps):
    """Salva os frames em GIF e tenta gerar MP4."""

    # Salvar GIF
    frames[0].save(
        f"{nome}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

    print(f"GIF salvo: {nome}.gif")

    # Salvar MP4
    try:
        import imageio.v2 as imageio

        imageio.mimsave(
            f"{nome}.mp4",
            [np.array(f) for f in frames],
            fps=fps
        )

        print(f"MP4 salvo: {nome}.mp4")

    except Exception as e:
        print("Não foi possível gerar MP4.")
        print("Erro:", e)
        print("O GIF foi salvo normalmente.")


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--modelo",
        nargs="+",
        default=["double_mascara/dqn_bomberman.pt"],
        help="Caminho do modelo treinado"
    )

    p.add_argument("--episodios", type=int, default=3)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--enemies", type=int, default=2)
    p.add_argument("--step-cost", type=float, default=0.1)
    p.add_argument("--shaping", type=float, default=0.5)
    p.add_argument("--mascara", type=int, default=0)
    p.add_argument("--enemy-bombs", type=int, default=1)
    p.add_argument("--bomb-timer", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--fps", type=int, default=4)

    p.add_argument(
        "--aleatorio",
        action="store_true",
        help="Grava partidas de um agente aleatório"
    )

    args = p.parse_args()

    # Cria o ambiente com as configurações do treinamento
    env = BombermanEnv(
        n_enemies=args.enemies,
        step_cost=args.step_cost,
        bomb_timer=args.bomb_timer,
        shaping=args.shaping,
        max_steps=args.max_steps,
        enemy_bombs=bool(args.enemy_bombs),
        mascara=bool(args.mascara)
    )

    jogadores = []

    if args.aleatorio:

        jogadores.append((
            "Aleatório",
            lambda obs, m: int(
                random.choice(list(np.flatnonzero(m)))
            )
        ))

        nome = "partida_aleatorio"

    else:

        from dqn import DQNAgent

        for caminho in args.modelo:

            # Verifica se o arquivo existe
            if not os.path.isfile(caminho):

                print("ERRO: modelo não encontrado!")
                print("Caminho informado:", caminho)
                print("Diretório atual:", os.getcwd())

                pasta = os.path.dirname(caminho) or "."

                if os.path.isdir(pasta):
                    print(
                        "Arquivos da pasta:",
                        os.listdir(pasta)
                    )

                return

            # Cria o agente
            agent = DQNAgent(
                in_channels=6,
                size=env.size,
                n_actions=env.action_space.n
            )

            # Carrega os pesos treinados
            agent.load(caminho)

            rotulo = (
                "Double DQN"
                if len(args.modelo) == 1
                else os.path.splitext(
                    os.path.basename(caminho)
                )[0]
            )

            # Escolhe ações usando a rede treinada
            jogadores.append((
                rotulo,
                (
                    lambda ag: (
                        lambda obs, m: ag.act(
                            obs,
                            m,
                            greedy=True
                        )
                    )
                )(agent)
            ))

        nome = (
            "partida_dqn"
            if len(jogadores) == 1
            else "evolucao"
        )

    todos = []

    # Executa as partidas
    for rotulo, escolher in jogadores:

        for k in range(args.episodios):

            frames, total, fim, acoes = jogar(
                escolher,
                env,
                args.seed + k,
                rotulo
            )

            print(
                f"{rotulo} | "
                f"partida {k + 1} | "
                f"passos={len(frames) - 7} | "
                f"recompensa={total:.1f} | "
                f"fim={fim} | "
                f"ações={dict(acoes)}"
            )

            todos += frames

    salvar(todos, nome, args.fps)


if __name__ == "__main__":
    main()
