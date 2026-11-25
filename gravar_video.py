import imageio
from base64 import b64encode
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import torch

def renderizar_frame(env, passo, recompensa_acumulada, escala=40):
    n = env.tamanho
    COR_FUNDO = [235, 240, 235]
    COR_PAREDE = [55, 65, 81]
    COR_BLOCO = [180, 115, 60]
    COR_BOMBA = [20, 20, 20]
    COR_CHAMA = [255, 110, 20]
    COR_AGENTE = [37, 99, 235]
    COR_INIMIGO = [220, 38, 38]

    grid = np.zeros((n, n, 3), dtype=np.uint8)
    grid[:, :] = COR_FUNDO
    grid[env.mapa == 1] = COR_PAREDE
    grid[env.mapa == 2] = COR_BLOCO

    for b in env.bombas:
        r, c = b["pos"]
        grid[r, c] = COR_BOMBA

    for r, c in env.chamas:
        grid[r, c] = COR_CHAMA

    for ini in env.inimigos:
        if ini["vivo"]:
            ir, ic = ini["pos"]
            grid[ir, ic] = COR_INIMIGO

    ar, ac = env.agente_pos
    grid[ar, ac] = COR_AGENTE

    img = np.repeat(np.repeat(grid, escala, axis=0), escala, axis=1)

    for i in range(0, n * escala, escala):
        img[i, :, :] = [200, 200, 200]
        img[:, i, :] = [200, 200, 200]

    fig, ax = plt.subplots(figsize=(5.5, 6.0), dpi=100)
    ax.imshow(img)
    ax.axis("off")

    for b in env.bombas:
        r, c = b["pos"]
        ax.text(c * escala + escala / 2, r * escala + escala / 2, str(b["timer"]),
                color="white", fontsize=11, fontweight="bold", ha="center", va="center")

    ax.text(ac * escala + escala / 2, ar * escala + escala / 2, "A",
            color="white", fontsize=11, fontweight="bold", ha="center", va="center")

    for ini in env.inimigos:
        if ini["vivo"]:
            ir, ic = ini["pos"]
            ax.text(ic * escala + escala / 2, ir * escala + escala / 2, "E",
                    color="white", fontsize=11, fontweight="bold", ha="center", va="center")

    inimigos_vivos = sum(1 for ini in env.inimigos if ini["vivo"])
    ax.set_title(
        f"Passo: {passo} | Recompensa: {recompensa_acumulada:.0f} | Inimigos: {inimigos_vivos}/2\n"
        f"[Azul: Agente | Vermelho: Inimigos | Preto: Bomba | Marrom: Bloco]",
        fontsize=9.5, pad=8
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame


def gravar_video_episodio(env, policy_net, arquivo_saida="bomberman_episodio.mp4", fps=4, tentar_vitoria=True):
    policy_net.eval()
    melhor_frames = None
    melhor_rec = -float("inf")
    melhor_info = None

    tentativas = 20 if tentar_vitoria else 1

    with torch.no_grad():
        for _ in range(tentativas):
            estado, _ = env.reset()
            done = False
            passo = 0
            rec_total = 0.0
            frames = [renderizar_frame(env, passo, rec_total)]

            while not done:
                tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                acao = int(torch.argmax(policy_net(tensor_s), dim=1).item())
                estado, r, term, trunc, info = env.step(acao)
                done = term or trunc

                passo += 1
                rec_total += r
                frames.append(renderizar_frame(env, passo, rec_total))

            frames.extend([frames[-1]] * fps)

            if rec_total > melhor_rec:
                melhor_rec = rec_total
                melhor_frames = frames
                melhor_info = info

            if info.get("vitoria", False):
                break

    imageio.mimsave(arquivo_saida, melhor_frames, fps=fps)
    print(f"Vídeo salvo em '{arquivo_saida}' | Vitória: {melhor_info.get('vitoria')} | Recompensa: {melhor_rec:.0f}")

    mp4 = open(arquivo_saida, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f'<video width="460" controls><source src="{data_url}" type="video/mp4"></video>')


# Executa e exibe o vídeo no Colab
gravar_video_episodio(env, policy_net, arquivo_saida="bomberman_episodio.mp4", fps=4)


if __name__ == "__main__":
    main()
