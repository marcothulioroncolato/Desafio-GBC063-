from IPython.display import Image, display


def renderizar_frame_multi(env, ep_atual, total_eps, passo, recompensa_acumulada, titulo_algo="PPO", escala=40):
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

    fig, ax = plt.subplots(figsize=(5.5, 6.0), dpi=90)
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
        f"[{titulo_algo}] Episódio {ep_atual}/{total_eps} | Passo: {passo}\n"
        f"Recompensa: {recompensa_acumulada:.0f} | Inimigos vivos: {inimigos_vivos}/2",
        fontsize=10, pad=8
    )

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame


def gravar_gif_episodios(
    env,
    modelo,
    arquivo_saida="bomberman_episodios.gif",
    num_episodios=5,
    fps=4,
    tipo="PPO",  # use "PPO" para modelo_ppo ou "DQN" para policy_net
):
    modelo.eval()
    todos_frames = []
    vitorias = 0

    with torch.no_grad():
        for ep in range(1, num_episodios + 1):
            estado, _ = env.reset()
            done = False
            passo = 0
            rec_total = 0.0
            todos_frames.append(renderizar_frame_multi(env, ep, num_episodios, passo, rec_total, tipo))

            while not done:
                tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                if tipo == "PPO":
                    acao_t, _, _, _ = modelo.agir(tensor_s, deterministico=True)
                    acao = int(acao_t.item())
                else:
                    acao = int(torch.argmax(modelo(tensor_s), dim=1).item())

                estado, r, term, trunc, info = env.step(acao)
                done = term or trunc

                passo += 1
                rec_total += r
                todos_frames.append(renderizar_frame_multi(env, ep, num_episodios, passo, rec_total, tipo))

            if info.get("vitoria", False):
                vitorias += 1

            # Pausa de 1.5 segundo ao fim de cada episódio para visualizar o desfecho
            todos_frames.extend([todos_frames[-1]] * int(fps * 1.5))
            print(f"Episódio {ep}/{num_episodios} concluído | Vitória: {info.get('vitoria')} | Recompensa: {rec_total:.0f}")

    # loop=0 faz o GIF repetir continuamente
    imageio.mimsave(arquivo_saida, todos_frames, fps=fps, loop=0)
    print(f"\nGIF salvo em '{arquivo_saida}' ({vitorias}/{num_episodios} vitórias)!")

    with open(arquivo_saida, "rb") as f:
        display(Image(data=f.read(), format="png"))


# Para gravar 5 episódios do PPO:
gravar_gif_episodios(env_ppo, modelo_ppo, arquivo_saida="bomberman_ppo_5eps.gif", num_episodios=5, fps=4, tipo="PPO")

# Se quiser gravar também 5 episódios do DQN para comparar lado a lado nos slides:
# gravar_gif_episodios(env, policy_net, arquivo_saida="bomberman_dqn_5eps.gif", num_episodios=5, fps=4, tipo="DQN")