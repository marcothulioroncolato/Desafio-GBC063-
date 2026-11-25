def media_movel(dados, janela=40):
    if len(dados) < janela:
        return np.array(dados)
    return np.convolve(dados, np.ones(janela) / janela, mode="valid")


def plotar_resultados(logs_treino, logs_eval, janela=40):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Recompensa acumulada por episódio
    axs[0, 0].plot(logs_treino["recompensa"], alpha=0.2, color="steelblue", label="Por episódio")
    mm_rep = media_movel(logs_treino["recompensa"], janela)
    axs[0, 0].plot(range(janela - 1, len(logs_treino["recompensa"])), mm_rep, color="navy", linewidth=2, label=f"Média móvel ({janela})")
    axs[0, 0].set_title("1. Recompensa Acumulada (Treino)")
    axs[0, 0].set_xlabel("Episódio")
    axs[0, 0].set_ylabel("Recompensa")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Taxa de Vitória vs Taxa de Morte na Avaliação (epsilon = 0)
    axs[0, 1].plot(logs_eval["episodio"], logs_eval["taxa_vitoria"], marker="o", color="green", linewidth=2, label="Vitória (%)")
    axs[0, 1].plot(logs_eval["episodio"], logs_eval["taxa_morte"], marker="x", color="crimson", linewidth=2, label="Morte (%)")
    axs[0, 1].set_title("2. Desfecho dos Episódios (Avaliação ε = 0)")
    axs[0, 1].set_xlabel("Episódio de Treino")
    axs[0, 1].set_ylabel("Percentual (%)")
    axs[0, 1].set_ylim(-5, 105)
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Inimigos Eliminados e Blocos Destruídos (Avaliação)
    axs[0, 2].plot(logs_eval["episodio"], logs_eval["inimigos_medios"], marker="^", color="darkorange", linewidth=2, label="Inimigos eliminados (máx 2)")
    axs[0, 2].plot(logs_eval["episodio"], logs_eval["blocos_medios"], marker="s", color="teal", linewidth=2, label="Blocos destruídos")
    axs[0, 2].set_title("3. Objetivos Cumpridos por Episódio (Avaliação)")
    axs[0, 2].set_xlabel("Episódio de Treino")
    axs[0, 2].set_ylabel("Média por episódio")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # 4. Recompensa Média na Avaliação (sem ruído de exploração)
    axs[1, 0].plot(logs_eval["episodio"], logs_eval["recompensa_media"], marker="o", color="darkgreen", linewidth=2)
    axs[1, 0].set_title("4. Recompensa Média (Avaliação ε = 0)")
    axs[1, 0].set_xlabel("Episódio de Treino")
    axs[1, 0].set_ylabel("Recompensa média")
    axs[1, 0].grid(True, alpha=0.3)

    # 5. Duração do Episódio (Passos) e Decaimento do Epsilon
    ax_p = axs[1, 1]
    mm_passos = media_movel(logs_treino["passos"], janela)
    l1 = ax_p.plot(range(janela - 1, len(logs_treino["passos"])), mm_passos, color="purple", linewidth=2, label="Passos (média móvel)")
    ax_p.set_title("5. Duração do Episódio vs. Epsilon (ε)")
    ax_p.set_xlabel("Episódio")
    ax_p.set_ylabel("Passos", color="purple")
    ax_p.grid(True, alpha=0.3)

    ax_e = ax_p.twinx()
    l2 = ax_e.plot(logs_treino["epsilon"], color="orange", linestyle="--", label="Epsilon (ε)")
    ax_e.set_ylabel("Epsilon", color="orange")
    linhas = l1 + l2
    ax_p.legend(linhas, [l.get_label() for l in linhas], loc="upper right")

    # 6. Loss da Rede Neural
    mm_loss = media_movel(logs_treino["loss"], janela)
    axs[1, 2].plot(range(janela - 1, len(logs_treino["loss"])), mm_loss, color="brown", linewidth=2)
    axs[1, 2].set_title("6. Loss do DQN (Média Móvel)")
    axs[1, 2].set_xlabel("Episódio")
    axs[1, 2].set_ylabel("Smooth L1 Loss")
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultados_dqn.png", dpi=200)
    plt.show()


plotar_resultados(logs_treino, logs_eval)

# Teste final em 100 episódios puramente greedy (epsilon = 0)
resultado_final = avaliar_agente(env, policy_net, num_episodios=100)
print("\n=== RESULTADO FINAL DO TESTE (100 episódios, ε = 0) ===")
for chave, valor in resultado_final.items():
    print(f"{chave}: {valor:.2f}")