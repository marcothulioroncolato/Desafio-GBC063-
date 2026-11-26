def media_movel(dados, janela=40):
    if len(dados) < janela:
        return np.array(dados)
    return np.convolve(dados, np.ones(janela) / janela, mode="valid")


def plotar_resultados_ppo(logs_ppo_treino, logs_ppo_eval, janela=40):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Recompensa acumulada por episódio
    axs[0, 0].plot(logs_ppo_treino["recompensa"], alpha=0.2, color="royalblue", label="Por episódio")
    mm_rep = media_movel(logs_ppo_treino["recompensa"], janela)
    axs[0, 0].plot(range(janela - 1, len(logs_ppo_treino["recompensa"])), mm_rep, color="navy", linewidth=2, label=f"Média móvel ({janela})")
    axs[0, 0].set_title("1. Recompensa Acumulada (Treino PPO)")
    axs[0, 0].set_xlabel("Episódio")
    axs[0, 0].set_ylabel("Recompensa")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Taxa de Vitória vs Taxa de Morte na Avaliação
    axs[0, 1].plot(logs_ppo_eval["episodio"], logs_ppo_eval["taxa_vitoria"], marker="o", color="green", linewidth=2, label="Vitória (%)")
    axs[0, 1].plot(logs_ppo_eval["episodio"], logs_ppo_eval["taxa_morte"], marker="x", color="crimson", linewidth=2, label="Morte (%)")
    axs[0, 1].set_title("2. Desfecho dos Episódios (Avaliação PPO)")
    axs[0, 1].set_xlabel("Episódio de Treino")
    axs[0, 1].set_ylabel("Percentual (%)")
    axs[0, 1].set_ylim(-5, 105)
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Inimigos Eliminados e Blocos Destruídos
    axs[0, 2].plot(logs_ppo_eval["episodio"], logs_ppo_eval["inimigos_medios"], marker="^", color="darkorange", linewidth=2, label="Inimigos eliminados (máx 2)")
    axs[0, 2].plot(logs_ppo_eval["episodio"], logs_ppo_eval["blocos_medios"], marker="s", color="teal", linewidth=2, label="Blocos destruídos")
    axs[0, 2].set_title("3. Objetivos Cumpridos por Episódio (Avaliação PPO)")
    axs[0, 2].set_xlabel("Episódio de Treino")
    axs[0, 2].set_ylabel("Média por episódio")
    axs[0, 2].legend()
    axs[0, 2].grid(True, alpha=0.3)

    # 4. Recompensa Média na Avaliação
    axs[1, 0].plot(logs_ppo_eval["episodio"], logs_ppo_eval["recompensa_media"], marker="o", color="darkgreen", linewidth=2)
    axs[1, 0].set_title("4. Recompensa Média (Avaliação PPO)")
    axs[1, 0].set_xlabel("Episódio de Treino")
    axs[1, 0].set_ylabel("Recompensa média")
    axs[1, 0].grid(True, alpha=0.3)

    # 5. Duração do Episódio (Passos) e Entropia da Política
    ax_p = axs[1, 1]
    mm_passos = media_movel(logs_ppo_treino["passos"], janela)
    l1 = ax_p.plot(range(janela - 1, len(logs_ppo_treino["passos"])), mm_passos, color="purple", linewidth=2, label="Passos (média móvel)")
    ax_p.set_title("5. Duração do Episódio (Treino PPO)")
    ax_p.set_xlabel("Episódio")
    ax_p.set_ylabel("Passos", color="purple")
    ax_p.legend(loc="upper right")
    ax_p.grid(True, alpha=0.3)

    # 6. Loss Total do PPO por Atualização
    axs[1, 2].plot(logs_ppo_treino["loss"], color="brown", linewidth=1.8)
    axs[1, 2].set_title("6. Loss do PPO por Atualização")
    axs[1, 2].set_xlabel("Iteração de Atualização")
    axs[1, 2].set_ylabel("PPO Loss")
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultados_ppo.png", dpi=200)
    plt.show()


def plotar_comparacao_ppo_dqn(logs_ppo_eval, logs_dqn_eval):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(logs_ppo_eval["episodio"], logs_ppo_eval["taxa_vitoria"], marker="o", color="royalblue", linewidth=2, label="PPO")
    axs[0, 0].plot(logs_dqn_eval["episodio"], logs_dqn_eval["taxa_vitoria"], marker="s", color="darkorange", linewidth=2, label="DQN")
    axs[0, 0].set_title("Comparação 1: Taxa de Vitória (%)")
    axs[0, 0].set_xlabel("Episódio de Treino")
    axs[0, 0].set_ylabel("Vitórias (%)")
    axs[0, 0].set_ylim(-5, 105)
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(logs_ppo_eval["episodio"], logs_ppo_eval["recompensa_media"], marker="o", color="royalblue", linewidth=2, label="PPO")
    axs[0, 1].plot(logs_dqn_eval["episodio"], logs_dqn_eval["recompensa_media"], marker="s", color="darkorange", linewidth=2, label="DQN")
    axs[0, 1].set_title("Comparação 2: Recompensa Média")
    axs[0, 1].set_xlabel("Episódio de Treino")
    axs[0, 1].set_ylabel("Recompensa")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(logs_ppo_eval["episodio"], logs_ppo_eval["inimigos_medios"], marker="o", color="royalblue", linewidth=2, label="PPO")
    axs[1, 0].plot(logs_dqn_eval["episodio"], logs_dqn_eval["inimigos_medios"], marker="s", color="darkorange", linewidth=2, label="DQN")
    axs[1, 0].set_title("Comparação 3: Inimigos Eliminados (máx 2)")
    axs[1, 0].set_xlabel("Episódio de Treino")
    axs[1, 0].set_ylabel("Inimigos")
    axs[1, 0].set_ylim(-0.1, 2.2)
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(logs_ppo_eval["episodio"], logs_ppo_eval["taxa_morte"], marker="o", color="royalblue", linewidth=2, label="PPO")
    axs[1, 1].plot(logs_dqn_eval["episodio"], logs_dqn_eval["taxa_morte"], marker="s", color="darkorange", linewidth=2, label="DQN")
    axs[1, 1].set_title("Comparação 4: Taxa de Morte (%)")
    axs[1, 1].set_xlabel("Episódio de Treino")
    axs[1, 1].set_ylabel("Mortes (%)")
    axs[1, 1].set_ylim(-5, 105)
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparacao_ppo_vs_dqn.png", dpi=200)
    plt.show()


plotar_resultados_ppo(logs_ppo_treino, logs_ppo_eval)

# Se `logs_eval` (do treino anterior do DQN) estiver na memória, plota a comparação lado a lado
if "logs_eval" in globals():
    plotar_comparacao_ppo_dqn(logs_ppo_eval, logs_eval)

resultado_final_ppo = avaliar_ppo(env_ppo, modelo_ppo, num_episodios=100)
print("\n=== RESULTADO FINAL DO TESTE PPO (100 episódios) ===")
for chave, valor in resultado_final_ppo.items():
    print(f"{chave}: {valor:.2f}")