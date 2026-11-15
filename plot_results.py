"""Gera os gráficos para a apresentação a partir dos CSVs do treino.
Uso: python plot_results.py [pasta]   (padrão: pasta atual)"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def media_movel(y, k):
    k = max(1, min(k, len(y)))
    return np.convolve(y, np.ones(k) / k, mode="valid"), k


def ler(caminho):
    d = np.genfromtxt(caminho, delimiter=",", names=True)
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def salvar(fig, pasta, nome):
    fig.savefig(os.path.join(pasta, nome), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", nome)


def main(pasta="."):
    d = ler(os.path.join(pasta, "rewards.csv"))
    ep = d["episodio"]
    k = 50

    # 1) recompensa por episódio
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, d["recompensa"], alpha=0.25, label="por episódio")
    m, kk = media_movel(d["recompensa"], k)
    ax.plot(ep[kk - 1:], m, linewidth=2, label=f"média móvel ({kk} episódios)")
    ax.set_title("Recompensa total por episódio")
    ax.set_xlabel("episódio"); ax.set_ylabel("recompensa total")
    ax.grid(alpha=0.3); ax.legend()
    salvar(fig, pasta, "1_recompensa.png")

    # 2) taxa de vitória
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m, kk = media_movel(d["vitoria"], 100)
    ax.plot(ep[kk - 1:], m * 100, linewidth=2, label=f"treino (média de {kk} episódios)")
    av = os.path.join(pasta, "avaliacoes.csv")
    if os.path.exists(av):
        a = ler(av)
        if len(a["episodio"]):
            ax.plot(a["episodio"], a["vitoria"] * 100, "o--", label="avaliação sem exploração")
    ax.set_title("Taxa de vitória (%)")
    ax.set_xlabel("episódio"); ax.set_ylabel("vitórias (%)"); ax.set_ylim(0, 100)
    ax.grid(alpha=0.3); ax.legend()
    salvar(fig, pasta, "2_taxa_vitoria.png")

    # 3) duração dos episódios
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m, kk = media_movel(d["passos"], k)
    ax.plot(ep[kk - 1:], m, linewidth=2)
    ax.set_title("Passos por episódio (média móvel)")
    ax.set_xlabel("episódio"); ax.set_ylabel("passos")
    ax.grid(alpha=0.3)
    salvar(fig, pasta, "3_passos.png")

    # 4) perda e epsilon
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    perda = d["perda_media"]
    ok = ~np.isnan(perda)
    ax1.plot(ep[ok], perda[ok], alpha=0.8)
    ax1.set_title("Perda média por episódio")
    ax1.set_xlabel("episódio"); ax1.set_ylabel("perda"); ax1.grid(alpha=0.3)
    ax2.plot(ep, d["epsilon"])
    ax2.set_title("Epsilon (exploração)")
    ax2.set_xlabel("episódio"); ax2.set_ylabel("epsilon"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    salvar(fig, pasta, "4_perda_epsilon.png")

    # 5) DQN x agente aleatório
    fin = os.path.join(pasta, "resultados_finais.csv")
    if os.path.exists(fin):
        with open(fin, encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))
        nomes = [l["agente"] for l in linhas]
        cores = ["tab:blue", "tab:gray"][: len(nomes)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        ax1.bar(nomes, [float(l["recompensa"]) for l in linhas], color=cores)
        ax1.set_title("Recompensa média"); ax1.axhline(0, color="k", linewidth=0.8)
        ax2.bar(nomes, [float(l["vitoria"]) * 100 for l in linhas], color=cores)
        ax2.set_title("Taxa de vitória (%)"); ax2.set_ylim(0, 100)
        fig.suptitle("DQN treinado x agente aleatório (100 mapas)")
        fig.tight_layout()
        salvar(fig, pasta, "5_dqn_vs_aleatorio.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
