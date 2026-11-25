
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)


class BombermanEnv:
    """
    Grade 11x11 conforme especificação do MDP:
    - 6 ações: 0=Cima, 1=Baixo, 2=Esquerda, 3=Direita, 4=Plantar Bomba, 5=Ficar Parado
    - 2 inimigos que se movem e também plantam bombas
    - Recompensas: +10 bloco, +50 inimigo, +100 vitória, -100 morte, -1 passo
    """

    ACOES = {
        0: (-1, 0),  # Cima
        1: (1, 0),   # Baixo
        2: (0, -1),  # Esquerda
        3: (0, 1),   # Direita
        4: (0, 0),   # Plantar bomba
        5: (0, 0),   # Ficar parado
    }

    def __init__(
        self,
        tamanho=11,
        num_inimigos=2,
        alcance_explosao=2,
        timer_bomba=4,
        max_passos=200,
        prob_bloco=0.35,
        prob_bomba_inimigo=0.15,
    ):
        self.tamanho = tamanho
        self.num_inimigos = num_inimigos
        self.alcance_explosao = alcance_explosao
        self.timer_bomba = timer_bomba
        self.max_passos = max_passos
        self.prob_bloco = prob_bloco
        self.prob_bomba_inimigo = prob_bomba_inimigo
        self.num_acoes = 6

    def reset(self):
        self.passos = 0
        # 0 = vazio, 1 = parede fixa, 2 = bloco destrutível
        self.mapa = np.zeros((self.tamanho, self.tamanho), dtype=np.int32)

        # Bordas e pilares fixos internos (padrão clássico Bomberman)
        for r in range(self.tamanho):
            for c in range(self.tamanho):
                if r == 0 or r == self.tamanho - 1 or c == 0 or c == self.tamanho - 1:
                    self.mapa[r, c] = 1
                elif r % 2 == 0 and c % 2 == 0:
                    self.mapa[r, c] = 1

        # Posições iniciais (agente no canto superior esquerdo, inimigos opostos)
        self.agente_pos = [1, 1]
        cantos_inimigos = [[self.tamanho - 2, self.tamanho - 2], [1, self.tamanho - 2], [self.tamanho - 2, 1]]
        random.shuffle(cantos_inimigos)
        self.inimigos = [
            {"pos": list(cantos_inimigos[i]), "vivo": True, "bomba_ativa": False}
            for i in range(self.num_inimigos)
        ]

        # Zonas seguras ao redor do agente e dos inimigos para não nascerem presos
        zonas_livres = set()
        for pos in [self.agente_pos] + [ini["pos"] for ini in self.inimigos]:
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                zonas_livres.add((pos[0] + dr, pos[1] + dc))

        for r in range(1, self.tamanho - 1):
            for c in range(1, self.tamanho - 1):
                if self.mapa[r, c] == 0 and (r, c) not in zonas_livres:
                    if random.random() < self.prob_bloco:
                        self.mapa[r, c] = 2

        # Lista de bombas: {"pos": [r, c], "timer": int, "dono": "agente" | int}
        self.bombas = []
        self.chamas = set()
        return self._obter_estado(), {}

    def _celula_bloqueada(self, r, c):
        if self.mapa[r, c] in (1, 2):
            return True
        for b in self.bombas:
            if b["pos"] == [r, c]:
                return True
        return False

    def _agente_tem_bomba_disponivel(self):
        return not any(b["dono"] == "agente" for b in self.bombas)

    def _calcular_explosao(self, pos_bomba):
        r0, c0 = pos_bomba
        celulas = {(r0, c0)}
        blocos_destruidos = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, self.alcance_explosao + 1):
                nr, nc = r0 + dr * dist, c0 + dc * dist
                if not (0 <= nr < self.tamanho and 0 <= nc < self.tamanho):
                    break
                if self.mapa[nr, nc] == 1:  # parede fixa para a chama
                    break
                celulas.add((nr, nc))
                if self.mapa[nr, nc] == 2:  # bloco destrutível para a chama e é destruído
                    blocos_destruidos.append((nr, nc))
                    break
        return celulas, blocos_destruidos

    def step(self, acao):
        self.passos += 1
        self.chamas.clear()

        recompensa = -1.0  # custo por passo
        blocos_destruidos_passo = 0
        inimigos_eliminados_passo = 0
        morreu = False
        vitoria = False

        # 1. Ação do Agente
        if acao in (0, 1, 2, 3):
            dr, dc = self.ACOES[acao]
            nr, nc = self.agente_pos[0] + dr, self.agente_pos[1] + dc
            if not self._celula_bloqueada(nr, nc):
                self.agente_pos = [nr, nc]
        elif acao == 4:
            if self._agente_tem_bomba_disponivel():
                self.bombas.append({
                    "pos": list(self.agente_pos),
                    "timer": self.timer_bomba,
                    "dono": "agente",
                })
        # acao == 5: ficar parado

        # 2. Ação dos 2 Inimigos (movimento + plantar bomba)
        for idx, ini in enumerate(self.inimigos):
            if not ini["vivo"]:
                continue

            # Decide se planta bomba (se não tiver bomba ativa)
            tem_bomba = any(b["dono"] == idx for b in self.bombas)
            if not tem_bomba and random.random() < self.prob_bomba_inimigo:
                self.bombas.append({
                    "pos": list(ini["pos"]),
                    "timer": self.timer_bomba,
                    "dono": idx,
                })

            # Move para uma célula vizinha livre (evitando ficar em cima da própria bomba se possível)
            movimentos_validos = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = ini["pos"][0] + dr, ini["pos"][1] + dc
                if not self._celula_bloqueada(nr, nc):
                    movimentos_validos.append([nr, nc])
            if movimentos_validos:
                ini["pos"] = random.choice(movimentos_validos)

        # 3. Atualiza timers das bombas e processa explosões (inclui reação em cadeia)
        for b in self.bombas:
            b["timer"] -= 1

        explodiram = [b for b in self.bombas if b["timer"] <= 0]
        restantes = [b for b in self.bombas if b["timer"] > 0]

        while explodiram:
            bomba_atual = explodiram.pop(0)
            celulas_fogo, blocos_atingidos = self._calcular_explosao(bomba_atual["pos"])
            self.chamas.update(celulas_fogo)

            for br, bc in blocos_atingidos:
                if self.mapa[br, bc] == 2:
                    self.mapa[br, bc] = 0
                    blocos_destruidos_passo += 1

            # Reação em cadeia se outra bomba estiver na área da chama
            novas_restantes = []
            for b_outra in restantes:
                if tuple(b_outra["pos"]) in celulas_fogo:
                    explodiram.append(b_outra)
                else:
                    novas_restantes.append(b_outra)
            restantes = novas_restantes

        self.bombas = restantes

        # 4. Verifica eliminações e morte do agente
        for ini in self.inimigos:
            if ini["vivo"] and tuple(ini["pos"]) in self.chamas:
                ini["vivo"] = False
                inimigos_eliminados_passo += 1

        if tuple(self.agente_pos) in self.chamas:
            morreu = True
        for ini in self.inimigos:
            if ini["vivo"] and ini["pos"] == self.agente_pos:
                morreu = True

        # 5. Recompensas exatas do slide
        recompensa += 10.0 * blocos_destruidos_passo
        recompensa += 50.0 * inimigos_eliminados_passo

        inimigos_vivos = sum(1 for ini in self.inimigos if ini["vivo"])
        if inimigos_vivos == 0 and not morreu:
            recompensa += 100.0
            vitoria = True

        if morreu:
            recompensa -= 100.0

        terminado = morreu or vitoria
        truncado = (self.passos >= self.max_passos) and not terminado

        info = {
            "vitoria": vitoria,
            "morreu": morreu,
            "blocos_destruidos_passo": blocos_destruidos_passo,
            "inimigos_eliminados_passo": inimigos_eliminados_passo,
        }
        return self._obter_estado(), recompensa, terminado, truncado, info

    def _obter_estado(self):
        """
        Retorna tensor espacial (7, 11, 11) normalizado em [0, 1] garantindo estado markoviano:
        Canal 0: Paredes fixas
        Canal 1: Blocos destrutíveis
        Canal 2: Bombas (valor = timer / timer_bomba)
        Canal 3: Chamas ativas
        Canal 4: Posição do agente
        Canal 5: Posição dos inimigos vivos
        Canal 6: Bombas disponíveis do agente (plano constante 1.0 ou 0.0)
        """
        estado = np.zeros((7, self.tamanho, self.tamanho), dtype=np.float32)
        estado[0] = (self.mapa == 1).astype(np.float32)
        estado[1] = (self.mapa == 2).astype(np.float32)

        for b in self.bombas:
            r, c = b["pos"]
            estado[2, r, c] = b["timer"] / float(self.timer_bomba)

        for r, c in self.chamas:
            estado[3, r, c] = 1.0

        ar, ac = self.agente_pos
        estado[4, ar, ac] = 1.0

        for ini in self.inimigos:
            if ini["vivo"]:
                ir, ic = ini["pos"]
                estado[5, ir, ic] = 1.0

        if self._agente_tem_bomba_disponivel():
            estado[6, :, :] = 1.0

        return estado
