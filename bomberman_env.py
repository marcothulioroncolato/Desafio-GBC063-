
"""Ambiente Bomberman simplificado (agente único) no padrão Gymnasium."""
import heapq
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BombermanEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    # 0=cima, 1=baixo, 2=esquerda, 3=direita, 4=colocar bomba, 5=ficar parado
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size=11, n_enemies=2, block_prob=0.3, bomb_timer=5,
                 bomb_range=2, max_steps=200, step_cost=1.0, shaping=0.0, enemy_bombs=True,
                 p_bomba_inimigo=0.05, p_bomba_perto=0.3, mascara=False, render_mode=None):
        super().__init__()
        self.size = size
        self.n_enemies = n_enemies
        self.block_prob = block_prob
        self.bomb_timer = bomb_timer
        self.bomb_range = bomb_range
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.shaping = shaping  # bônus por se aproximar do inimigo (0 = desligado)
        self.enemy_bombs = enemy_bombs  # inimigos também plantam bombas
        self.p_bomba_inimigo = p_bomba_inimigo  # chance por passo de plantar (longe do agente)
        self.p_bomba_perto = p_bomba_perto  # idem, com o agente a até 3 casas
        self.mascara = mascara  # True = informa em info["mask"] quais ações são seguras/úteis
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        # canais: 0 paredes, 1 blocos, 2 agente, 3 inimigos, 4 timer das bombas (do agente e dos inimigos), 5 chamas
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6, size, size), dtype=np.float32)

    # ---------- ciclo de vida ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = self.size
        self.walls = np.zeros((n, n), dtype=bool)
        self.walls[1::2, 1::2] = True
        self.agent = (0, 0)
        safe = {(0, 0), (0, 1), (1, 0)}
        self.blocks = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                if (not self.walls[i, j] and (i, j) not in safe
                        and self.np_random.random() < self.block_prob):
                    self.blocks[i, j] = True
        cand = [(i, j) for i in range(n) for j in range(n)
                if not self.walls[i, j] and not self.blocks[i, j] and i + j >= n // 2 + 2]
        idx = self.np_random.choice(len(cand), size=self.n_enemies, replace=False)
        self.enemies = [cand[k] for k in idx]
        self.bombs = []  # cada bomba: [linha, coluna, tempo restante, dono] (0=agente, 1=inimigo)
        self.flames = np.zeros((n, n), dtype=bool)
        self.steps = 0
        self._d_prev = self._dist_inimigo()
        return self._obs(), {"mask": self._mask_info()}

    def step(self, action):
        reward = -self.step_cost  # custo por passo
        self.steps += 1
        self.flames[:] = False

        if action in self.MOVES:
            dr, dc = self.MOVES[action]
            r, c = self.agent[0] + dr, self.agent[1] + dc
            if self._free(r, c):
                self.agent = (r, c)
        elif action == 4 and not any(b[3] == 0 for b in self.bombs):  # 1 bomba do agente por vez
            self.bombs.append([self.agent[0], self.agent[1], self.bomb_timer, 0])

        self._agir_inimigos()
        dead = self.agent in self.enemies
        causa = "contato com inimigo" if dead else None

        n_killed = 0
        for b in self.bombs:
            b[2] -= 1
        prontas = [b for b in self.bombs if b[2] <= 0]
        if prontas:
            self.bombs = [b for b in self.bombs if b[2] > 0]
            alcances = [(b, self._alcance(b[0], b[1])) for b in prontas]  # antes de destruir blocos
            for b, cells in alcances:
                if self.agent in cells:
                    causa = "própria bomba" if b[3] == 0 else "bomba do inimigo"
            do_agente = set()
            for b, cells in alcances:
                for (r, c) in cells:
                    self.flames[r, c] = True
                    if b[3] == 0 and self.blocks[r, c]:
                        do_agente.add((r, c))
            reward += 10.0 * len(do_agente)  # blocos destruídos pelas bombas do agente
            self.blocks[self.flames] = False
            killed = [e for e in self.enemies if self.flames[e]]
            self.enemies = [e for e in self.enemies if not self.flames[e]]
            n_killed = len(killed)
            reward += 50.0 * n_killed  # inimigos eliminados (por qualquer explosão)
            if self.flames[self.agent]:
                dead = True

        if self.shaping > 0 and self.enemies:
            d_novo = self._dist_inimigo()
            if n_killed == 0 and not dead:
                reward += self.shaping * (self._d_prev - d_novo)  # aproximou = positivo
            self._d_prev = d_novo

        terminated = False
        win = False
        if dead:
            reward -= 100.0
            terminated = True
        elif self._won():
            reward += 100.0
            terminated = True
            win = True
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"win": win, "mask": self._mask_info(), "causa": causa if dead else None}
        return self._obs(), reward, terminated, truncated, info

    # ---------- auxiliares ----------
    def _won(self):
        """Vitória: todos os inimigos mortos; sem inimigos, todos os blocos destruídos."""
        if self.n_enemies > 0:
            return not self.enemies
        return not self.blocks.any()

    def _dist_inimigo(self):
        """Custo do caminho até o inimigo mais próximo. Parede não passa; bloco custa 3,
        então destruir um bloco que está no caminho encurta a distância."""
        if not self.enemies:
            return 0.0
        n = self.size
        alvo = set(self.enemies)
        dist = {self.agent: 0}
        fila = [(0, self.agent)]
        while fila:
            d, (r, c) = heapq.heappop(fila)
            if (r, c) in alvo:
                return float(d)
            if d > dist[(r, c)]:
                continue
            for dr, dc in self.MOVES.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not self.walls[nr, nc]:
                    nd = d + (3 if self.blocks[nr, nc] else 1)
                    if nd < dist.get((nr, nc), 10**9):
                        dist[(nr, nc)] = nd
                        heapq.heappush(fila, (nd, (nr, nc)))
        return float(2 * n * n)

    def _free(self, r, c):
        n = self.size
        if not (0 <= r < n and 0 <= c < n):
            return False
        if self.walls[r, c] or self.blocks[r, c]:
            return False
        if self._bomba_em(r, c):
            return False
        return True

    def _mask_info(self):
        return self.action_mask() if self.mascara else np.ones(6, dtype=np.float32)

    def action_mask(self):
        """6 flags (1 = ação permitida). Bloqueia o que é inútil ou suicida: andar contra
        parede/bloco/inimigo, plantar sem rota de fuga a tempo, entrar onde não dá tempo
        de sair da explosão e ficar parado dentro da zona de uma bomba."""
        mask = np.zeros(6, dtype=np.float32)
        zona = self._perigo()
        for a, (dr, dc) in self.MOVES.items():
            p = (self.agent[0] + dr, self.agent[1] + dc)
            if self._free(*p) and p not in self.enemies and self._ok_pos(p, zona):
                mask[a] = 1.0
        if not any(b[3] == 0 for b in self.bombs):  # 1 bomba do agente por vez
            nova = zona | set(self._alcance(*self.agent))
            d, _ = self._bfs_seguro(self.agent, nova)
            if d is not None and d <= self.bomb_timer - 1:  # tem tempo de escapar
                mask[4] = 1.0
        if self.agent not in zona:
            mask[5] = 1.0
        if mask.sum() == 0:  # sem saída: libera tudo para não travar
            mask[:] = 1.0
        return mask

    def _ok_pos(self, pos, zona):
        """Ir para 'pos' é seguro? Fora da zona de perigo, sim; dentro, só se der tempo de sair."""
        if pos not in zona:
            return True
        d, _ = self._bfs_seguro(pos, zona)
        if d is None:
            return False
        t_min = min(b[2] for b in self.bombs if pos in self._alcance(b[0], b[1]))
        return d <= t_min - 1

    def _bomba_em(self, r, c):
        return any(b[0] == r and b[1] == c for b in self.bombs)

    def _alcance(self, r0, c0):
        """Células atingidas por uma bomba em (r0, c0): a própria casa e até bomb_range
        casas nas 4 direções. A explosão para na parede e no primeiro bloco."""
        n = self.size
        cells = [(r0, c0)]
        for dr, dc in self.MOVES.values():
            for k in range(1, self.bomb_range + 1):
                r, c = r0 + dr * k, c0 + dc * k
                if not (0 <= r < n and 0 <= c < n) or self.walls[r, c]:
                    break
                cells.append((r, c))
                if self.blocks[r, c]:
                    break
        return cells

    def _perigo(self):
        zona = set()
        for b in self.bombs:
            zona.update(self._alcance(b[0], b[1]))
        return zona

    def _bfs_seguro(self, pos, zona):
        """(distância, primeiro passo) até a célula fora da zona de perigo mais próxima."""
        fila = deque([(pos, 0, None)])
        visto = {pos}
        while fila:
            (r, c), d, primeiro = fila.popleft()
            if (r, c) not in zona:
                return d, primeiro
            for dr, dc in self.MOVES.values():
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visto and self._free(nr, nc):
                    visto.add((nr, nc))
                    fila.append(((nr, nc), d + 1, primeiro or (nr, nc)))
        return None, None

    def _agir_inimigos(self):
        """Inimigos (regras fixas, não aprendem): fogem de bombas, às vezes plantam uma
        (só se houver rota de fuga a tempo) e, no resto do tempo, andam ao acaso."""
        novos = []
        for pos in self.enemies:
            zona = self._perigo()
            if pos in zona:  # em perigo: foge pelo caminho mais curto
                _, passo = self._bfs_seguro(pos, zona)
                novos.append(passo if passo is not None else pos)
                continue
            if self.enemy_bombs and self._quer_plantar(pos, zona):
                self.bombs.append([pos[0], pos[1], self.bomb_timer, 1])
                novos.append(pos)
                continue
            opcoes = [pos] + [(pos[0] + dr, pos[1] + dc) for dr, dc in self.MOVES.values()
                              if self._free(pos[0] + dr, pos[1] + dc)]
            seguras = [o for o in opcoes if o not in zona]
            opcoes = seguras or opcoes
            novos.append(opcoes[self.np_random.integers(len(opcoes))])
        self.enemies = novos

    def _quer_plantar(self, pos, zona):
        if sum(1 for b in self.bombs if b[3] == 1) >= len(self.enemies):
            return False
        perto = abs(pos[0] - self.agent[0]) + abs(pos[1] - self.agent[1]) <= 3
        if self.np_random.random() >= (self.p_bomba_perto if perto else self.p_bomba_inimigo):
            return False
        nova = zona | set(self._alcance(pos[0], pos[1]))
        d, _ = self._bfs_seguro(pos, nova)
        return d is not None and d <= self.bomb_timer - 2  # tem tempo de escapar

    def _obs(self):
        n = self.size
        obs = np.zeros((6, n, n), dtype=np.float32)
        obs[0] = self.walls
        obs[1] = self.blocks
        obs[2][self.agent] = 1.0
        for e in self.enemies:
            obs[3][e] = 1.0
        for r, c, t, _ in self.bombs:
            obs[4][r, c] = t / self.bomb_timer
        obs[5] = self.flames
        return obs

    def render(self):
        n = self.size
        rows = []
        for i in range(n):
            row = ""
            for j in range(n):
                if (i, j) == self.agent:
                    row += "A"
                elif (i, j) in self.enemies:
                    row += "E"
                elif self._bomba_em(i, j):
                    row += "B"
                elif self.flames[i, j]:
                    row += "*"
                elif self.walls[i, j]:
                    row += "#"
                elif self.blocks[i, j]:
                    row += "+"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)


if __name__ == "__main__":
    env = BombermanEnv()
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        print(f"episódio {ep}: passos={env.steps} recompensa={total:.0f}")
    print(env.render())

