
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BombermanEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    # 0=cima, 1=baixo, 2=esquerda, 3=direita, 4=colocar bomba, 5=ficar parado
    MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size=11, n_enemies=2, block_prob=0.3, bomb_timer=3,
                 bomb_range=2, max_steps=200, step_cost=1.0, render_mode=None):
        super().__init__()
        self.size = size
        self.n_enemies = n_enemies
        self.block_prob = block_prob
        self.bomb_timer = bomb_timer
        self.bomb_range = bomb_range
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        # canais: 0 paredes, 1 blocos, 2 agente, 3 inimigos, 4 timer da bomba, 5 chamas
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
        self.bomb = None  # [linha, coluna, tempo restante]
        self.flames = np.zeros((n, n), dtype=bool)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        reward = -self.step_cost  # custo por passo
        self.steps += 1
        self.flames[:] = False

        if action in self.MOVES:
            dr, dc = self.MOVES[action]
            r, c = self.agent[0] + dr, self.agent[1] + dc
            if self._free(r, c):
                self.agent = (r, c)
        elif action == 4 and self.bomb is None:
            self.bomb = [self.agent[0], self.agent[1], self.bomb_timer]

        self._move_enemies()
        dead = self.agent in self.enemies

        if self.bomb is not None:
            self.bomb[2] -= 1
            if self.bomb[2] <= 0:
                for (r, c) in self._explode():
                    self.flames[r, c] = True
                    if self.blocks[r, c]:
                        self.blocks[r, c] = False
                        reward += 10.0  # bloco destruído
                killed = [e for e in self.enemies if self.flames[e]]
                self.enemies = [e for e in self.enemies if not self.flames[e]]
                reward += 50.0 * len(killed)  # inimigos eliminados
                if self.flames[self.agent]:
                    dead = True

        terminated = False
        win = False
        if dead:
            reward -= 100.0
            terminated = True
        elif not self.enemies:
            reward += 100.0
            terminated = True
            win = True
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"win": win}

    # ---------- auxiliares ----------
    def _free(self, r, c):
        n = self.size
        if not (0 <= r < n and 0 <= c < n):
            return False
        if self.walls[r, c] or self.blocks[r, c]:
            return False
        if self.bomb is not None and (r, c) == (self.bomb[0], self.bomb[1]):
            return False
        return True

    def _move_enemies(self):
        new = []
        for (r, c) in self.enemies:
            options = [(r, c)]
            for dr, dc in self.MOVES.values():
                if self._free(r + dr, c + dc):
                    options.append((r + dr, c + dc))
            new.append(options[self.np_random.integers(len(options))])
        self.enemies = new

    def _explode(self):
        r0, c0, _ = self.bomb
        self.bomb = None
        n = self.size
        cells = [(r0, c0)]
        for dr, dc in self.MOVES.values():
            for k in range(1, self.bomb_range + 1):
                r, c = r0 + dr * k, c0 + dc * k
                if not (0 <= r < n and 0 <= c < n) or self.walls[r, c]:
                    break
                cells.append((r, c))
                if self.blocks[r, c]:
                    break  # a explosão para no primeiro bloco
        return cells

    def _obs(self):
        n = self.size
        obs = np.zeros((6, n, n), dtype=np.float32)
        obs[0] = self.walls
        obs[1] = self.blocks
        obs[2][self.agent] = 1.0
        for e in self.enemies:
            obs[3][e] = 1.0
        if self.bomb is not None:
            obs[4][self.bomb[0], self.bomb[1]] = self.bomb[2] / self.bomb_timer
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
                elif self.bomb is not None and (i, j) == (self.bomb[0], self.bomb[1]):
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
