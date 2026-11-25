
class DQN_CNN(nn.Module):
    def __init__(self, num_canais=7, num_acoes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_canais, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, num_acoes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ReplayBuffer:
    def __init__(self, capacidade=50_000):
        self.buffer = deque(maxlen=capacidade)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_next, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def avaliar_agente(env, policy_net, num_episodios=30):
    policy_net.eval()
    vitorias, mortes, recompensas, inimigos, blocos, passos_lista = [], [], [], [], [], []

    with torch.no_grad():
        for _ in range(num_episodios):
            estado, _ = env.reset()
            done = False
            rep_ep, inim_ep, bloc_ep, passos_ep = 0.0, 0, 0, 0

            while not done:
                tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                acao = int(torch.argmax(policy_net(tensor_s), dim=1).item())
                estado, r, term, trunc, info = env.step(acao)
                done = term or trunc

                rep_ep += r
                passos_ep += 1
                inim_ep += info["inimigos_eliminados_passo"]
                bloc_ep += info["blocos_destruidos_passo"]

            vitorias.append(1 if info["vitoria"] else 0)
            mortes.append(1 if info["morreu"] else 0)
            recompensas.append(rep_ep)
            inimigos.append(inim_ep)
            blocos.append(bloc_ep)
            passos_lista.append(passos_ep)

    policy_net.train()
    return {
        "taxa_vitoria": float(np.mean(vitorias)) * 100.0,
        "taxa_morte": float(np.mean(mortes)) * 100.0,
        "recompensa_media": float(np.mean(recompensas)),
        "inimigos_medios": float(np.mean(inimigos)),
        "blocos_medios": float(np.mean(blocos)),
        "passos_medios": float(np.mean(passos_lista)),
    }


def treinar_dqn(
    env,
    episodios=1500,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    eps_inicio=1.0,
    eps_fim=0.05,
    eps_decay=0.997,
    target_update=25,
    intervalo_eval=50,
):
    estado_inicial, _ = env.reset()
    num_canais = estado_inicial.shape[0]

    policy_net = DQN_CNN(num_canais, env.num_acoes).to(device)
    target_net = DQN_CNN(num_canais, env.num_acoes).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    memoria = ReplayBuffer(capacidade=50_000)

    epsilon = eps_inicio
    logs_treino = {
        "recompensa": [],
        "passos": [],
        "vitoria": [],
        "morte": [],
        "blocos": [],
        "inimigos": [],
        "epsilon": [],
        "loss": [],
    }
    logs_eval = {
        "episodio": [],
        "taxa_vitoria": [],
        "taxa_morte": [],
        "recompensa_media": [],
        "inimigos_medios": [],
        "blocos_medios": [],
        "passos_medios": [],
    }

    for ep in range(1, episodios + 1):
        estado, _ = env.reset()
        done = False
        rep_ep, passos_ep, blocos_ep, inim_ep = 0.0, 0, 0, 0
        losses_ep = []

        while not done:
            if random.random() < epsilon:
                acao = random.randrange(env.num_acoes)
            else:
                with torch.no_grad():
                    tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                    acao = int(torch.argmax(policy_net(tensor_s), dim=1).item())

            prox_estado, recompensa, term, trunc, info = env.step(acao)
            done = term or trunc

            memoria.push(estado, acao, recompensa, prox_estado, float(done))
            estado = prox_estado

            rep_ep += recompensa
            passos_ep += 1
            blocos_ep += info["blocos_destruidos_passo"]
            inim_ep += info["inimigos_eliminados_passo"]

            if len(memoria) >= batch_size:
                s_b, a_b, r_b, s_next_b, d_b = memoria.sample(batch_size)

                s_t = torch.tensor(s_b, device=device)
                a_t = torch.tensor(a_b, device=device).unsqueeze(1)
                r_t = torch.tensor(r_b, device=device)
                s_next_t = torch.tensor(s_next_b, device=device)
                d_t = torch.tensor(d_b, device=device)

                q_atual = policy_net(s_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    q_prox_max = target_net(s_next_t).max(1)[0]
                    q_alvo = r_t + gamma * q_prox_max * (1.0 - d_t)

                loss = loss_fn(q_atual, q_alvo)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                losses_ep.append(loss.item())

        epsilon = max(eps_fim, epsilon * eps_decay)

        if ep % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        logs_treino["recompensa"].append(rep_ep)
        logs_treino["passos"].append(passos_ep)
        logs_treino["vitoria"].append(1 if info["vitoria"] else 0)
        logs_treino["morte"].append(1 if info["morreu"] else 0)
        logs_treino["blocos"].append(blocos_ep)
        logs_treino["inimigos"].append(inim_ep)
        logs_treino["epsilon"].append(epsilon)
        logs_treino["loss"].append(np.mean(losses_ep) if losses_ep else 0.0)

        if ep % intervalo_eval == 0:
            res = avaliar_agente(env, policy_net, num_episodios=30)
            logs_eval["episodio"].append(ep)
            for k in res:
                logs_eval[k].append(res[k])
            print(
                f"Ep {ep:4d} | Eps: {epsilon:.2f} | "
                f"Vitória: {res['taxa_vitoria']:5.1f}% | "
                f"Morte: {res['taxa_morte']:5.1f}% | "
                f"Inimigos: {res['inimigos_medios']:.2f}/2 | "
                f"Blocos: {res['blocos_medios']:.2f} | "
                f"Rec: {res['recompensa_media']:6.1f}"
            )

    torch.save(policy_net.state_dict(), "dqn_bomberman.pth")
    return policy_net, logs_treino, logs_eval


# Executa o treinamento
env = BombermanEnv()
policy_net, logs_treino, logs_eval = treinar_dqn(env, episodios=1500)
