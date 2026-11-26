class ActorCriticCNN(nn.Module):
    def __init__(self, num_canais=7, num_acoes=6):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(num_canais, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, num_acoes)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.base(x)
        return self.actor(feat), self.critic(feat).squeeze(-1)

    def agir(self, x, deterministico=False):
        logits, valor = self.forward(x)
        dist = Categorical(logits=logits)
        if deterministico:
            acao = torch.argmax(logits, dim=-1)
        else:
            acao = dist.sample()
        log_prob = dist.log_prob(acao)
        return acao, log_prob, dist.entropy(), valor


def avaliar_ppo(env, modelo, num_episodios=30):
    modelo.eval()
    vitorias, mortes, recompensas, inimigos, blocos, passos_lista = [], [], [], [], [], []

    with torch.no_grad():
        for _ in range(num_episodios):
            estado, _ = env.reset()
            done = False
            rep_ep, inim_ep, bloc_ep, passos_ep = 0.0, 0, 0, 0

            while not done:
                tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
                acao_t, _, _, _ = modelo.agir(tensor_s, deterministico=True)
                estado, r, term, trunc, info = env.step(int(acao_t.item()))
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

    modelo.train()
    return {
        "taxa_vitoria": float(np.mean(vitorias)) * 100.0,
        "taxa_morte": float(np.mean(mortes)) * 100.0,
        "recompensa_media": float(np.mean(recompensas)),
        "inimigos_medios": float(np.mean(inimigos)),
        "blocos_medios": float(np.mean(blocos)),
        "passos_medios": float(np.mean(passos_lista)),
    }


def treinar_ppo(
    env,
    episodios_alvo=1500,
    passos_por_coleta=1024,
    epocas_ppo=4,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=3e-4,
    coef_valor=0.5,
    coef_entropia=0.02,
    intervalo_eval=50,
):
    estado_inicial, _ = env.reset()
    modelo = ActorCriticCNN(estado_inicial.shape[0], env.num_acoes).to(device)
    optimizer = optim.Adam(modelo.parameters(), lr=lr)

    logs_ppo_treino = {
        "recompensa": [],
        "passos": [],
        "vitoria": [],
        "morte": [],
        "blocos": [],
        "inimigos": [],
        "entropia": [],
        "loss": [],
    }
    logs_ppo_eval = {
        "episodio": [],
        "taxa_vitoria": [],
        "taxa_morte": [],
        "recompensa_media": [],
        "inimigos_medios": [],
        "blocos_medios": [],
        "passos_medios": [],
    }

    ep_concluidos = 0
    prox_eval = intervalo_eval

    estado, _ = env.reset()
    rep_ep, passos_ep, blocos_ep, inim_ep = 0.0, 0, 0, 0

    while ep_concluidos < episodios_alvo:
        estados_b, acoes_b, logprobs_b, recompensas_b, dones_b, valores_b = [], [], [], [], [], []

        for _ in range(passos_por_coleta):
            tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                acao_t, logprob_t, _, valor_t = modelo.agir(tensor_s, deterministico=False)

            acao = int(acao_t.item())
            prox_estado, recompensa, term, trunc, info = env.step(acao)
            done = term or trunc

            estados_b.append(estado)
            acoes_b.append(acao)
            logprobs_b.append(logprob_t.item())
            recompensas_b.append(recompensa)
            dones_b.append(float(done))
            valores_b.append(valor_t.item())

            estado = prox_estado
            rep_ep += recompensa
            passos_ep += 1
            blocos_ep += info["blocos_destruidos_passo"]
            inim_ep += info["inimigos_eliminados_passo"]

            if done:
                ep_concluidos += 1
                logs_ppo_treino["recompensa"].append(rep_ep)
                logs_ppo_treino["passos"].append(passos_ep)
                logs_ppo_treino["vitoria"].append(1 if info["vitoria"] else 0)
                logs_ppo_treino["morte"].append(1 if info["morreu"] else 0)
                logs_ppo_treino["blocos"].append(blocos_ep)
                logs_ppo_treino["inimigos"].append(inim_ep)

                if ep_concluidos >= prox_eval and ep_concluidos <= episodios_alvo:
                    res = avaliar_ppo(env, modelo, num_episodios=30)
                    logs_ppo_eval["episodio"].append(prox_eval)
                    for k in res:
                        logs_ppo_eval[k].append(res[k])
                    print(
                        f"[PPO] Ep {prox_eval:4d} | "
                        f"Vitória: {res['taxa_vitoria']:5.1f}% | "
                        f"Morte: {res['taxa_morte']:5.1f}% | "
                        f"Inimigos: {res['inimigos_medios']:.2f}/2 | "
                        f"Blocos: {res['blocos_medios']:.2f} | "
                        f"Rec: {res['recompensa_media']:6.1f}"
                    )
                    prox_eval += intervalo_eval

                estado, _ = env.reset()
                rep_ep, passos_ep, blocos_ep, inim_ep = 0.0, 0, 0, 0

        with torch.no_grad():
            tensor_s = torch.tensor(estado, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, _, prox_valor_t = modelo.agir(tensor_s, deterministico=False)
            prox_valor = prox_valor_t.item()

        vantagens = np.zeros(passos_por_coleta, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(passos_por_coleta)):
            if t == passos_por_coleta - 1:
                next_non_terminal = 1.0 - dones_b[t]
                next_val = prox_valor
            else:
                next_non_terminal = 1.0 - dones_b[t]
                next_val = valores_b[t + 1]
            delta = recompensas_b[t] + gamma * next_val * next_non_terminal - valores_b[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            vantagens[t] = gae

        retornos = vantagens + np.array(valores_b, dtype=np.float32)
        vantagens = (vantagens - vantagens.mean()) / (vantagens.std() + 1e-8)

        s_t = torch.tensor(np.array(estados_b), dtype=torch.float32, device=device)
        a_t = torch.tensor(acoes_b, dtype=torch.long, device=device)
        old_logprobs_t = torch.tensor(logprobs_b, dtype=torch.float32, device=device)
        retornos_t = torch.tensor(retornos, dtype=torch.float32, device=device)
        vantagens_t = torch.tensor(vantagens, dtype=torch.float32, device=device)

        indices = np.arange(passos_por_coleta)
        losses_upd, entropias_upd = [], []

        for _ in range(epocas_ppo):
            np.random.shuffle(indices)
            for inicio in range(0, passos_por_coleta, batch_size):
                mb = indices[inicio : inicio + batch_size]
                logits, valores_pred = modelo(s_t[mb])
                dist = Categorical(logits=logits)
                novos_logprobs = dist.log_prob(a_t[mb])
                entropia = dist.entropy().mean()

                ratio = torch.exp(novos_logprobs - old_logprobs_t[mb])
                surr1 = ratio * vantagens_t[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * vantagens_t[mb]
                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = nn.functional.mse_loss(valores_pred, retornos_t[mb])

                loss = loss_actor + coef_valor * loss_critic - coef_entropia * entropia

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 0.5)
                optimizer.step()

                losses_upd.append(loss.item())
                entropias_upd.append(entropia.item())

        logs_ppo_treino["loss"].append(np.mean(losses_upd))
        logs_ppo_treino["entropia"].append(np.mean(entropias_upd))

    torch.save(modelo.state_dict(), "ppo_bomberman.pth")
    return modelo, logs_ppo_treino, logs_ppo_eval


# Executa o treinamento do PPO
env_ppo = BombermanEnv()
modelo_ppo, logs_ppo_treino, logs_ppo_eval = treinar_ppo(env_ppo, episodios_alvo=1500)