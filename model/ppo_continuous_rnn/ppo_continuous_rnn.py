"""
model/ppo_continuous_rnn/ppo_continuous_rnn.py
连续动作 Beta‑PPO + RNN
"""

import torch, torch.nn as nn, numpy as np, statistics
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SequentialSampler

def orthogonal_init(layer, gain=np.sqrt(2)):
    for n, p in layer.named_parameters():
        if "bias" in n:   nn.init.constant_(p, 0.)
        elif "weight" in n: nn.init.orthogonal_(p, gain=gain)
    return layer

class Actor_Critic_Beta_RNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.act_dim   = args.action_dim
        self.hid_dim   = args.hidden_dim
        self.use_gru   = args.use_gru
        act_fn         = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # --- Actor ---
        self.a_fc1 = nn.Linear(args.state_dim, self.hid_dim)
        self.a_rnn = (nn.GRU if self.use_gru else nn.LSTM)(self.hid_dim, self.hid_dim, batch_first=True)
        self.a_fc2 = nn.Linear(self.hid_dim, self.act_dim*2)   # α, β

        # --- Critic ---
        self.c_fc1 = nn.Linear(args.state_dim, self.hid_dim)
        self.c_rnn = (nn.GRU if self.use_gru else nn.LSTM)(self.hid_dim, self.hid_dim, batch_first=True)
        self.c_fc2 = nn.Linear(self.hid_dim, 1)

        self.act_fn = act_fn
        if args.use_orthogonal_init:
            for m in [self.a_fc1, self.a_rnn, self.a_fc2,
                      self.c_fc1, self.c_rnn, self.c_fc2]:
                gain = 0.01 if m is self.a_fc2 else np.sqrt(2)
                orthogonal_init(m, gain)

    # ---------- forward ----------
    def actor(self, s):
        x = self.act_fn(self.a_fc1(s))
        x, self.a_hid = self.a_rnn(x, self.a_hid)
        return self.a_fc2(x)

    def critic(self, s):
        x = self.act_fn(self.c_fc1(s))
        x, self.c_hid = self.c_rnn(x, self.c_hid)
        return self.c_fc2(x)

class PPO_continuous_RNN:
    def __init__(self, args):
        self.args = args
        self.ac = Actor_Critic_Beta_RNN(args).to(args.device)
        self.opt = torch.optim.Adam(self.ac.parameters(),
                                    lr=args.lr, eps=(1e-5 if args.set_adam_eps else 1e-8))
        # cache
        self.M = args.batch_size
        self.mini_M = args.mini_batch_size
        self.clip = args.epsilon
        self.K = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.device = args.device
        self.lr0 = args.lr
        self.max_steps = args.max_train_steps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay

    # —— RNN hidden state reset ——
    def reset_rnn(self):
        self.ac.a_hid = None
        self.ac.c_hid = None

    # —— 选动作 ——
    def choose_action(self, s, evaluate=False):
        s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, state]
        with torch.no_grad():
            out = self.ac.actor(s)               # [1, 2*act_dim]
            alpha, beta = torch.chunk(nn.Softplus()(out)+1.0, 2, dim=-1)
            dist = Beta(alpha, beta)
            if evaluate:
                a = alpha / (alpha+beta)         # 均值
            else:
                a = dist.rsample()
            logp = dist.log_prob(a).sum(-1)      # scalar
            return a.squeeze(0).cpu().numpy(), logp.item()

    def get_value(self, s):
        s = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.ac.critic(s).item()

    # —— PPO‑Clip 更新 ——
    def train(self, buffer, total_steps:int):
        batch = buffer.get_training_data()

        # to device
        for k in ['s','a','a_logprob','adv','v_target','active']:
            dtype = torch.float32
            batch[k] = batch[k].to(self.device, dtype=dtype)

        metrics = {k:[] for k in ['loss/policy','loss/value','loss/entropy',
                                  'loss/clip_frac','loss/approx_kl']}

        for _ in range(self.K):
            for idx in BatchSampler(SequentialSampler(range(self.M)), self.mini_M, False):
                self.reset_rnn()

                s   = batch['s'][idx]
                a   = batch['a'][idx]
                adv = batch['adv'][idx]
                vt  = batch['v_target'][idx]
                old_logp = batch['a_logprob'][idx]
                active_m = batch['active'][idx]

                out = self.ac.actor(s)
                alpha, beta = torch.chunk(nn.Softplus()(out)+1.0, 2, dim=-1)
                dist = Beta(alpha, beta)
                new_logp = dist.log_prob(a).sum(-1)
                ratio = torch.exp(new_logp - old_logp)

                # policy loss
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv
                pol_loss = -torch.min(surr1, surr2)
                pol_loss = (pol_loss*active_m).sum()/active_m.sum()

                # value loss
                v      = self.ac.critic(s).squeeze(-1)
                v_loss = ((v - vt)**2 * active_m).sum()/active_m.sum()

                entropy = dist.entropy().sum(-1)
                entropy = (entropy*active_m).sum()/active_m.sum()

                loss = pol_loss + 0.5*v_loss - self.entropy_coef*entropy
                self.opt.zero_grad()
                loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()

                with torch.no_grad():
                    clip_frac = ((ratio>1+self.clip)|(ratio<1-self.clip)).float()
                    clip_frac = (clip_frac*active_m).sum()/active_m.sum()
                    approx_kl = (old_logp - new_logp)
                    approx_kl = (approx_kl*active_m).sum()/active_m.sum()

                metrics['loss/policy'].append(pol_loss.item())
                metrics['loss/value'].append(v_loss.item())
                metrics['loss/entropy'].append(entropy.item())
                metrics['loss/clip_frac'].append(clip_frac.item())
                metrics['loss/approx_kl'].append(approx_kl.item())

        # learning‑rate decay
        if self.use_lr_decay:
            lr_now = 0.9*self.lr0*(1 - total_steps/self.max_steps) + 0.1*self.lr0
            for g in self.opt.param_groups: g['lr'] = lr_now

        # mean over mini‑batches
        return {k:statistics.mean(v) for k,v in metrics.items()}