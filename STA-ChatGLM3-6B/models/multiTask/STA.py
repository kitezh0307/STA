import torch
import torch.nn as nn
import torch.nn.functional as F


from ..subNets.Textmodel import Language_model

__all__ = ['STA']

class STA(nn.Module):
    def __init__(self, args):
        super(STA, self).__init__()
        self.args = args

        self.LLM = Language_model(args)
        for p in self.LLM.parameters():
            p.requires_grad = False

        text_in, audio_in, video_in = args.feature_dims[:]
        hidden_dim = 256

        self.audio_TFE = MultiTFE(audio_in, hidden_dim)
        self.video_TFE = MultiTFE(video_in, hidden_dim)

        self.ts = TemS(max_sh=3)

        self.cross_attn_audio2video = CA(hidden_dim)
        self.cross_attn_video2audio = CA(hidden_dim)

        self.proj_fusion = nn.Linear(hidden_dim, text_in)
        self.compress = nn.Linear(args.compress_dim, args.tokens)

        self.tri_nce = TriModalInfoNCE(
            audio_dim=hidden_dim,
            video_dim=hidden_dim,
            text_dim=text_in,
            proj_dim=256,
            temperature=0.07,
            use_ln=True
        )


    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        text_emb = self.LLM.text_embedding(text[:, 0, :].long())
        audio_feat = self.audio_TFE(audio)
        video_feat = self.video_TFE(video)

        audio_feat, ts_loss = self.ts(audio_feat, video_feat)

        audio_feat = self.cross_attn_audio2video(audio_feat, video_feat)
        video_feat = self.cross_attn_video2audio(video_feat, audio_feat)

        add_fusion = self.proj_fusion(torch.cat([audio_feat, video_feat], dim=1))
        add_fusion = add_fusion.permute(0, 2, 1)
        add_fusion = self.compress(add_fusion)
        fusion = add_fusion.permute(0, 2, 1)


        LLM_input = torch.cat([fusion, text_emb], dim=1)

        LLM_output = self.LLM(LLM_input, labels)

        contrastive_loss = self.tri_nce(audio_feat, video_feat, text_emb)

        total_loss = LLM_output.loss + self.args.contrastive_weight * contrastive_loss + self.args.s_weight * ts_loss


        return {
            'Loss': total_loss,
        }

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text

        text_emb = self.LLM.text_embedding(text[:, 0, :].long())

        audio_feat = self.audio_TFE(audio)
        video_feat = self.video_TFE(video)

        audio_feat = self.cross_attn_audio2video(audio_feat, video_feat)
        video_feat = self.cross_attn_video2audio(video_feat, audio_feat)

        add_fusion = self.proj_fusion(torch.cat([audio_feat, video_feat], dim=1))
        add_fusion = add_fusion.permute(0, 2, 1)
        add_fusion = self.compress(add_fusion)
        fusion = add_fusion.permute(0, 2, 1)

        LLM_input = torch.cat([fusion, text_emb], dim=1)

        return self.LLM.generate(LLM_input)


class TriModalInfoNCE(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim,
                 proj_dim=256, temperature=0.07, use_ln=True):
        super().__init__()
        self.proj_a = nn.Linear(audio_dim, proj_dim)
        self.proj_v = nn.Linear(video_dim, proj_dim)
        self.proj_t = nn.Linear(text_dim, proj_dim)

        self.use_ln = use_ln
        if use_ln:
            self.ln_a = nn.LayerNorm(proj_dim)
            self.ln_v = nn.LayerNorm(proj_dim)
            self.ln_t = nn.LayerNorm(proj_dim)

        self.temperature = temperature

    def forward(self, audio_feat, video_feat, text_feat):
        a = audio_feat.mean(dim=1)
        v = video_feat.mean(dim=1)
        t = text_feat[:, 0, :]

        a = self.proj_a(a)
        v = self.proj_v(v)
        t = self.proj_t(t)

        if self.use_ln:
            a = self.ln_a(a)
            v = self.ln_v(v)
            t = self.ln_t(t)

        a = F.normalize(a, dim=-1)
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        loss_av = self._infonce(a, v)
        loss_at = self._infonce(a, t)
        loss_vt = self._infonce(v, t)

        return (loss_av + loss_at + loss_vt) / 3.0

    def _infonce(self, x, y):
        sim = torch.matmul(x, y.t()) / self.temperature
        labels = torch.arange(x.size(0), device=x.device)
        loss_i = F.cross_entropy(sim, labels)
        loss_j = F.cross_entropy(sim.t(), labels)
        return (loss_i + loss_j) / 2.0

class MultiTFE(nn.Module):
    def __init__(self, input_dim, hidden_dim, skip_step=2, kernel_size=3):
        super().__init__()
        self.skip_step = skip_step
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.res_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)

        skip_x = x[:, :, ::self.skip_step]
        skip_x = self.res_conv(skip_x)

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.interpolate(out, size=skip_x.size(2), mode='linear', align_corners=False)
        out = out + skip_x

        return out.transpose(1, 2)

class CA(nn.Module):
    def __init__(self, dim, num_heads=2):
        super(CA, self).__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        B, Tq, C = query.size()
        Tkv = key_value.size(1)
        H = self.num_heads
        d = C // H

        Q = self.q_proj(query).view(B, Tq, H, d).transpose(1, 2)
        K = self.k_proj(key_value).view(B, Tkv, H, d).transpose(1, 2)
        V = self.v_proj(key_value).view(B, Tkv, H, d).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Tq, C)
        out = self.out_proj(out)
        return out

class TemS(nn.Module):
    def __init__(self, hidden_dim=256, max_sh=3):
        super().__init__()
        self.max_s = max_sh
        self.sp = nn.Linear(hidden_dim, 1)

    def forward(self, audio_feat, video_feat):
        """
        audio_feat: [B, Ta, F]
        video_feat: [B, Tv, F]
        """
        B, T, C = audio_feat.shape

        audio_global = audio_feat.mean(dim=1)  # [B, C]
        delta_t_pred = torch.tanh(self.sp(audio_global)) * self.max_s  # [B, 1]

        t = torch.arange(T, device=audio_feat.device).float().view(1, T, 1)  # [1, T, 1]
        t_s = t + delta_t_pred.view(B, 1, 1)  # [B, T, 1]

        t0 = t_s.floor().long().clamp(0, T-1)
        t1 = (t0 + 1).clamp(0, T-1)
        alpha = (t_s - t0.float())

        batch_idx = torch.arange(B, device=audio_feat.device).unsqueeze(1)
        a_s = audio_feat[batch_idx, t0.squeeze(-1)] * (1 - alpha) + \
                    audio_feat[batch_idx, t1.squeeze(-1)] * alpha

        video_global = video_feat.mean(dim=1)
        a_s_global = a_s.mean(dim=1)
        s_loss = F.mse_loss(a_s_global, video_global)

        return a_s, s_loss