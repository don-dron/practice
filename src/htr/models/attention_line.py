"""Строковый энкодер CNN+BiLSTM и LSTM-декодер с аддитивным вниманием; обучение по CE (§4.2)."""

from __future__ import annotations

import torch
from torch import nn

from htr.models.crnn_ctc import ConvBlock


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attn_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        time_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """encoder_outputs [T,B,D], decoder_hidden [B,Hd]; lengths [B]."""
        t_src = encoder_outputs.size(0)
        b = decoder_hidden.size(0)
        e_lin = self.encoder_proj(encoder_outputs)
        q_lin = self.decoder_proj(decoder_hidden).unsqueeze(0).expand(t_src, -1, -1)
        energy = torch.tanh(e_lin + q_lin)
        scores = self.score(energy).squeeze(-1).transpose(0, 1)
        mw = t_src
        ar = torch.arange(mw, device=scores.device).unsqueeze(0).expand(b, -1)
        lng = time_lengths.unsqueeze(1).clamp(min=1, max=mw)
        # -1e9 переполняет float16 в AMP; брать предел типа softmax-инпутов.
        scores = scores.masked_fill(ar >= lng, torch.finfo(scores.dtype).min)
        attn_w = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn_w.unsqueeze(1), encoder_outputs.transpose(0, 1)).squeeze(1)
        return ctx, attn_w


class AttentionLineSeq2Seq(nn.Module):
    """
    char_vocab_without_blank = len(itos)-1 (без CTC-blank).
    Embedding: символы 0..v-1, SOS = v, PAD = v+1.
    Головка: символы 0..v-1 + EOS (индекс v).
    """

    def __init__(
        self,
        in_channels: int,
        char_vocab_without_blank: int,
        lstm_hidden_encoder: int = 256,
        lstm_layers_encoder: int = 2,
        decoder_hidden: int = 256,
        attn_dim: int = 256,
        embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        v = int(char_vocab_without_blank)
        self.char_vocab_without_blank = v
        self.sos_idx_emb = v
        self.pad_idx_emb = v + 1
        emb_rows = v + 2

        stride_h = [(2, 2), (2, 2), (2, 1), (2, 1)]
        ch = [(in_channels, 64), (64, 128), (128, 256), (256, 256)]
        blocks: list[nn.Module] = []
        ci = in_channels
        for (_, co), pt in zip(ch, stride_h):
            blocks.append(ConvBlock(ci, co, pool=pt))
            ci = co

        self.cnn = nn.Sequential(*blocks)

        feat_c = ci

        self.enc_lstm = nn.LSTM(
            input_size=feat_c,
            hidden_size=lstm_hidden_encoder,
            num_layers=lstm_layers_encoder,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if lstm_layers_encoder > 1 else 0.0,
        )

        enc_dim = lstm_hidden_encoder * 2

        self.embedding = nn.Embedding(emb_rows, embed_dim, padding_idx=self.pad_idx_emb)

        self.dec_cell = nn.LSTMCell(embed_dim + enc_dim, decoder_hidden)

        self.attn = AdditiveAttention(enc_dim, decoder_hidden, attn_dim)

        self.eos_logits = v

        self.out_lin = nn.Linear(decoder_hidden, v + 1)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:
        if inputs.shape[2] != 32:

            raise ValueError("ожидается H=32 для AttentionLineSeq2Seq")

        b = inputs.size(0)

        feats_map = self.cnn(inputs)

        feats = feats_map.mean(dim=2).permute(2, 0, 1)

        seq, _ = self.enc_lstm(feats)

        t_src = seq.size(0)

        tlens = torch.full((b,), t_src, device=inputs.device, dtype=torch.long).clamp(1, t_src)

        return seq, tlens

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "AttentionLineSeq2Seq без CTC; используйте compute_loss_ce / greedy_inference.",
        )

    def compute_loss_ce(
        self,
        inputs: torch.Tensor,
        teacher_input_ids: torch.Tensor,
        target_logits_classes: torch.Tensor,
        decoder_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        enc, tlens_enc = self.encode(inputs)
        _, lmax = teacher_input_ids.shape
        device = inputs.device



        b = inputs.size(0)

        h_dec = torch.zeros(b, self.dec_cell.hidden_size, device=device)

        c_dec = torch.zeros_like(h_dec)

        criterion = nn.CrossEntropyLoss(reduction="none")

        accum = inputs.new_zeros(())
        denom = decoder_valid_mask.float().sum().clamp(min=1.0)

        for t in range(lmax):
            if not decoder_valid_mask[:, t].any():
                continue
            emb = self.embedding(teacher_input_ids[:, t])
            ctx, _ = self.attn(enc, h_dec, tlens_enc)

            lstm_in = torch.cat([emb, ctx], dim=-1)
            h_dec, c_dec = self.dec_cell(lstm_in, (h_dec, c_dec))

            logits = self.out_lin(self.drop(h_dec))

            targ = target_logits_classes[:, t]
            step_ce = criterion(logits, targ)
            m = decoder_valid_mask[:, t].float()
            accum = accum + (step_ce * m).sum()

        return accum / denom

    def greedy_inference(self, inputs: torch.Tensor, max_steps: int) -> list[list[int]]:
        enc, tlens_enc = self.encode(inputs)

        bsz = inputs.size(0)

        sequences: list[list[int]] = [[] for _ in range(bsz)]
        finished = torch.zeros(bsz, dtype=torch.bool, device=inputs.device)

        h_dec = torch.zeros(bsz, self.dec_cell.hidden_size, device=inputs.device)

        c_dec = torch.zeros_like(h_dec)

        cur_inp = torch.full((bsz,), self.sos_idx_emb, dtype=torch.long, device=inputs.device)

        for _step in range(max_steps):
            if finished.all():
                break
            emb = self.embedding(cur_inp)
            ctx, _ = self.attn(enc, h_dec, tlens_enc)
            lstm_in = torch.cat([emb, ctx], dim=-1)

            h_dec, c_dec = self.dec_cell(lstm_in, (h_dec, c_dec))

            logits = self.out_lin(h_dec)

            predicted = logits.argmax(dim=-1)

            for bi in range(bsz):
                if finished[bi].item():
                    continue

                tk = predicted[bi].item()

                if tk == self.eos_logits:

                    finished[bi] = True

                elif tk < self.char_vocab_without_blank:

                    sequences[bi].append(tk)

                else:

                    finished[bi] = True

            next_in = predicted.clone()

            for bi in range(bsz):
                if finished[bi]:
                    next_in[bi] = self.pad_idx_emb

                elif predicted[bi].item() == self.eos_logits:

                    next_in[bi] = self.pad_idx_emb

                elif predicted[bi].item() < self.char_vocab_without_blank:

                    next_in[bi] = predicted[bi]

                else:

                    next_in[bi] = self.pad_idx_emb

            cur_inp = next_in.detach()

        return sequences


def build_attention_line(cfg: dict, num_charset_classes: int) -> AttentionLineSeq2Seq:
    v_chars = max(1, int(num_charset_classes) - 1)
    mc = dict(cfg)

    return AttentionLineSeq2Seq(
        in_channels=int(mc.get("in_channels", 1)),
        char_vocab_without_blank=v_chars,
        lstm_hidden_encoder=int(mc.get("lstm_hidden_encoder", mc.get("lstm_hidden", 256))),
        lstm_layers_encoder=int(mc.get("encoder_lstm_layers", mc.get("lstm_layers", 2))),
        decoder_hidden=int(mc.get("decoder_hidden", 256)),
        attn_dim=int(mc.get("attn_dim", 256)),
        embed_dim=int(mc.get("decoder_embed_dim", 256)),
        dropout=float(mc.get("dropout", 0.0)),
    )
