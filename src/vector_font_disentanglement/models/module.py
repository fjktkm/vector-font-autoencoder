import math
from typing import Annotated, Any, cast

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torchfont.io.pens import (
    CMD_DIM,
    COORD_DIM,
    TYPE_DIM,
    TYPE_TO_IDX,
)


class FontDisentangler(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        num_style_classes: int,
        num_content_classes: int,
        patch_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = PatchEmbedding(d_model, patch_size, dropout)
        self.pos_enc = RotaryEmbedding(d_model, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.bos_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.style_proj = nn.Linear(d_model, d_model)
        self.content_proj = nn.Linear(d_model, d_model)
        self.grl_style = GradientReversal()
        self.grl_content = GradientReversal()
        self.style_discriminator = nn.Linear(d_model, num_style_classes)
        self.content_discriminator = nn.Linear(d_model, num_content_classes)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.unembed = PatchUnembedding(d_model, patch_size)

    def forward(  # noqa: PLR0913
        self,
        ops: Tensor,
        coords: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
        src_padding_mask: Tensor | None,
        tgt_padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b = ops.size(0)
        paches = self.embed(ops, coords)
        cls = self.cls_token.expand(b, 1, -1)
        src = torch.cat([cls, paches], dim=1)
        memory = self.encoder(
            self.pos_enc(src),
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )
        summary = memory[:, 0, :]
        style_feat = self.style_proj(summary)
        content_feat = self.content_proj(summary)
        style_logits = self.style_discriminator(self.grl_style(style_feat))
        content_logits = self.content_discriminator(self.grl_content(content_feat))
        combined = (style_feat + content_feat).unsqueeze(1)
        bos = self.bos_token.expand(b, 1, -1)
        tgt = torch.cat([combined, bos, paches[:, :-1, :]], dim=1)
        output = self.decoder(
            self.pos_enc(tgt),
            mask=tgt_mask,
            src_key_padding_mask=tgt_padding_mask,
        )
        pred = output[:, 1:, :]
        ops_logits, coords_pred = self.unembed(pred)
        return ops_logits, coords_pred, style_logits, content_logits


class FontAutoencoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        patch_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = PatchEmbedding(d_model, patch_size, dropout)
        self.pos_enc = RotaryEmbedding(d_model, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.bos_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.unembed = PatchUnembedding(d_model, patch_size)

    def forward(  # noqa: PLR0913
        self,
        ops: Tensor,
        coords: Tensor,
        src_mask: Tensor | None,
        tgt_mask: Tensor | None,
        src_padding_mask: Tensor | None,
        tgt_padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        b = ops.size(0)
        patches = self.embed(ops, coords)
        cls = self.cls_token.expand(b, 1, -1)
        src = torch.cat([cls, patches], dim=1)
        memory = self.encoder(
            self.pos_enc(src),
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )
        summary = memory[:, 0:1, :]
        bos = self.bos_token.expand(b, 1, -1)
        tgt = torch.cat([summary, bos, patches[:, :-1, :]], dim=1)
        output = self.decoder(
            self.pos_enc(tgt),
            mask=tgt_mask,
            src_key_padding_mask=tgt_padding_mask,
        )
        pred = output[:, 1:, :]
        return self.unembed(pred)


class PatchEmbedding(nn.Module):
    scale: Annotated[Tensor, "buffer"]
    op_eye: Annotated[Tensor, "buffer"]

    def __init__(self, d_model: int, patch_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(CMD_DIM * patch_size, d_model)
        self.dropout = nn.Dropout(dropout)

        scale = torch.tensor(math.sqrt(d_model), dtype=torch.float32)
        self.register_buffer("scale", scale)
        op_eye = torch.eye(TYPE_DIM, dtype=torch.float32)
        self.register_buffer("op_eye", op_eye)

    def forward(self, ops: Tensor, coords: Tensor) -> Tensor:
        batch_size, num_patches, tokens_per_patch = ops.shape
        one_hot_ops = self.op_eye.type_as(coords)[ops]
        one_hot_flat = one_hot_ops.view(
            batch_size,
            num_patches,
            tokens_per_patch * TYPE_DIM,
        )
        coords_flat = coords.view(
            batch_size,
            num_patches,
            tokens_per_patch * COORD_DIM,
        )
        tokens = torch.cat([one_hot_flat, coords_flat], dim=-1)
        x = self.proj(tokens)
        x = x * self.scale.type_as(x)
        return self.dropout(x)


class PatchUnembedding(nn.Module):
    inv_scale: Annotated[Tensor, "buffer"]

    def __init__(self, d_model: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.ops_proj = nn.Linear(d_model, TYPE_DIM * patch_size)
        self.coords_proj = nn.Linear(d_model, COORD_DIM * patch_size)

        inv_scale = torch.tensor(1.0 / math.sqrt(d_model), dtype=torch.float32)
        self.register_buffer("inv_scale", inv_scale)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, num_patches, _ = x.shape
        scaled = x * self.inv_scale.type_as(x)
        ops_logits = self.ops_proj(scaled).view(
            batch_size,
            num_patches,
            self.patch_size,
            TYPE_DIM,
        )
        coords = self.coords_proj(scaled).view(
            batch_size,
            num_patches,
            self.patch_size,
            COORD_DIM,
        )
        return ops_logits, coords


class RotaryEmbedding(nn.Module):
    inv_freq: Annotated[Tensor, "buffer"]

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()

        if dim % 2:
            msg = "dim must be even for RoPE."
            raise ValueError(msg)

        half_dim = dim // 2
        freqs = base ** (-(torch.arange(half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer("inv_freq", freqs)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        half_dim = self.inv_freq.numel()

        pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        ang = torch.einsum("i,j->ij", pos, self.inv_freq)
        sin = ang.sin()[None, :, :, None].type_as(x)
        cos = ang.cos()[None, :, :, None].type_as(x)

        x = x.view(batch, seq_len, half_dim, 2)
        x1, x2 = x[..., :1], x[..., 1:]
        y = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return y.view(batch, seq_len, dim)


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:  # noqa: ANN401, ARG004
        return x

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor]:  # noqa: ANN401, ARG004
        return (-grad_outputs[0],)


class GradientReversal(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return cast("Tensor", _GRLFn.apply(x))


class ReconstructionLoss(nn.Module):
    coord_mask: Annotated[Tensor, "buffer"]

    def __init__(
        self,
        pad_id: int = 0,
        lambda_ops: float = 1.0,
        lambda_coords: float = 500.0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.lambda_ops = lambda_ops
        self.lambda_coords = lambda_coords

        mask = torch.zeros(TYPE_DIM, COORD_DIM, dtype=torch.float32)
        mask[TYPE_TO_IDX["moveTo"], -2:] = 1.0
        mask[TYPE_TO_IDX["lineTo"], -2:] = 1.0
        mask[TYPE_TO_IDX["curveTo"], :] = 1.0
        self.register_buffer("coord_mask", mask)

    def forward(
        self,
        ops_logits: Tensor,
        coords_pred: Tensor,
        ops_tgt: Tensor,
        coords_tgt: Tensor,
    ) -> Tensor:
        ce_sum = f.cross_entropy(
            ops_logits.view(-1, TYPE_DIM),
            ops_tgt.view(-1),
            ignore_index=self.pad_id,
            reduction="sum",
        )
        ce_denom = ops_tgt.ne(self.pad_id).sum().clamp_min(1)
        ce = ce_sum / ce_denom

        se = f.mse_loss(coords_pred, coords_tgt, reduction="none")
        base = self.coord_mask.to(coords_pred.dtype)[ops_tgt]
        valid = ops_tgt.ne(self.pad_id).unsqueeze(-1).type_as(coords_pred)
        mask = base * valid

        mse = (se * mask).sum() / mask.sum().clamp_min(1.0)

        return self.lambda_ops * ce + self.lambda_coords * mse
