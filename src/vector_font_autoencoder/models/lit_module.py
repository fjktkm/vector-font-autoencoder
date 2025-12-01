from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import AdamW

from vector_font_autoencoder.models.module import (
    FontAutoencoder,
    ReconstructionLoss,
)
from vector_font_autoencoder.models.optim import WarmupCosineAnnealingLR
from vector_font_autoencoder.utils.visualize import plot_glyph_tensor


def create_masks(
    ops: Tensor,
    pad_id: int = 0,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    b, p, _ = ops.shape
    device = ops.device

    src_pad = ops[..., 0].eq(pad_id)
    src_padding_mask = torch.cat(
        [
            torch.zeros(b, 1, dtype=torch.bool, device=device),
            src_pad,
        ],
        dim=1,
    )

    tgt_padding_mask = torch.cat(
        [
            torch.zeros(b, 2, dtype=torch.bool, device=device),
            src_pad[:, :-1],
        ],
        dim=1,
    )

    src_mask = None
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
        p + 1,
        device=device,
        dtype=torch.bool,
    )

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def combine_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    types_list, coords_list = zip(*batch, strict=True)

    sizes = [t.size(0) for t in types_list]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    total_samples = int(offsets[-1])

    max_seq_len = max(x.size(1) for x in types_list)
    types_ref, coords_ref = types_list[0], coords_list[0]

    combined_types = types_ref.new_zeros(total_samples, max_seq_len, types_ref.size(2))
    combined_coords = coords_ref.new_zeros(
        total_samples,
        max_seq_len,
        coords_ref.size(2),
        coords_ref.size(3),
    )

    for i, (types, coords) in enumerate(zip(types_list, coords_list, strict=True)):
        start, end = offsets[i], offsets[i + 1]
        seq_len = types.size(1)
        combined_types[start:end, :seq_len] = types
        combined_coords[start:end, :seq_len] = coords

    return combined_types, combined_coords


class LitFontAutoencoder(LightningModule):
    def __init__(  # noqa: PLR0913
        self,
        patch_size: int = 32,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 3072,
        num_layers: int = 6,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FontAutoencoder(
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        )
        self.loss = ReconstructionLoss()
        self.lr = lr

    def forward(
        self,
        ops: Tensor,
        coords: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(ops)

        return self.model(
            ops=ops,
            coords=coords,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    def training_step(
        self,
        batch: list[tuple[Tensor, Tensor]],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        ops, coords = combine_fn(batch)
        ops_logits, coords_pred = self(ops, coords)

        loss = self.loss(ops_logits, coords_pred, ops, coords)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )
        return loss

    def validation_step(
        self,
        batch: list[tuple[Tensor, Tensor]],
        batch_idx: int,
    ) -> None:
        ops, coords = combine_fn(batch)
        ops_logits, coords_pred = self(ops, coords)

        if self.trainer.global_rank == 0:
            lg = self.trainer.logger
            if isinstance(lg, (list, tuple)):
                lg = lg[0]
            log_dir = Path(lg.log_dir)  # type: ignore  # noqa: PGH003
            log_dir.mkdir(parents=True, exist_ok=True)
            ops0_pred = ops_logits[0].argmax(dim=-1).view(-1)
            coords0_pred = coords_pred[0].view(-1, coords_pred.size(-1))
            out_path = log_dir / f"preview_batch{batch_idx:02d}.pdf"
            plot_glyph_tensor(ops0_pred, coords0_pred, out_path=out_path)

        loss = self.loss(ops_logits, coords_pred, ops, coords)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )

    def test_step(
        self,
        batch: list[tuple[Tensor, Tensor]],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        ops, coords = combine_fn(batch)
        ops_logits, coords_pred = self(ops, coords)

        loss = self.loss(ops_logits, coords_pred, ops, coords)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=torch.distributed.is_initialized(),
        )

    def predict_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        ops, coords = batch

        b = ops.size(0)
        patches = self.model.embed(ops, coords)
        cls = self.model.cls_token.expand(b, 1, -1)
        src = torch.cat([cls, patches], dim=1)
        memory = self.model.encoder(
            self.model.pos_enc(src),
            mask=None,
            src_key_padding_mask=None,
        )
        summary = memory[:, 0:1, :]
        bos = self.model.bos_token.expand(b, 1, -1)
        tgt = torch.cat([summary, bos], dim=1)

        generated_ops = []
        generated_coords = []

        for _ in range(16):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1),
                device=tgt.device,
                dtype=torch.bool,
            )

            out = self.model.decoder(
                self.model.pos_enc(tgt),
                mask=tgt_mask,
                src_key_padding_mask=None,
            )

            pred_patch = out[:, -1:, :]
            ops_logits, coords_pred = self.model.unembed(pred_patch)

            next_ops = ops_logits.argmax(dim=-1)
            next_patch = self.model.embed(next_ops, coords_pred)

            tgt = torch.cat([tgt, next_patch], dim=1)
            generated_ops.append(next_ops)
            generated_coords.append(coords_pred)

        ops_pred = torch.cat(generated_ops, dim=1)
        coords_pred = torch.cat(generated_coords, dim=1)

        lg = self.trainer.logger
        if isinstance(lg, (list, tuple)):
            lg = lg[0]
        log_dir = Path(lg.log_dir)  # type: ignore  # noqa: PGH003
        log_dir.mkdir(parents=True, exist_ok=True)

        rank = self.trainer.global_rank

        for i in range(b):
            out_dir = (
                log_dir
                / "predict"
                / f"rank{rank}"
                / f"batch{batch_idx}"
                / f"dataloader{dataloader_idx}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            gt_path = out_dir / f"sample{i}_gt.pdf"
            pred_path = out_dir / f"sample{i}_pred.pdf"
            plot_glyph_tensor(
                ops[i].view(-1),
                coords[i].view(-1, coords.size(-1)),
                out_path=gt_path,
            )
            plot_glyph_tensor(
                ops_pred[i].view(-1),
                coords_pred[i].view(-1, coords_pred.size(-1)),
                out_path=pred_path,
            )

    def on_predict_end(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        if not hasattr(datamodule, "interpolate_dataloader"):
            return

        loader = datamodule.interpolate_dataloader()

        device = self.device

        for batch_idx, (sans_batch, serif_batch) in enumerate(loader):
            sans_ops, sans_coords = sans_batch
            serif_ops, serif_coords = serif_batch

            sans_ops = sans_ops.to(device)
            sans_coords = sans_coords.to(device)
            serif_ops = serif_ops.to(device)
            serif_coords = serif_coords.to(device)

            def encode(ops: Tensor, coords: Tensor) -> Tensor:
                b = ops.size(0)
                patches = self.model.embed(ops, coords)
                cls = self.model.cls_token.expand(b, 1, -1)
                src = torch.cat([cls, patches], dim=1)
                memory = self.model.encoder(
                    self.model.pos_enc(src),
                    mask=None,
                    src_key_padding_mask=None,
                )
                return memory[:, 0:1, :]

            z_sans = encode(sans_ops, sans_coords)
            z_serif = encode(serif_ops, serif_coords)

            lg = self.trainer.logger
            if isinstance(lg, (list, tuple)):
                lg = lg[0]
            log_dir = Path(lg.log_dir)  # type: ignore  # noqa: PGH003
            log_dir.mkdir(parents=True, exist_ok=True)
            rank = self.trainer.global_rank

            for i in range(sans_ops.size(0)):
                out_dir = (
                    log_dir
                    / "interpolate"
                    / f"rank{rank}"
                    / f"batch{batch_idx}"
                    / f"sample{i}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_glyph_tensor(
                    sans_ops[i].view(-1),
                    sans_coords[i].view(-1, sans_coords.size(-1)),
                    out_path=out_dir / "step0_gt.pdf",
                )
                plot_glyph_tensor(
                    serif_ops[i].view(-1),
                    serif_coords[i].view(-1, serif_coords.size(-1)),
                    out_path=out_dir / "step1_gt.pdf",
                )

            n_steps = 11
            for t in torch.linspace(0, 1, n_steps, device=device):
                z_interp = (1 - t) * z_sans + t * z_serif

                bos = self.model.bos_token.expand(z_interp.size(0), 1, -1)
                tgt = torch.cat([z_interp, bos], dim=1)

                generated_ops = []
                generated_coords = []

                for _ in range(16):
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        tgt.size(1),
                        device=device,
                        dtype=torch.bool,
                    )

                    out = self.model.decoder(
                        self.model.pos_enc(tgt),
                        mask=tgt_mask,
                        src_key_padding_mask=None,
                    )

                    pred_patch = out[:, -1:, :]
                    ops_logits, coords_pred = self.model.unembed(pred_patch)

                    next_ops = ops_logits.argmax(dim=-1)
                    next_patch = self.model.embed(next_ops, coords_pred)

                    tgt = torch.cat([tgt, next_patch], dim=1)
                    generated_ops.append(next_ops)
                    generated_coords.append(coords_pred)

                ops_pred = torch.cat(generated_ops, dim=1)
                coords_pred = torch.cat(generated_coords, dim=1)

                for i in range(ops_pred.size(0)):
                    ops_i = ops_pred[i].view(-1)
                    coords_i = coords_pred[i].view(-1, coords_pred.size(-1))
                    out_path = (
                        log_dir
                        / "interpolate"
                        / f"rank{rank}"
                        / f"batch{batch_idx}"
                        / f"sample{i}"
                        / f"step{t:.1f}.pdf"
                    )
                    plot_glyph_tensor(ops_i, coords_i, out_path=out_path)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        no_decay = ["bias", "LayerNorm.weight"]

        params = list(self.named_parameters())
        decay_params = [
            p
            for n, p in params
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ]
        nodecay_params = [
            p for n, p in params if p.requires_grad and any(nd in n for nd in no_decay)
        ]

        optimizer = AdamW(
            [
                {"params": decay_params},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        total_steps = max(
            1,
            int(getattr(self.trainer, "estimated_stepping_batches", 0)),
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
