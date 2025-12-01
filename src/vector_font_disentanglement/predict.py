import logging
import warnings
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from vector_font_disentanglement.datasets.lit_data_module import LitGoogleFonts
from vector_font_disentanglement.models.lit_module import LitFontAutoencoder

logging.getLogger("fontTools").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_float32_matmul_precision("high")


def main() -> None:
    patch_size = 32
    version = 6
    ckpt_dir = Path(f"lightning_logs/version_{version}/checkpoints")

    ckpt_files = sorted(ckpt_dir.glob("*.ckpt"))
    ckpt_path = str(ckpt_files[-1])
    print(f"Loading checkpoint: {ckpt_path}")

    data_module = LitGoogleFonts(
        patch_size=patch_size,
        ref="52cf487c1ed08e32f6e247b3e9ebcb069f42d30a",
    )
    model = LitFontAutoencoder.load_from_checkpoint(ckpt_path)

    tb_logger = TensorBoardLogger(".")
    csv_logger = CSVLogger(".", version=tb_logger.version)

    trainer = pl.Trainer(
        precision="16-mixed" if torch.cuda.is_available() else None,
        logger=[tb_logger, csv_logger],
        use_distributed_sampler=False,
    )

    trainer.predict(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
