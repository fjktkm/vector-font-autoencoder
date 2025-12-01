import logging
import warnings

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from vector_font_disentanglement.datasets.lit_data_module import LitGoogleFonts
from vector_font_disentanglement.models.lit_module import LitFontAutoencoder

logging.getLogger("fontTools").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_float32_matmul_precision("high")


def main() -> None:
    patch_size = 32

    data_module = LitGoogleFonts(patch_size=patch_size)

    model = LitFontAutoencoder(patch_size=patch_size)

    tb_logger = TensorBoardLogger(".")
    csv_logger = CSVLogger(".", version=tb_logger.version)
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="vector-font-disentanglement-{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    summary = ModelSummary(max_depth=2)
    profiler = SimpleProfiler(filename="profiler.txt")

    trainer = pl.Trainer(
        max_epochs=4,
        precision="16-mixed" if torch.cuda.is_available() else None,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint, lr_monitor, summary],
        val_check_interval=1 / 8,
        profiler=profiler,
        use_distributed_sampler=False,
    )

    trainer.fit(model, datamodule=data_module)

    trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()
