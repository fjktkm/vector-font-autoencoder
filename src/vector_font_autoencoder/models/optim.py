from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class WarmupCosineAnnealingLR(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        training_steps: int,
        warmup_ratio: float = 0.06,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        warmup_steps = int(training_steps * warmup_ratio)
        cosine_steps = training_steps - warmup_steps

        warmup = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=eta_min,
        )

        super().__init__(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
            last_epoch=last_epoch,
        )
