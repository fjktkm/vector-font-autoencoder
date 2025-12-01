from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from torchfont.io.pens import TYPE_TO_IDX

from vector_font_disentanglement.utils.pen import PlotPen

palette = sns.color_palette("deep")
bluegray = palette[0]


def plot_glyph_tensor(
    ops: Tensor,
    coords: Tensor,
    out_path: str | Path,
    figsize: tuple[int, int] = (6, 6),
) -> None:
    sns.set_theme(style="ticks")
    ops_idx = ops.detach().cpu().view(-1).tolist()
    pts = coords.detach().cpu().numpy()
    pen = PlotPen()

    for i, oi in enumerate(ops_idx):
        if oi == TYPE_TO_IDX["eos"]:
            break
        if oi == TYPE_TO_IDX["pad"]:
            continue
        x1, y1, x2, y2, x3, y3 = pts[i]
        if oi == TYPE_TO_IDX["moveTo"]:
            pen.moveTo((x3, y3))
        elif oi == TYPE_TO_IDX["lineTo"]:
            pen.lineTo((x3, y3))
        elif oi == TYPE_TO_IDX["curveTo"]:
            pen.curveTo((x1, y1), (x2, y2), (x3, y3))
        elif oi == TYPE_TO_IDX["closePath"]:
            pen.closePath()

    fig = pen.draw(figsize=figsize)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
