import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.pens.basePen import BasePen
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

type Point = tuple[float, float]


class PlotPen(BasePen):
    def __init__(self) -> None:
        super().__init__(glyphSet=None)
        self.vertices = []
        self.codes = []
        self._subpath_start = None

    def _moveTo(self, pt: Point) -> None:  # noqa: N802
        self.vertices.append(pt)
        self.codes.append(MplPath.MOVETO)
        self._subpath_start = pt

    def _lineTo(self, pt: Point) -> None:  # noqa: N802
        if not self.codes or self.codes[-1] == MplPath.CLOSEPOLY:
            self._moveTo(pt)
            return
        self.vertices.append(pt)
        self.codes.append(MplPath.LINETO)

    def _curveToOne(self, pt1: Point, pt2: Point, pt3: Point) -> None:  # noqa: N802
        if not self.codes or self.codes[-1] == MplPath.CLOSEPOLY:
            self._moveTo(pt3)
            return
        self.vertices.extend([pt1, pt2, pt3])
        self.codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

    def _closePath(self) -> None:  # noqa: N802
        if self._subpath_start is not None:
            self.vertices.append(self._subpath_start)
            self.codes.append(MplPath.CLOSEPOLY)
        self._subpath_start = None

    def draw(self, figsize: tuple[int, int] = (6, 6)) -> Figure:
        sns.set_theme(style="ticks")
        palette = sns.color_palette("deep")
        bluegray = palette[0]

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        if self.vertices:
            path = MplPath(self.vertices, self.codes)
            patch = PathPatch(
                path,
                facecolor=(*bluegray, 0.75),
                edgecolor=bluegray,
            )
            ax.add_patch(patch)

        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        return fig
