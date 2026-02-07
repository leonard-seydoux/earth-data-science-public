import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from scipy.ndimage import gaussian_filter

from matplotlib.path import Path
from matplotlib.patches import PathPatch

from shapely.geometry import box
from shapely.ops import unary_union


def cycler_cmap(n):
    """Create a ListedColormap from the axis color cycle."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return mcolors.ListedColormap(colors[:n])


def brighter(color, amount=0.5):
    """Return a brighter version of the given color."""
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    new_c = c + (white - c) * amount
    return mcolors.to_hex(new_c)


def get_boundaries(model, ax, n=100):
    """Get decision boundary from classification model."""
    x1 = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], n)
    x2 = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], n)
    x1, x2 = np.meshgrid(x1, x2)
    z = model.predict(np.c_[x1.ravel(), x2.ravel()])
    return x1, x2, z.reshape(x1.shape)


def xkcd_style():
    """Set matplotlib to XKCD style with customizations."""
    plt.xkcd()
    plt.rcParams["path.effects"] = [patheffects.withStroke(linewidth=0)]
    plt.style.use("contents/scripts/matplotlibrc")


def plot_samples(ax, x, **kwargs):
    """Plot classes with different colors and markers."""
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ls", "")
    kwargs.setdefault("mec", "k")
    kwargs.setdefault("ms", 5)
    ax.plot(*x.T[:2], **kwargs)


def samples(ax, X, y=None, **kwargs):
    """Plot labeled samples."""
    # Default scatter kwargs
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ls", "")
    kwargs.setdefault("mec", "k")
    kwargs.setdefault("ms", 5)

    # Plot each class separately
    y = y if y is not None else np.zeros(X.shape[0], dtype=int)
    categories = sorted(set(y))
    for category in categories:
        ax.plot(*X[y == category].T[:2], label=f"Class {category}", **kwargs)


def colorized_boundaries(ax, model):
    """Plot class boundaries for classification model."""
    x1, x2, boundaries = get_boundaries(model, ax, n=500)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors[: len(np.unique(boundaries))]
    cmap = mcolors.ListedColormap(colors)
    ax.contourf(x1, x2, boundaries, cmap=cmap, alpha=0.3)


def draw_boundaries(ax, model, n=500, **kwargs):
    """Plot cluster boundaries for KMeans model."""
    # Get boundaries
    x1, x2, boundaries = get_boundaries(model, ax, n=n)
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # Default contour kwargs
    kwargs.setdefault("levels", np.arange(-0.5, len(np.unique(boundaries)), 1))
    kwargs.setdefault("linewidths", 1)
    kwargs.setdefault("colors", brighter("C5", -1))

    # Draw contours
    return ax.contour(x1, x2, boundaries, **kwargs)


def _ring_to_verts_codes(ring):
    ring = np.asarray(ring.coords, dtype=float)
    verts = ring.copy()
    codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    return verts, codes


def _shapely_to_pathpatch(geom, **patch_kwargs):
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    verts_all, codes_all = [], []

    for poly in polys:
        v, c = _ring_to_verts_codes(poly.exterior)
        verts_all.append(v)
        codes_all.append(c)

        for hole in poly.interiors:
            v, c = _ring_to_verts_codes(hole)
            verts_all.append(v)
            codes_all.append(c)

    verts = np.vstack(verts_all)
    codes = np.concatenate(codes_all)

    return PathPatch(Path(verts, codes), **patch_kwargs)


def space_invaders(ax, center=(0.5, 0.5), size=0.4, **kwargs):
    """
    Draw the pixel invader as ONE merged patch (with holes preserved).
    Requires shapely.
    """
    pixels = np.array(
        [
            (4, 8),
            (8, 8),
            (5, 7),
            (7, 7),
            (3, 6),
            (4, 6),
            (5, 6),
            (6, 6),
            (7, 6),
            (8, 6),
            (9, 6),
            (2, 5),
            (3, 5),
            (5, 5),
            (6, 5),
            (7, 5),
            (9, 5),
            (10, 5),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (9, 4),
            (10, 4),
            (2, 3),
            (4, 3),
            (8, 3),
            (10, 3),
            (4, 2),
            (5, 2),
            (7, 2),
            (8, 2),
        ],
        dtype=float,
    )

    # Each pixel is a unit square centered at (x,y): [x-0.5,x+0.5]×[y-0.5,y+0.5]
    ll = pixels - 0.5

    # Normalize to a square bbox then map to axis coords (keeps aspect)
    min_xy = ll.min(axis=0)
    max_xy = (ll + 1.0).max(axis=0)
    wh = max_xy - min_xy
    s = float(max(wh[0], wh[1]))
    scale = size / s
    offset = np.array(center) - (min_xy + np.array([s, s]) / 2.0) * scale

    # Build squares in final coords and union them
    squares = [
        box(
            x0 * scale + offset[0],
            y0 * scale + offset[1],
            (x0 + 1.0) * scale + offset[0],
            (y0 + 1.0) * scale + offset[1],
        )
        for x0, y0 in ll
    ]
    merged = unary_union(squares)

    kwargs.setdefault("facecolor", "C0")
    kwargs.setdefault("edgecolor", "k")
    kwargs.setdefault("linewidth", 1.5)
    kwargs.setdefault("joinstyle", "round")
    patch = _shapely_to_pathpatch(merged, **kwargs)
    ax.add_patch(patch)
    return patch
