"""Hover information for matplotlib volume and contour plots.

The manager attaches to a Figure's `motion_notify_event` and, for any
registered axes, samples the underlying 2D arrays at the cursor position to
build a single status-bar string of the form

    x=..., y=...; <value_label>=<v>; ROI: <name> (<id>)

Each axes can have several registered layers (one volume + one label map, for
instance), so an atlas overlaid on a power-Doppler scan shows both the data
value and the ROI name simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.backend_bases import MouseEvent

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import Event
    from matplotlib.figure import Figure


@dataclass
class _SliceLayer:
    """One sample-able 2D array attached to a single axes."""

    x_coords: np.ndarray
    """Sorted (ascending) physical coordinates of the column dimension."""
    y_coords: np.ndarray
    """Sorted (ascending) physical coordinates of the row dimension."""
    data_2d: np.ndarray
    """The values to sample, shape `(len(y_coords), len(x_coords))`."""
    role: Literal["volume", "labels"]
    """How to interpret samples: as a numeric value or as an integer ROI id."""
    value_label: str = "value"
    """Display name for the numeric value (used only when `role == 'volume'`)."""
    units: str | None = None
    """Units string appended to the numeric value, if any."""


class _RegionHoverManager:
    """Show the data value and/or ROI name under the cursor in the status bar.

    Attributes
    ----------
    roi_labels : dict[int, str]
        The id-to-name mapping. Updated in-place when new label slices are
        registered with additional ids.
    """

    def __init__(self):
        self.roi_labels: dict[int, str] = dict()
        self._ax_layers: dict["Axes", list[_SliceLayer]] = {}
        self._attached = False

    def is_attached(self) -> bool:
        """Return `True` if this hover manager is attached to a figure."""
        return self._attached

    def attach_figure(self, figure: "Figure") -> None:
        """Attach this hover manager to a figure.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to attach to.
        """
        self._cid = figure.canvas.mpl_connect("motion_notify_event", self._on_hover)
        toolbar = figure.canvas.toolbar
        if toolbar is not None and hasattr(toolbar, "_mouse_event_to_message"):
            # Avoid the duplicated "x=..., y=..., [value]" toolbar message
            # so our format_coord output is shown verbatim.
            toolbar._mouse_event_to_message = (  # type: ignore[method-assign]
                _custom_mouse_event_to_message
            )

        self._attached = True

    def register_data_to_axis(
        self,
        ax: "Axes",
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        data_2d: np.ndarray,
        role: Literal["volume", "labels"],
        value_label: str = "value",
        units: str | None = None,
    ) -> None:
        """Attach a 2D slice for hover lookup on `ax`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes the slice is drawn on.
        x_coords : (W,) numpy.ndarray
            Ascending physical coordinates of the column dimension.
        y_coords : (H,) numpy.ndarray
            Ascending physical coordinates of the row dimension.
        data_2d : (H, W) numpy.ndarray
            Numeric values sampled at `(y_coords, x_coords)`.
        role : {"volume", "labels"}
            How to interpret the slice: as a numeric volume or as integer labels.
        value_label : str, default: "value"
            Display name shown next to the sampled value.
        units : str, optional
            Units string appended after the value.
        """
        self._ax_layers.setdefault(ax, []).append(
            _SliceLayer(
                x_coords=np.asarray(x_coords),
                y_coords=np.asarray(y_coords),
                data_2d=np.asarray(data_2d),
                role=role,
                value_label=value_label,
                units=units,
            )
        )

    def _on_hover(self, event: "Event") -> None:
        """Handle a `motion_notify_event` and rewrite the axes' `format_coord`.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            Event delivered by matplotlib's canvas.
        """
        if not isinstance(event, MouseEvent):
            return

        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        layers = self._ax_layers.get(ax)
        if not layers:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        info = f"x={x:.3g}, y={y:.3g}"
        value_part: str | None = None
        roi_part: str | None = None

        # Iterate latest-first so an overlay shadows earlier registrations.
        for layer in reversed(layers):
            sample = _sample(layer, x, y)
            if sample is None:
                continue
            if layer.role == "volume" and value_part is None:
                value_part = _format_value(sample, layer)
            elif layer.role == "labels" and roi_part is None:
                label = int(sample)
                if label != 0:
                    name = self.roi_labels.get(label)
                    if name is not None:
                        roi_part = f"ROI: {name} ({label})"
                    else:
                        roi_part = f"id={label}"

        if value_part is not None:
            info += "; " + value_part
        if roi_part is not None:
            info += "; " + roi_part

        ax.format_coord = lambda x, y, _info=info: _info  # type: ignore[method-assign]


def _sample(layer: _SliceLayer, x: float, y: float) -> float | None:
    """Return the value at `(x, y)` using nearest-coordinate lookup."""
    if x < layer.x_coords.min() or x > layer.x_coords.max():
        return None
    if y < layer.y_coords.min() or y > layer.y_coords.max():
        return None
    j = int(np.argmin(np.abs(layer.x_coords - x)))
    i = int(np.argmin(np.abs(layer.y_coords - y)))
    return float(layer.data_2d[i, j])


def _format_value(value: float, layer: _SliceLayer) -> str:
    """Format a sampled volume value as `<label>=<value>[ <units>]`."""
    if np.isnan(value):
        rendered = "nan"
    elif np.issubdtype(layer.data_2d.dtype, np.integer):
        rendered = f"{int(value)}"
    else:
        rendered = f"{value:.4g}"
    if layer.units:
        return f"{layer.value_label}={rendered} {layer.units}"
    return f"{layer.value_label}={rendered}"


def _custom_mouse_event_to_message(event):
    """Custom `_mouse_event_to_message` returning only `format_coord`'s output.

    Mirrors the matplotlib default at
    https://github.com/matplotlib/matplotlib/blob/v3.9.0/lib/matplotlib/backend_bases.py
    but skips the data-value suffix that would otherwise duplicate our value/ROI info.
    """
    if event.inaxes and event.inaxes.get_navigate():
        try:
            return event.inaxes.format_coord(event.xdata, event.ydata)
        except (ValueError, OverflowError):
            return ""
    return ""


def _normalize_roi_labels(
    roi_labels: dict | None,
) -> dict[int, str]:
    """Coerce a user-provided `roi_labels` dict to `{int: str}`."""
    if not roi_labels:
        return {}
    out: dict[int, str] = {}
    for k, v in roi_labels.items():
        try:
            out[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return out
