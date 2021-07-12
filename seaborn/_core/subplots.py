from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from seaborn._core.rules import categorical_order

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from seaborn._core.data import PlotData


class Subplots:

    def __init__(
        # TODO defined TypedDict types for these specs
        self,
        subplot_spec,
        facet_spec,
        pair_spec,
        data: PlotData,
    ):

        self.subplot_spec = subplot_spec.copy()
        self.facet_spec = facet_spec.copy()
        self.pair_spec = pair_spec.copy()

        self._check_dimension_uniqueness(data)
        self._determine_grid_dimensions(data)
        self._handle_wrapping()
        self._determine_axis_sharing()

    def _check_dimension_uniqueness(self, data: PlotData) -> None:
        """Reject specs that pair and facet on (or wrap to) same figure dimension."""
        collisions = {"x": ["columns", "rows"], "y": ["rows", "columns"]}
        for pair_axis, (multi_dim, wrap_dim) in collisions.items():
            if self.facet_spec.get("wrap") and "col" in data and "row" in data:
                err = "Cannot wrap facets when specifying both `col` and `row`."
            elif (
                self.pair_spec.get("wrap")
                and self.pair_spec.get("cartesian", True)
                and len(self.pair_spec.get("x", [])) > 1
                and len(self.pair_spec.get("y", [])) > 1
            ):
                err = "Cannot wrap subplots when pairing on both `x` and `y`."
            elif pair_axis not in self.pair_spec:
                continue
            elif multi_dim[:3] in data:
                err = f"Cannot facet the {multi_dim} while pairing on {pair_axis}."
            elif wrap_dim[:3] in data and self.facet_spec.get("wrap"):
                err = f"Cannot wrap the {wrap_dim} while pairing on {pair_axis}."
            elif wrap_dim[:3] in data and self.pair_spec.get("wrap"):
                err = f"Cannot wrap the {multi_dim} while faceting the {wrap_dim}."
            else:
                continue

            raise RuntimeError(err)  # TODO what err class? Define PlotSpecError?

    def _determine_grid_dimensions(self, data: PlotData):

        self.grid_dimensions = {}
        for dim, axis in zip(["col", "row"], ["x", "y"]):

            if dim in data:
                self.grid_dimensions[dim] = categorical_order(
                    data.frame[dim], self.facet_spec.get(f"{dim}_order"),
                )
            elif axis in self.pair_spec:
                self.grid_dimensions[dim] = [None for _ in self.pair_spec[axis]]
            else:
                self.grid_dimensions[dim] = [None]

            self.subplot_spec[f"n{dim}s"] = len(self.grid_dimensions[dim])

        if not self.pair_spec.get("cartesian", True):
            self.subplot_spec["nrows"] = 1

        self.n_subplots = self.subplot_spec["ncols"] * self.subplot_spec["nrows"]

    def _handle_wrapping(self) -> None:

        self.wrap = wrap = self.facet_spec.get("wrap") or self.pair_spec.get("wrap")
        if not wrap:
            return

        wrap_dim = "row" if self.subplot_spec["nrows"] > 1 else "col"
        flow_dim = {"row": "col", "col": "row"}[wrap_dim]
        n_subplots = self.subplot_spec[f"n{wrap_dim}s"]
        flow = int(np.ceil(n_subplots / wrap))

        if wrap < self.subplot_spec[f"n{wrap_dim}s"]:
            self.subplot_spec[f"n{wrap_dim}s"] = wrap
        self.subplot_spec[f"n{flow_dim}s"] = flow
        self.n_subplots = n_subplots
        self.wrap_dim = wrap_dim

    def _determine_axis_sharing(self) -> None:

        axis_to_dim = {"x": "col", "y": "row"}
        for axis in "xy":
            key = f"share{axis}"
            if key in self.subplot_spec:
                # Always use user-specified value, if present
                val = self.subplot_spec[key]
            else:
                if axis in self.pair_spec:
                    # Paired axes are shared along one dimension by default
                    if self.wrap in [None, 1] and self.pair_spec.get("cartesian", True):
                        val = axis_to_dim[axis]
                    else:
                        val = False
                else:
                    # This will pick up faceted plots, as well as single subplot
                    # figures, where the value doesn't really matter
                    val = True
            self.subplot_spec[key] = val

    def init_figure(self, pyplot: bool):  # TODO figsize param or figure_kws dict?

        if pyplot:
            figure = plt.figure()
        else:
            figure = mpl.figure.Figure()
        self._figure = figure

        axs = figure.subplots(**self.subplot_spec, squeeze=False)

        if self.wrap:
            axs_flat = axs.ravel({"col": "C", "row": "F"}[self.wrap_dim])
            axs, extra = np.split(axs_flat, [self.n_subplots])
            for ax in extra:
                ax.remove()
            if self.wrap_dim == "col":
                axs = axs[np.newaxis, :]
            else:
                axs = axs[:, np.newaxis]

        if not self.pair_spec.get("cartesian", True):
            indices = np.arange(self.n_subplots)
            iter_axs = zip(zip(indices, indices), axs.flat)
        else:
            iter_axs = np.ndenumerate(axs)

        self._subplot_list = []
        for (i, j), ax in iter_axs:

            info = {"ax": ax}

            nrows, ncols = self.subplot_spec["nrows"], self.subplot_spec["ncols"]
            if not self.wrap:
                info["left"] = j % ncols == 0
                info["right"] = (j + 1) % ncols == 0
                info["top"] = i == 0
                info["bottom"] = i == nrows - 1
            elif self.wrap_dim == "col":
                info["left"] = j % ncols == 0
                info["right"] = ((j + 1) % ncols == 0) or ((j + 1) == self.n_subplots)
                info["top"] = j < ncols
                info["bottom"] = j >= (self.n_subplots - ncols)
            elif self.wrap_dim == "row":
                info["left"] = i < nrows
                info["right"] = i >= self.n_subplots - nrows
                info["top"] = i % nrows == 0
                info["bottom"] = ((i + 1) % nrows == 0) or ((i + 1) == self.n_subplots)

            if not self.pair_spec["cartesian"]:
                info["top"] = j < ncols
                info["bottom"] = j >= self.n_subplots - ncols

            for dim in ["row", "col"]:
                idx = {"row": i, "col": j}[dim]
                info[dim] = self.grid_dimensions[dim][idx]

            for axis in "xy":

                idx = {"x": j, "y": i}[axis]
                if axis in self.pair_spec:
                    key = f"{axis}{idx}"
                else:
                    key = axis
                info[axis] = key

            self._subplot_list.append(info)

        # TODO moving the parts of existing code that depend on data or Plot, so they
        # need to be implemented in a separate method (or in Plot._setup_figure)

    def __iter__(self):
        pass  # TODO use this for looping over each subplot?
