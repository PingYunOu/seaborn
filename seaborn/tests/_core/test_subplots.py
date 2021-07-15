import pytest

from seaborn._core.data import PlotData
from seaborn._core.subplots import Subplots
from seaborn._core.rules import categorical_order


class TestSpecificationChecks:

    def test_both_facets_and_wrap(self, long_df):

        data = PlotData(long_df, dict(col="a", row="b"))
        err = "Cannot wrap facets when specifying both `col` and `row`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {"wrap": 3}, {}, data)

    def test_cartesian_xy_pairing_and_wrap(self, long_df):

        data = PlotData(long_df, {})
        err = "Cannot wrap subplots when pairing on both `x` and `y`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, {"x": ["x", "y"], "y": ["a", "b"], "wrap": 3}, data)

    def test_col_facets_and_x_pairing(self, long_df):

        data = PlotData(long_df, {"col": "a"})
        err = "Cannot facet the columns while pairing on `x`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, {"x": ["x", "y"]}, data)

    def test_wrapped_columns_and_y_pairing(self, long_df):

        data = PlotData(long_df, {"col": "a"})
        err = "Cannot wrap the columns while pairing on `y`."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {"wrap": 2}, {"y": ["x", "y"]}, data)

    def test_wrapped_x_pairing_and_facetd_rows(self, long_df):

        data = PlotData(long_df, {"row": "a"})
        err = "Cannot wrap the columns while faceting the rows."
        with pytest.raises(RuntimeError, match=err):
            Subplots({}, {}, {"x": ["x", "y", "z"], "wrap": 2}, data)


class TestGridDimensions:

    def test_single_facet(self, long_df):

        key = "a"
        data = PlotData(long_df, {"col": key})
        s = Subplots({}, {}, {}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == n_levels
        assert s.subplot_spec["nrows"] == 1

    def test_two_facets(self, long_df):

        col_key = "a"
        row_key = "b"
        data = PlotData(long_df, {"col": col_key, "row": row_key})
        s = Subplots({}, {}, {}, data)

        n_cols = len(categorical_order(long_df[col_key]))
        n_rows = len(categorical_order(long_df[row_key]))
        assert s.n_subplots == n_cols * n_rows
        assert s.subplot_spec["ncols"] == n_cols
        assert s.subplot_spec["nrows"] == n_rows

    def test_col_facet_wrapped(self, long_df):

        key = "b"
        wrap = 3
        data = PlotData(long_df, {"col": key})
        s = Subplots({}, {"wrap": wrap}, {}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == n_levels // wrap + 1

    def test_row_facet_wrapped(self, long_df):

        key = "b"
        wrap = 3
        data = PlotData(long_df, {"row": key})
        s = Subplots({}, {"wrap": wrap}, {}, data)

        n_levels = len(categorical_order(long_df[key]))
        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == n_levels // wrap + 1
        assert s.subplot_spec["nrows"] == wrap

    def test_col_facet_wrapped_single_row(self, long_df):

        key = "b"
        n_levels = len(categorical_order(long_df[key]))
        wrap = n_levels + 2
        data = PlotData(long_df, {"col": key})
        s = Subplots({}, {"wrap": wrap}, {}, data)

        assert s.n_subplots == n_levels
        assert s.subplot_spec["ncols"] == n_levels
        assert s.subplot_spec["nrows"] == 1

    def test_x_and_y_paired(self, long_df):

        x = ["x", "y", "z"]
        y = ["a", "b"]
        data = PlotData({}, {})
        s = Subplots({}, {}, {"x": x, "y": y}, data)

        assert s.n_subplots == len(x) * len(y)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == len(y)

    def test_x_paired(self, long_df):

        x = ["x", "y", "z"]
        data = PlotData(long_df, {"y": "a"})
        s = Subplots({}, {}, {"x": x}, data)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == len(x)
        assert s.subplot_spec["nrows"] == 1

    def test_y_paired(self, long_df):

        y = ["x", "y", "z"]
        data = PlotData(long_df, {"x": "a"})
        s = Subplots({}, {}, {"y": y}, data)

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == 1
        assert s.subplot_spec["nrows"] == len(y)

    def test_x_paired_and_wrapped(self, long_df):

        x = ["a", "b", "x", "y", "z"]
        wrap = 3
        data = PlotData(long_df, {"y": "t"})
        s = Subplots({}, {}, {"x": x, "wrap": wrap}, data)

        assert s.n_subplots == len(x)
        assert s.subplot_spec["ncols"] == wrap
        assert s.subplot_spec["nrows"] == len(x) // wrap + 1

    def test_y_paired_and_wrapped(self, long_df):

        y = ["a", "b", "x", "y", "z"]
        wrap = 2
        data = PlotData(long_df, {"x": "a"})
        s = Subplots({}, {}, {"y": y, "wrap": wrap}, data)

        assert s.n_subplots == len(y)
        assert s.subplot_spec["ncols"] == len(y) // wrap + 1
        assert s.subplot_spec["nrows"] == wrap
