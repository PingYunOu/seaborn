import functools
import itertools
import warnings
import imghdr

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from seaborn._core.plot import Plot
from seaborn._core.rules import categorical_order
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat

assert_vector_equal = functools.partial(assert_series_equal, check_names=False)


class MockStat(Stat):

    def __call__(self, data):

        return data


class MockMark(Mark):

    # TODO we need to sort out the stat application, it is broken right now
    # default_stat = MockStat
    grouping_vars = ["hue"]

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.passed_keys = []
        self.passed_data = []
        self.passed_axes = []
        self.n_splits = 0

    def _plot_split(self, keys, data, ax, mappings, kws):

        self.n_splits += 1
        self.passed_keys.append(keys)
        self.passed_data.append(data)
        self.passed_axes.append(ax)


class TestInit:

    def test_empty(self):

        p = Plot()
        assert p._data._source_data is None
        assert p._data._source_vars == {}

    def test_data_only(self, long_df):

        p = Plot(long_df)
        assert p._data._source_data is long_df
        assert p._data._source_vars == {}

    def test_df_and_named_variables(self, long_df):

        variables = {"x": "a", "y": "z"}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], long_df[col])
        assert p._data._source_data is long_df
        assert p._data._source_vars.keys() == variables.keys()

    def test_df_and_mixed_variables(self, long_df):

        variables = {"x": "a", "y": long_df["z"]}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            if isinstance(col, str):
                assert_vector_equal(p._data.frame[var], long_df[col])
            else:
                assert_vector_equal(p._data.frame[var], col)
        assert p._data._source_data is long_df
        assert p._data._source_vars.keys() == variables.keys()

    def test_vector_variables_only(self, long_df):

        variables = {"x": long_df["a"], "y": long_df["z"]}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], col)
        assert p._data._source_data is None
        assert p._data._source_vars.keys() == variables.keys()

    def test_vector_variables_no_index(self, long_df):

        variables = {"x": long_df["a"].to_numpy(), "y": long_df["z"].to_list()}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], pd.Series(col))
            assert p._data.names[var] is None
        assert p._data._source_data is None
        assert p._data._source_vars.keys() == variables.keys()

    def test_scales(self, long_df):

        p = Plot(long_df, x="x", y="y")
        for var in "xy":
            assert var in p._scales
            assert p._scales[var].type == "unknown"


class TestLayerAddition:

    def test_without_data(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark())
        p._setup_layers()
        layer, = p._layers
        assert_frame_equal(p._data.frame, layer.data.frame)

    def test_with_new_variable_by_name(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y="y")
        p._setup_layers()
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_with_new_variable_by_vector(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y=long_df["y"])
        p._setup_layers()
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_with_late_data_definition(self, long_df):

        p = Plot().add(MockMark(), data=long_df, x="x", y="y")
        p._setup_layers()
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_with_new_data_definition(self, long_df):

        long_df_sub = long_df.sample(frac=.5)

        p = Plot(long_df, x="x", y="y").add(MockMark(), data=long_df_sub)
        p._setup_layers()
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(
                layer.data.frame[var], long_df_sub[var].reindex(long_df.index)
            )

    def test_drop_variable(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark(), y=None)
        p._setup_layers()
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x"]
        assert "y" not in layer
        assert_vector_equal(layer.data.frame["x"], long_df["x"])

    def test_stat_default(self):

        class MarkWithDefaultStat(Mark):
            default_stat = MockStat

        p = Plot().add(MarkWithDefaultStat())
        layer, = p._layers
        assert layer.stat.__class__ is MockStat

    def test_stat_nondefault(self):

        class MarkWithDefaultStat(Mark):
            default_stat = MockStat

        class OtherMockStat(MockStat):
            pass

        p = Plot().add(MarkWithDefaultStat(), OtherMockStat())
        layer, = p._layers
        assert layer.stat.__class__ is OtherMockStat

    @pytest.mark.parametrize(
        "arg,expected",
        [("x", "x"), ("y", "y"), ("v", "x"), ("h", "y")],
    )
    def test_orient(self, arg, expected):

        class MockMarkTrackOrient(MockMark):
            def _adjust(self, data):
                self.orient_at_adjust = self.orient
                return data

        class MockStatTrackOrient(MockStat):
            def setup(self, data):
                super().setup(data)
                self.orient_at_setup = self.orient
                return self

        m = MockMarkTrackOrient()
        s = MockStatTrackOrient()
        Plot(x=[1, 2, 3], y=[1, 2, 3]).add(m, s, orient=arg).plot()

        assert m.orient == expected
        assert m.orient_at_adjust == expected
        assert s.orient == expected
        assert s.orient_at_setup == expected


class TestAxisScaling:

    def test_inference(self, long_df):

        for col, scale_type in zip("zat", ["numeric", "categorical", "datetime"]):
            p = Plot(long_df, x=col, y=col).add(MockMark())
            for var in "xy":
                assert p._scales[var].type == "unknown"
            p._setup_layers()
            p._setup_scales()
            for var in "xy":
                assert p._scales[var].type == scale_type

    def test_inference_concatenates(self):

        p = Plot(x=[1, 2, 3]).add(MockMark(), x=["a", "b", "c"])
        p._setup_layers()
        p._setup_scales()
        assert p._scales["x"].type == "categorical"

    def test_categorical_explicit_order(self):

        p = Plot(x=["b", "c", "a"]).scale_categorical("x", order=["c", "a", "b"])

        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series(["c", "a", "b"])).cat.codes.to_list() == [0, 1, 2]

    def test_numeric_as_categorical(self):

        p = Plot(x=[2, 1, 3]).scale_categorical("x")

        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series([1, 2, 3])).cat.codes.to_list() == [0, 1, 2]

    def test_numeric_as_categorical_explicit_order(self):

        p = Plot(x=[1, 2, 3]).scale_categorical("x", order=[2, 1, 3])

        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series([2, 1, 3])).cat.codes.to_list() == [0, 1, 2]

    def test_numeric_as_datetime(self):

        p = Plot(x=[1, 2, 3]).scale_datetime("x")
        scl = p._scales["x"]
        assert scl.type == "datetime"

        numbers = [2, 1, 3]
        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        assert_series_equal(
            scl.cast(pd.Series(numbers)),
            pd.Series(dates, dtype="datetime64[ns]")
        )

    @pytest.mark.xfail
    def test_categorical_as_numeric(self):

        # TODO marked as expected fail because we have not implemented this yet
        # see notes in ScaleWrapper.cast

        strings = ["2", "1", "3"]
        p = Plot(x=strings).scale_numeric("x")
        scl = p._scales["x"]
        assert scl.type == "numeric"
        assert_series_equal(
            scl.cast(pd.Series(strings)),
            pd.Series(strings).astype(float)
        )

    def test_categorical_as_datetime(self):

        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        p = Plot(x=dates).scale_datetime("x")
        scl = p._scales["x"]
        assert scl.type == "datetime"
        assert_series_equal(
            scl.cast(pd.Series(dates, dtype=object)),
            pd.Series(dates, dtype="datetime64[ns]")
        )

    def test_mark_data_log_transform(self, long_df):

        col = "z"
        m = MockMark()
        Plot(long_df, x=col).scale_numeric("x", "log").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df[col])

    def test_mark_data_log_transfrom_with_stat(self, long_df):

        class Mean(Stat):
            def __call__(self, data):
                return data.mean()

        col = "z"
        grouper = "a"
        m = MockMark()
        s = Mean()

        Plot(long_df, x=grouper, y=col).scale_numeric("y", "log").add(m, s).plot()

        expected = (
            long_df[col]
            .pipe(np.log)
            .groupby(long_df[grouper], sort=False)
            .mean()
            .pipe(np.exp)
            .reset_index(drop=True)
        )
        assert_vector_equal(m.passed_data[0]["y"], expected)

    def test_mark_data_from_categorical(self, long_df):

        col = "a"
        m = MockMark()
        Plot(long_df, x=col).add(m).plot()

        levels = categorical_order(long_df[col])
        level_map = {x: float(i) for i, x in enumerate(levels)}
        assert_vector_equal(m.passed_data[0]["x"], long_df[col].map(level_map))

    def test_mark_data_from_datetime(self, long_df):

        col = "t"
        m = MockMark()
        Plot(long_df, x=col).add(m).plot()

        assert_vector_equal(m.passed_data[0]["x"], long_df[col].map(mpl.dates.date2num))


class TestPlotting:

    def test_matplotlib_object_creation(self):

        p = Plot()
        p._setup_figure()
        assert isinstance(p._figure, mpl.figure.Figure)
        for sub in p._subplot_list:
            assert isinstance(sub["ax"], mpl.axes.Axes)

    def test_empty(self):

        m = MockMark()
        Plot().plot()
        assert m.n_splits == 0

    def test_single_split_single_layer(self, long_df):

        m = MockMark()
        p = Plot(long_df, x="f", y="z").add(m).plot()
        assert m.n_splits == 1

        assert m.passed_keys[0] == {}
        assert m.passed_axes[0] is p._subplot_list[0]["ax"]
        assert_frame_equal(m.passed_data[0], p._data.frame)

    def test_single_split_multi_layer(self, long_df):

        vs = [{"hue": "a", "size": "z"}, {"hue": "b", "style": "c"}]

        class NoGroupingMark(MockMark):
            grouping_vars = []

        ms = [NoGroupingMark(), NoGroupingMark()]
        Plot(long_df).add(ms[0], **vs[0]).add(ms[1], **vs[1]).plot()

        for m, v in zip(ms, vs):
            for var, col in v.items():
                assert_vector_equal(m.passed_data[0][var], long_df[col])

    def check_splits_single_var(self, plot, mark, split_var, split_keys):

        assert mark.n_splits == len(split_keys)
        assert mark.passed_keys == [{split_var: key} for key in split_keys]

        full_data = plot._data.frame
        for i, key in enumerate(split_keys):

            split_data = full_data[full_data[split_var] == key]
            assert_frame_equal(mark.passed_data[i], split_data)

    def check_splits_multi_vars(self, plot, mark, split_vars, split_keys):

        assert mark.n_splits == np.prod([len(ks) for ks in split_keys])

        expected_keys = [
            dict(zip(split_vars, level_keys))
            for level_keys in itertools.product(*split_keys)
        ]
        assert mark.passed_keys == expected_keys

        full_data = plot._data.frame
        for i, keys in enumerate(itertools.product(*split_keys)):

            use_rows = pd.Series(True, full_data.index)
            for var, key in zip(split_vars, keys):
                use_rows &= full_data[var] == key
            split_data = full_data[use_rows]
            assert_frame_equal(mark.passed_data[i], split_data)

    @pytest.mark.parametrize(
        "split_var", [
            "hue",  # explicitly declared on the Mark
            "group",  # implicitly used for all Mark classes
        ])
    def test_one_grouping_variable(self, long_df, split_var):

        split_col = "a"

        m = MockMark()
        p = Plot(long_df, x="f", y="z", **{split_var: split_col}).add(m).plot()

        split_keys = categorical_order(long_df[split_col])
        assert m.passed_axes == [p._subplot_list[0]["ax"] for _ in split_keys]
        self.check_splits_single_var(p, m, split_var, split_keys)

    def test_two_grouping_variables(self, long_df):

        split_vars = ["hue", "group"]
        split_cols = ["a", "b"]
        variables = {var: col for var, col in zip(split_vars, split_cols)}

        m = MockMark()
        p = Plot(long_df, y="z", **variables).add(m).plot()

        split_keys = [categorical_order(long_df[col]) for col in split_cols]
        assert m.passed_axes == [
            p._subplot_list[0]["ax"] for _ in itertools.product(*split_keys)
        ]
        self.check_splits_multi_vars(p, m, split_vars, split_keys)

    def test_facets_no_subgroups(self, long_df):

        split_var = "col"
        split_col = "b"

        m = MockMark()
        p = Plot(long_df, x="f", y="z", **{split_var: split_col}).add(m).plot()

        split_keys = categorical_order(long_df[split_col])
        assert m.passed_axes == list(p._figure.axes)
        self.check_splits_single_var(p, m, split_var, split_keys)

    def test_facets_one_subgroup(self, long_df):

        facet_var, facet_col = "col", "a"
        group_var, group_col = "group", "b"

        m = MockMark()
        p = (
            Plot(long_df, x="f", y="z", **{group_var: group_col, facet_var: facet_col})
            .add(m)
            .plot()
        )

        split_keys = [categorical_order(long_df[col]) for col in [facet_col, group_col]]
        assert m.passed_axes == [
            ax
            for ax in list(p._figure.axes)
            for _ in categorical_order(long_df[group_col])
        ]
        self.check_splits_multi_vars(p, m, [facet_var, group_var], split_keys)

    def test_layer_specific_facet_disabling(self, long_df):

        axis_vars = {"x": "y", "y": "z"}
        row_var = "a"

        m = MockMark()
        p = Plot(long_df, **axis_vars, row=row_var).add(m, row=None).plot()

        col_levels = categorical_order(long_df[row_var])
        assert len(p._figure.axes) == len(col_levels)

        for data in m.passed_data:
            for var, col in axis_vars.items():
                assert_vector_equal(data[var], long_df[col])

    def test_paired_variables(self, long_df):

        x = ["x", "y"]
        y = ["f", "z"]

        m = MockMark()
        Plot(long_df).pair(x, y).add(m).plot()

        var_product = itertools.product(y, x)

        for data, (y_i, x_i) in zip(m.passed_data, var_product):
            assert_vector_equal(data["x"], long_df[x_i].astype(float))
            assert_vector_equal(data["y"], long_df[y_i].astype(float))

    def test_paired_one_dimension(self, long_df):

        x = ["y", "z"]

        m = MockMark()
        Plot(long_df).pair(x).add(m).plot()

        for data, x_i in zip(m.passed_data, x):
            assert_vector_equal(data["x"], long_df[x_i].astype(float))

    def test_paired_variables_one_subset(self, long_df):

        x = ["x", "y"]
        y = ["f", "z"]
        group = "a"

        long_df["x"] = long_df["x"].astype(float)  # simplify vector comparison

        m = MockMark()
        Plot(long_df, group=group).pair(x, y).add(m).plot()

        groups = categorical_order(long_df[group])
        var_product = itertools.product(y, x, groups)

        for data, (y_i, x_i, g_i) in zip(m.passed_data, var_product):
            rows = long_df[group] == g_i
            assert_vector_equal(data["x"], long_df.loc[rows, x_i])
            assert_vector_equal(data["y"], long_df.loc[rows, y_i])

    def test_paired_and_faceted(self, long_df):

        x = ["y", "z"]
        y = "f"
        row = "c"

        m = MockMark()
        Plot(long_df, y=y, row=row).pair(x).add(m).plot()

        facets = categorical_order(long_df[row])
        var_product = itertools.product(facets, x)

        for data, (f_i, x_i) in zip(m.passed_data, var_product):
            rows = long_df[row] == f_i
            assert_vector_equal(data["x"], long_df.loc[rows, x_i])
            assert_vector_equal(data["y"], long_df.loc[rows, y])

    def test_adjustments(self, long_df):

        orig_df = long_df.copy(deep=True)

        class AdjustableMockMark(MockMark):
            def _adjust(self, data):
                data["x"] = data["x"] + 1
                return data

        m = AdjustableMockMark()
        Plot(long_df, x="z", y="z").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] + 1)
        assert_vector_equal(m.passed_data[0]["y"], long_df["z"])

        assert_frame_equal(long_df, orig_df)   # Test data was not mutated

    def test_adjustments_log_scale(self, long_df):

        class AdjustableMockMark(MockMark):
            def _adjust(self, data):
                data["x"] = data["x"] - 1
                return data

        m = AdjustableMockMark()
        Plot(long_df, x="z", y="z").scale_numeric("x", "log").add(m).plot()
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] / 10)

    def test_clone(self, long_df):

        p1 = Plot(long_df)
        p2 = p1.clone()
        assert isinstance(p2, Plot)
        assert p1 is not p2
        assert p1._data._source_data is not p2._data._source_data

        p2.add(MockMark())
        assert not p1._layers

    def test_default_is_no_pyplot(self):

        p = Plot().plot()

        assert not plt.get_fignums()
        assert isinstance(p._figure, mpl.figure.Figure)

    def test_with_pyplot(self):

        p = Plot().plot(pyplot=True)

        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()
        assert p._figure is fig

    def test_show(self):

        p = Plot()

        with warnings.catch_warnings(record=True) as msg:
            out = p.show(block=False)
        assert out is None
        assert not hasattr(p, "_figure")

        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()

        gui_backend = (
            # From https://github.com/matplotlib/matplotlib/issues/20281
            fig.canvas.manager.show != mpl.backend_bases.FigureManagerBase.show
        )
        if not gui_backend:
            assert msg

    def test_png_representation(self):

        p = Plot()
        out = p._repr_png_()

        assert not hasattr(p, "_figure")
        assert isinstance(out, bytes)
        assert imghdr.what("", out) == "png"

    @pytest.mark.xfail(reason="Plot.save not yet implemented")
    def test_save(self):

        Plot().save()


class TestFacetInterface:

    @pytest.fixture(scope="class", params=["row", "col"])
    def dim(self, request):
        return request.param

    @pytest.fixture(scope="class", params=["reverse", "subset", "expand"])
    def reorder(self, request):
        return {
            "reverse": lambda x: x[::-1],
            "subset": lambda x: x[:-1],
            "expand": lambda x: x + ["z"],
        }[request.param]

    def check_facet_results_1d(self, p, df, dim, key, order=None):

        p = p.plot()

        order = categorical_order(df[key], order)
        assert len(p._figure.axes) == len(order)

        other_dim = {"row": "col", "col": "row"}[dim]

        for subplot, level in zip(p._subplot_list, order):
            assert subplot[dim] == level
            assert subplot[other_dim] is None
            assert subplot["ax"].get_title() == f"{key} = {level}"
            assert getattr(subplot["ax"].get_gridspec(), f"n{dim}s") == len(order)

    def test_1d_from_init(self, long_df, dim):

        key = "a"
        p = Plot(long_df, **{dim: key})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_facet(self, long_df, dim):

        key = "a"
        p = Plot(long_df).facet(**{dim: key})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_init_as_vector(self, long_df, dim):

        key = "a"
        p = Plot(long_df, **{dim: long_df[key]})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_facet_as_vector(self, long_df, dim):

        key = "a"
        p = Plot(long_df).facet(**{dim: long_df[key]})
        self.check_facet_results_1d(p, long_df, dim, key)

    def test_1d_from_init_with_order(self, long_df, dim, reorder):

        key = "a"
        order = reorder(categorical_order(long_df[key]))
        p = Plot(long_df, **{dim: key}).facet(**{f"{dim}_order": order})
        self.check_facet_results_1d(p, long_df, dim, key, order)

    def test_1d_from_facet_with_order(self, long_df, dim, reorder):

        key = "a"
        order = reorder(categorical_order(long_df[key]))
        p = Plot(long_df).facet(**{dim: key, f"{dim}_order": order})
        self.check_facet_results_1d(p, long_df, dim, key, order)

    def check_facet_results_2d(self, p, df, variables, order=None):

        p = p.plot()

        if order is None:
            order = {dim: categorical_order(df[key]) for dim, key in variables.items()}

        levels = itertools.product(*[order[dim] for dim in ["row", "col"]])
        assert len(p._subplot_list) == len(list(levels))

        for subplot, (row_level, col_level) in zip(p._subplot_list, levels):
            assert subplot["row"] == row_level
            assert subplot["col"] == col_level
            assert subplot["axes"].get_title() == (
                f"{variables['row']} = {row_level} | {variables['col']} = {col_level}"
            )
            gridspec = subplot["axes"].get_gridspec()
            assert gridspec.nrows == len(levels["row"])
            assert gridspec.ncols == len(levels["col"])

    def test_2d_from_init(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot(long_df, **variables)
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_facet(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot(long_df).facet(**variables)
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_init_and_facet(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot(long_df, row=variables["row"]).facet(col=variables["col"])
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_facet_with_data(self, long_df):

        variables = {"row": "a", "col": "c"}
        p = Plot().facet(**variables, data=long_df)
        self.check_facet_results_2d(p, long_df, variables)

    def test_2d_from_facet_with_order(self, long_df, reorder):

        variables = {"row": "a", "col": "c"}
        order = {
            dim: reorder(categorical_order(long_df[key]))
            for dim, key in variables.items()
        }

        order_kws = {"row_order": order["row"], "col_order": order["col"]}
        p = Plot(long_df).facet(**variables, **order_kws)
        self.check_facet_results_2d(p, long_df, variables, order)

    def test_axis_sharing(self, long_df):

        variables = {"row": "a", "col": "c"}

        p = Plot(long_df).facet(**variables)

        p1 = p.clone().plot()
        root, *other = p1._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert all(shareset.joined(root, ax) for ax in other)

        p2 = p.clone().configure(sharex=False, sharey=False).plot()
        root, *other = p2._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)

        p3 = p.clone().configure(sharex="col", sharey="row").plot()
        shape = (
            len(categorical_order(long_df[variables["row"]])),
            len(categorical_order(long_df[variables["col"]])),
        )
        axes_matrix = np.reshape(p3._figure.axes, shape)

        for (shared, unshared), vectors in zip(
            ["yx", "xy"], [axes_matrix, axes_matrix.T]
        ):
            for root, *other in vectors:
                shareset = {
                    axis: getattr(root, f"get_shared_{axis}_axes")() for axis in "xy"
                }
                assert all(shareset[shared].joined(root, ax) for ax in other)
                assert not any(shareset[unshared].joined(root, ax) for ax in other)


class TestPairInterface:

    def check_pair_grid(self, p, x, y):

        xys = itertools.product(y, x)

        for (y_i, x_j), subplot in zip(xys, p._subplot_list):

            ax = subplot["ax"]
            assert ax.get_xlabel() == "" if x_j is None else x_j
            assert ax.get_ylabel() == "" if y_i is None else y_i

            gs = subplot["ax"].get_gridspec()
            assert gs.ncols == len(x)
            assert gs.nrows == len(y)

    @pytest.mark.parametrize(
        "vector_type", [list, np.array, pd.Series, pd.Index]
    )
    def test_all_numeric(self, long_df, vector_type):

        x, y = ["x", "y", "z"], ["s", "f"]
        p = Plot(long_df).pair(vector_type(x), vector_type(y)).plot()
        self.check_pair_grid(p, x, y)

    def test_single_variable_key(self, long_df):

        x, y = ["x", "y"], "z"

        p = Plot(long_df).pair(x, y).plot()
        self.check_pair_grid(p, x, [y])

    @pytest.mark.parametrize("dim", ["x", "y"])
    def test_single_dimension(self, long_df, dim):

        variables = {"x": None, "y": None}
        variables[dim] = ["x", "y", "z"]
        p = Plot(long_df).pair(**variables).plot()
        variables = {k: [v] if v is None else v for k, v in variables.items()}
        self.check_pair_grid(p, **variables)

    def test_non_cartesian(self, long_df):

        x = ["x", "y"]
        y = ["f", "z"]

        p = Plot(long_df).pair(x, y, cartesian=False).plot()

        for i, subplot in enumerate(p._subplot_list):
            ax = subplot["ax"]
            assert ax.get_xlabel() == x[i]
            assert ax.get_ylabel() == y[i]
            assert ax.get_gridspec().nrows == 1
            assert ax.get_gridspec().ncols == len(x) == len(y)

    def test_with_facets(self, long_df):

        x = "x"
        y = ["y", "z"]
        col = "a"

        p = Plot(long_df, x=x).facet(col).pair(y=y).plot()

        facet_levels = categorical_order(long_df[col])
        dims = itertools.product(y, facet_levels)

        for (y_i, col_i), subplot in zip(dims, p._subplot_list):

            ax = subplot["ax"]
            assert ax.get_xlabel() == x
            assert ax.get_ylabel() == y_i
            assert ax.get_title() == f"{col} = {col_i}"

            gs = subplot["ax"].get_gridspec()
            assert gs.ncols == len(facet_levels)
            assert gs.nrows == len(y)

    @pytest.mark.parametrize("variables", [("rows", "y"), ("columns", "x")])
    def test_error_on_facet_overlap(self, long_df, variables):

        facet_dim, pair_axis = variables
        p = Plot(long_df, **{facet_dim[:3]: "a"}).pair(**{pair_axis: ["x", "y"]})
        expected = f"Cannot facet on the {facet_dim} while pairing on {pair_axis}."
        with pytest.raises(RuntimeError, match=expected):
            p.plot()

    @pytest.mark.parametrize("variables", [("columns", "y"), ("rows", "x")])
    def test_error_on_wrap_overlap(self, long_df, variables):

        facet_dim, pair_axis = variables
        p = (
            Plot(long_df, **{facet_dim[:3]: "a"})
            .facet(wrap=2)
            .pair(**{pair_axis: ["x", "y"]})
        )
        expected = f"Cannot wrap the {facet_dim} while pairing on {pair_axis}."
        with pytest.raises(RuntimeError, match=expected):
            p.plot()

    def test_axis_sharing(self, long_df):

        p = Plot(long_df).pair(x=["a", "b"], y=["y", "z"])
        shape = 2, 2

        p1 = p.clone().plot()
        axes_matrix = np.reshape(p1._figure.axes, shape)

        for root, *other in axes_matrix:  # Test row-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert not any(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert all(y_shareset.joined(root, ax) for ax in other)

        for root, *other in axes_matrix.T:  # Test col-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert all(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert not any(y_shareset.joined(root, ax) for ax in other)

        p2 = p.clone().configure(sharex=False, sharey=False).plot()
        root, *other = p2._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)

    def test_axis_sharing_with_facets(self, long_df):

        p = Plot(long_df, y="y").pair(x=["a", "b"]).facet(row="c").plot()
        shape = 2, 2

        axes_matrix = np.reshape(p._figure.axes, shape)

        for root, *other in axes_matrix:  # Test row-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert not any(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert all(y_shareset.joined(root, ax) for ax in other)

        for root, *other in axes_matrix.T:  # Test col-wise sharing
            x_shareset = getattr(root, "get_shared_x_axes")()
            assert all(x_shareset.joined(root, ax) for ax in other)
            y_shareset = getattr(root, "get_shared_y_axes")()
            assert all(y_shareset.joined(root, ax) for ax in other)


# TODO Current untested includes:
# - anything having to do with semantic mapping
# - interaction with existing matplotlib objects
# - any important corner cases in the original test_core suite
