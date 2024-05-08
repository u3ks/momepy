import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame, GeoSeries
from libpysal.graph import Graph
from pandas import Series

__all__ = ["courtyards", "area_ratio", "count", "node_density"]


def courtyards(geometry: GeoDataFrame | GeoSeries, graph: Graph) -> Series:
    """Calculate the number of courtyards within the joined structure.

    Adapted from :cite:`schirmer2015`.

    Parameters
    ----------
    geometry : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    graph : libpysal.graph.Graph
        A spatial weights matrix for the geodataframe,
        it is used to denote adjacent buildings.

    Returns
    -------
    Series
        A Series containing the resulting values.

    Examples
    --------
    >>> courtyards = mm.calculate_courtyards(buildings_df, graph)
    """

    def _calculate_courtyards(group):
        """helper function to carry out the per group calculations"""
        return shapely.get_num_interior_rings(
            shapely.union_all(shapely.buffer(group.values, 0.01))
        )

    # calculate per group courtyards
    temp_result = (
        geometry["geometry"]
        .groupby(graph.component_labels.values)
        .apply(_calculate_courtyards)
    )
    # assign group courtyard values to each building in the group
    ## note: this is faster than using transform directly
    result = Series(
        temp_result.loc[graph.component_labels.values].values, index=geometry.index
    )

    return result


def area_ratio(
    left: Series, right: Series, right_group_key: Series | np.ndarray
) -> pd.Series:
    """
        Calculate covered area ratio or floor area ratio of objects.
        .. math::
            \\textit{covering object area} \\over \\textit{covered object area}

        Adapted from :cite:`schirmer2015`.

        Parameters
        ----------
        left : Series
            A GeoDataFrame with the areas of the objects being covered (e.g. land unit).
        right : Series
            A GeoDataFrame with the areas of the covering objects (e.g. building).
        right_group_key: np.array | pd.Series
            The group key that assigns objects from ``right`` to ``left``.
        Returns
        -------
        Series

    <<<<<<< HEAD

        Examples
        --------
        >>> tessellation_df['CAR'] = mm.area_ratio(tessellation_df['area'],
        ...                                       buildings_df['area'],
        ...                                       buildings_df['uID'])
    """

    if isinstance(right_group_key, np.ndarray):
        right_group_key = pd.Series(right_group_key, index=right.index)

    results = pd.Series(np.nan, left.index)
    stats = (
        right.loc[right_group_key.index.values].groupby(right_group_key.values).sum()
    )
    results.loc[stats.index.values] = stats.values

    return results / left.values


def count(
    left: GeoDataFrame,
    right: GeoDataFrame,
    right_group_key: Series | np.ndarray,
    weighted=False,
) -> Series:
    """
    Calculate the number of elements within an aggregated structure. Aggregated
    structures can typically be blocks, street segments, or street nodes (their
    snapepd objects). The right gdf has to have a unique ID of aggregated structures
    assigned before hand (e.g. using :py:func:`momepy.get_network_id`).
    If ``weighted=True``, the number of elements will be divided by the area of
    length (based on geometry type) of aggregated elements, to return relative value.

    .. math::
        \\sum_{i \\in aggr} (n_i);\\space \\frac{\\sum_{i \\in aggr} (n_i)}{area_{aggr}}

    Adapted from :cite:`hermosilla2012` and :cite:`feliciotti2018`.

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing aggregation to analyse.
    right : GeoDataFrame
        A GeoDataFrame containing objects to analyse.
    right_group_key: np.array | pd.Series
        The group key that assigns objects from ``right`` to ``left``.
    weighted : bool (default False)
        If ``True``, count will be divided by the area or length.

    Examples
    --------
    >>> blocks_df['buildings_count'] = mm.count(blocks_df,
    ...                                         buildings_df,
    ...                                         buildings_df['bID'])
    """

    if isinstance(right_group_key, np.ndarray):
        right_group_key = Series(right_group_key, index=right.index)

    stats = (
        right.loc[right_group_key.index.values].groupby(right_group_key.values).size()
    )

    # fill missing values with 0s
    results = Series(0, left.index)
    results.loc[stats.index.values] = stats.values
    if weighted:
        if left.geometry[0].geom_type in ["Polygon", "MultiPolygon"]:
            results = results / left.geometry.area
        elif left.geometry[0].geom_type in ["LineString", "MultiLineString"]:
            results = results / left.geometry.length
        else:
            raise TypeError("Geometry type does not support weighting.")

    return results


def node_density(left, right, graph, weighted=False) -> Series:
    """
    Calculate the density of nodes neighbours on street network defined in
    ``spatial_weights``. Calculated as the number of neighbouring
    nodes / cummulative length of street network within neighbours.
    ``node_start``,  ``node_end``, ``mm_len`` is standard output of
    :py:func:`momepy.nx_to_gdf` and is compulsory ``right`` to have
    these columns.

    Adapted from :cite:`dibble2017`.

    Parameters
    ----------
    left : GeoDataFrame
        A GeoDataFrame containing nodes of street network.
    right : GeoDataFrame
        A GeoDataFrame containing edges of street network.
    graph :  libpysal.graph.Graph
        A spatial weights matrix capturing relationship between nodes.
    weighted : bool (default False)
        If ``True``, density will take into account node degree as ``k-1``.

    Returns
    -------
    Series
        A Series containing resulting values.

    Examples
    --------
    >>> nodes['density'] = mm.node_density(nodes, edges, sw)
    """

    if (
        np.intersect1d(["node_start", "node_end", "mm_len"], right.columns).shape[0]
        != 3
    ):
        raise ValueError(
            "'node_start', 'node_end', 'mm_len' are needed for the calculations."
        )

    def _calc_nodedensity(group, edges):
        """ "Helper function to calculate group values."""
        neighbours = group.index.values

        lengths = edges.loc[
            np.in1d(edges["node_start"], neighbours)
            & np.in1d(edges["node_end"], neighbours),
            "mm_len",
        ].sum()
        return group.sum() / lengths if lengths else 0

    if weighted:
        summation_values = left["degree"] - 1
    else:
        summation_values = pd.Series(np.ones(left.shape[0]), index=left.index)

    return graph.apply(summation_values, _calc_nodedensity, edges=right)
