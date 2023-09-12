from typing import Literal

import polars as pl
import pyranges as pr

from genomic_ranges.contig_ranges import ContigRanges
from genomic_ranges.methods.intersection import _intersection


class GenomicRanges:
    def __init__(self, obj):
        if isinstance(obj, dict):
            if all(list(isinstance(v, ContigRanges) for v in obj.values())):
                contig_ranges = obj
            else:
                raise NotImplementedError
        elif isinstance(obj, pr.PyRanges):
            contig_ranges = {}
            for key, df in obj.dfs.items():
                if isinstance(key, tuple):
                    literals = {"Chromosome": key[0], "Strand": key[1]}
                else:
                    literals = {"Chromosome": key}
                contig_ranges[key] = ContigRanges(
                    pl.from_pandas(df.drop(columns=literals.keys())), literals
                )
        else:
            raise ValueError(
                "`GenomicRanges` expects either a dictionary of polars "
                "DataFrames or a PyRanges object as input."
            )
        # TODO: make sure all cotig_ranges have the same keys in `._literals`
        self._contig_ranges = contig_ranges

    def _conc_dfs(self, **kwargs):
        with pl.StringCache():
            return pl.concat(self.dfs.values(), **kwargs)

    def __str__(self):
        # NOTE: this cats the dfs for now and we should find something more efficient
        comb = self._conc_dfs()
        return comb.__str__()

    def __repr__(self):
        # NOTE: this cats the dfs for now and we should find something more efficient
        comb = self._conc_dfs()
        return comb.__repr__()

    @property
    def dfs(self):
        return {k: v.df for k, v in self._contig_ranges.items()}

    def to_pyranges(self, key_dict=None):
        """
        Create PyRanges object.

        Parameters
        -------
        key_dict
            Dictionary for renaming the corresponding GenomicRanges columns to
            "Chromosome" and (optionally) "Strand" in the PyRanges object.

        Returns
        -------
        PyRanges DataFrame.

        Examples
        --------
        # TODO: fix examples to work with new interface
        Read BED file to Polars DataFrame with pyarrow engine.

        >>> bed_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow")

        Create PyRanges object directly from Polars DataFrame.

        >>> bed_df_pr = create_pyranges_from_polars_df(bed_df_pl=bed_df_pl)

        """
        dfs = {}
        for cr in self._contig_ranges.values():
            df = cr.df.to_pandas()
            if key_dict is None and "Chromosome" not in df.columns:
                raise ValueError(
                    "Either a `key_dict` needs to be passed or a "
                    "column 'Chromosome' needs to be present."
                )

            if key_dict:
                # the `key_dict` needs to contain 'Chromosome' and can contain 'Strand'
                if "Chromosome" not in key_dict:
                    raise ValueError(
                        "`key_dict` needs to contain a value for 'Chromosome'."
                    )
                chr_lit_val = cr._literals[key_dict["Chromosome"]]
                strand_lit_val = None
                df = df.rename(columns={key_dict["Chromosome"]: "Chromosome"})
                if "Strand" in key_dict:
                    strand_lit_val = cr._literals[key_dict["Strand"]]
                    df = df.rename(columns={key_dict["Strand"]: "Strand"})
            else:
                chr_lit_val = cr._literals["Chromosome"]
                strand_lit_val = cr._literals.get("Strand")

            key = chr_lit_val
            if strand_lit_val is not None:
                key = (chr_lit_val, strand_lit_val)
            dfs[key] = df

        # create empty PyRanges object to populate later
        pr_obj = pr.PyRanges()

        pr_obj.__dict__["dfs"] = dfs
        pr_obj.__dict__["features"] = pr.genomicfeatures.GenomicFeaturesMethods(pr_obj)
        pr_obj.__dict__["stats"] = pr.statistics.StatisticsMethods(pr_obj)

        return pr_obj

    def intersection(
        self,
        other: "GenomicRanges",
        how: Literal["all", "containment", "first", "last"] | str | None = None,
        regions1_info: bool = True,
        regions2_info: bool = False,
        regions1_coord: bool = False,
        regions2_coord: bool = False,
        regions1_suffix: str = "@1",
        regions2_suffix: str = "@2",
    ) -> GenomicRanges:
        """
        Get overlapping subintervals between first set and second set of regions.

        Parameters
        ----------
        regions1_df_pl
            Polars DataFrame containing BED entries for first set of regions.
        regions2_df_pl
            Polars DataFrame containing BED entries for second set of regions.
        how
            What intervals to report:
              - ``"all"`` (``None``): all overlaps with second set or regions.
              - ``"containment"``: only overlaps where region of first set is contained
                within region of second set.
              - ``"first"``: first overlap with second set of regions.
              - ``"last"``: last overlap with second set of regions.
              - ``"outer"``: all regions for first and all regions of second (outer join).
                If no overlap was found for a region, the other region set will contain
                ``None`` for that entry.
              - ``"left"``: all first set of regions and overlap with second set of regions
                (left join).
                If no overlap was found for a region in the first set, the second region
                set will contain None for that entry.
              - ``"right"``: all second set of regions and overlap with first set of regions
                (right join).
                If no overlap was found for a region in the second set, the first region
                set will contain ``None`` for that entry.
        regions1_info
            Add non-coordinate columns from first set of regions to output of intersection.
        regions2_info
            Add non-coordinate columns from first set of regions to output of intersection.
        regions1_coord
            Add coordinates from first set of regions to output of intersection.
        regions2_coord
            Add coordinates from second set of regions to output of intersection.
        regions1_suffix
            Suffix added to coordinate columns of first set of regions.
        regions2_suffix
            Suffix added to coordinate and info columns of second set of regions.

        strandedness
            Note: Not implemented yet.
            {``None``, ``"same"``, ``"opposite"``, ``False``}, default ``None``, i.e. auto
            Whether to compare PyRanges on the same strand, the opposite or ignore strand
            information. The default, ``None``, means use ``"same"`` if both PyRanges are
            stranded, otherwise ignore the strand information.

        Returns
        -------
        intersection_df_pl
            Polars Dataframe containing BED entries with the intersection.

        Examples
        --------
        >>> regions1_df_pl = pl.from_dict(
        ...     {
        ...         "Chromosome": ["chr1"] * 3,
        ...         "Start": [1, 4, 10],
        ...         "End": [3, 9, 11],
        ...         "ID": ["a", "b", "c"],
        ...     }
        ... )
        >>> regions1_df_pl
        shape: (3, 4)
        ┌────────────┬───────┬─────┬─────┐
        │ Chromosome ┆ Start ┆ End ┆ ID  │
        │ ---        ┆ ---   ┆ --- ┆ --- │
        │ str        ┆ i64   ┆ i64 ┆ str │
        ╞════════════╪═══════╪═════╪═════╡
        │ chr1       ┆ 1     ┆ 3   ┆ a   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 4     ┆ 9   ┆ b   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 10    ┆ 11  ┆ c   │
        └────────────┴───────┴─────┴─────┘

        >>> regions2_df_pl = pl.from_dict(
        ...     {
        ...         "Chromosome": ["chr1"] * 3,
        ...         "Start": [2, 2, 9],
        ...         "End": [3, 9, 10],
        ...         "Name": ["reg1", "reg2", "reg3"]
        ...     }
        ... )
        >>> regions2_df_pl
        shape: (3, 4)
        ┌────────────┬───────┬─────┬──────┐
        │ Chromosome ┆ Start ┆ End ┆ Name │
        │ ---        ┆ ---   ┆ --- ┆ ---  │
        │ str        ┆ i64   ┆ i64 ┆ str  │
        ╞════════════╪═══════╪═════╪══════╡
        │ chr1       ┆ 2     ┆ 3   ┆ reg1 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ chr1       ┆ 2     ┆ 9   ┆ reg2 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ chr1       ┆ 9     ┆ 10  ┆ reg3 │
        └────────────┴───────┴─────┴──────┘

        >>> intersection(regions1_df_pl, regions2_df_pl)
        shape: (3, 3)
        ┌────────────┬───────┬─────┬─────┐
        │ Chromosome ┆ Start ┆ End ┆ ID  │
        │ ---        ┆ ---   ┆ --- ┆ --- │
        │ str        ┆ i64   ┆ i64 ┆ str │
        ╞════════════╪═══════╪═════╪═════╡
        │ chr1       ┆ 2     ┆ 3   ┆ a   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 2     ┆ 3   ┆ a   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 4     ┆ 9   ┆ b   │
        └────────────┴───────┴─────┴─────┘

        >>> intersection(regions1_df_pl, regions2_df_pl, how="first")
        shape: (2, 4)
        ┌────────────┬───────┬─────┬─────┐
        │ Chromosome ┆ Start ┆ End ┆ ID  │
        │ ---        ┆ ---   ┆ --- ┆ --- │
        │ str        ┆ i64   ┆ i64 ┆ str │
        ╞════════════╪═══════╪═════╪═════╡
        │ chr1       ┆ 2     ┆ 3   ┆ a   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 4     ┆ 9   ┆ b   │
        └────────────┴───────┴─────┴─────┘

        >>> intersection(
        ...     regions1_df_pl,
        ...     regions2_df_pl,
        ...     how="containment",
        ...     regions1_info=False,
        ...     regions2_info=True,
        ... )
        shape: (1, 4)
        ┌────────────┬───────┬─────┬──────┐
        │ Chromosome ┆ Start ┆ End ┆ Name │
        │ ---        ┆ ---   ┆ --- ┆ ---  │
        │ str        ┆ i64   ┆ i64 ┆ str  │
        ╞════════════╪═══════╪═════╪══════╡
        │ chr1       ┆ 4     ┆ 9   ┆ reg2 │
        └────────────┴───────┴─────┴──────┘

        >>> intersection(
        ...     regions1_df_pl,
        ...     regions2_df_pl,
        ...     regions1_coord=True,
        ...     regions2_coord=True,
        ... )
        shape: (3, 10)
        ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────────────┬─────────┬───────┬─────┐
        │ Chromosome ┆ Start ┆ End ┆ Chromosome@1 ┆ Start@1 ┆ End@1 ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ ID  │
        │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---          ┆ ---     ┆ ---   ┆ --- │
        │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str          ┆ i64     ┆ i64   ┆ str │
        ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════════════╪═════════╪═══════╪═════╡
        │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 9     ┆ a   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 1       ┆ 3     ┆ chr1         ┆ 2       ┆ 3     ┆ a   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┤
        │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 4       ┆ 9     ┆ chr1         ┆ 2       ┆ 9     ┆ b   │
        └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────────────┴─────────┴───────┴─────┘

        >>> intersection(
        ...     regions1_df_pl,
        ...     regions2_df_pl,
        ...     regions1_info=False,
        ...     regions_info=True,
        ...     regions2_coord=True,
        ... )
        shape: (3, 7)
        ┌────────────┬───────┬─────┬──────────────┬─────────┬───────┬──────┐
        │ Chromosome ┆ Start ┆ End ┆ Chromosome@2 ┆ Start@2 ┆ End@2 ┆ Name │
        │ ---        ┆ ---   ┆ --- ┆ ---          ┆ ---     ┆ ---   ┆ ---  │
        │ str        ┆ i64   ┆ i64 ┆ str          ┆ i64     ┆ i64   ┆ str  │
        ╞════════════╪═══════╪═════╪══════════════╪═════════╪═══════╪══════╡
        │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 2       ┆ 9     ┆ reg2 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ chr1       ┆ 2     ┆ 3   ┆ chr1         ┆ 2       ┆ 3     ┆ reg1 │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │ chr1       ┆ 4     ┆ 9   ┆ chr1         ┆ 2       ┆ 9     ┆ reg2 │
        └────────────┴───────┴─────┴──────────────┴─────────┴───────┴──────┘

        """
        intersection_chrom_dfs_pl = {}

        for chrom_and_or_strand, contig_ranges1 in self._contig_ranges.items():
            if chrom_and_or_strand in other._contig_ranges:
                contig_ranges2 = other._contig_ranges[chrom_and_or_strand]

                # Find intersection between regions form first and second per chromosome
                # dataframe and return index positions in both dataframes for those
                # intersections.
                regions1_indexes, regions2_indexes = _intersection(
                    contig_ranges1=contig_ranges1,
                    contig_ranges2=contig_ranges2,
                    how=how,
                )

                # Skip empty intersections.
                if regions1_indexes.shape[0] == 0:
                    continue

                # Get all regions from first and second per chromosome dataframe for the
                # index positions calculated above.
                intersection_chrom_df_pl = (
                    contig_ranges1.df[regions1_indexes]
                    .select(pl.all().suffix(regions1_suffix))
                    .hstack(
                        (contig_ranges2.df[regions2_indexes]).select(
                            pl.all().suffix(regions2_suffix)
                        )
                    )
                )

                # Calculate intersection start and end coordinates and return the columns
                # of interest.
                intersection_chrom_ldf_pl = (
                    intersection_chrom_df_pl.lazy()
                    .with_columns(
                        [
                            # Chromosome name for intersection.
                            pl.coalesce(
                                pl.col(f"Chromosome{regions1_suffix}"),
                                pl.col(f"Chromosome{regions2_suffix}"),
                            ).alias("Chromosome"),
                            # Calculate start coordinate for intersection.
                            pl.when(
                                pl.col(f"Start{regions1_suffix}")
                                > pl.col(f"Start{regions2_suffix}")
                            )
                            .then(pl.col(f"Start{regions1_suffix}"))
                            .otherwise(
                                pl.coalesce(
                                    pl.col(f"Start{regions2_suffix}"),
                                    pl.col(f"Start{regions1_suffix}"),
                                )
                            )
                            .alias("Start"),
                            # Calculate end coordinate for intersection.
                            pl.when(
                                pl.col(f"End{regions1_suffix}")
                                < pl.col(f"End{regions2_suffix}")
                            )
                            .then(pl.col(f"End{regions1_suffix}"))
                            .otherwise(
                                pl.coalesce(
                                    pl.col(f"End{regions2_suffix}"),
                                    pl.col(f"End{regions1_suffix}"),
                                )
                            )
                            .alias("End"),
                        ]
                    )
                    .pipe(
                        function=_filter_intersection_output_columns,
                        regions1_info=regions1_info,
                        regions2_info=regions2_info,
                        regions1_coord=regions1_coord,
                        regions2_coord=regions2_coord,
                        regions1_suffix=regions1_suffix,
                        regions2_suffix=regions2_suffix,
                    )
                )

                intersection_chrom_dfs_pl[chrom_and_or_strand] = ContigRanges(
                    intersection_chrom_ldf_pl.collect(), contig_ranges1._literals
                )

        return GenomicRanges(intersection_chrom_dfs_pl)


def _filter_intersection_output_columns(
    df: pl.DataFrame | pl.LazyFrame,
    regions1_info: bool,
    regions2_info: bool,
    regions1_coord: bool,
    regions2_coord: bool,
    regions1_suffix: str,
    regions2_suffix: str,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Filter intersection output columns.

    Parameters
    ----------
    df
        Polars DataFrame or LazyFrame with intersection results.
    regions1_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions2_info
        Add non-coordinate columns from first set of regions to output of intersection.
    regions1_coord
        Add coordinates from first set of regions to output of intersection.
    regions2_coord
        Add coordinates from second set of regions to output of intersection.
    regions1_suffix
        Suffix added to coordinate columns of first set of regions.
    regions2_suffix
        Suffix added to coordinate and info columns of second set of regions.

    Returns
    -------
    Polars LazyFrame with intersection results with only the requested columns.

    """
    # Get coordinate column names for first set of regions.
    regions1_coord_columns = [
        f"Chromosome{regions1_suffix}",
        f"Start{regions1_suffix}",
        f"End{regions1_suffix}",
    ]

    # Get coordinate column names for second set of regions.
    regions2_coord_columns = [
        f"Chromosome{regions2_suffix}",
        f"Start{regions2_suffix}",
        f"End{regions2_suffix}",
    ]

    # Get info column names for first set of regions
    # (all columns except coordinate columns).
    regions1_suffix_length = len(regions1_suffix)
    regions1_info_columns = [
        # Remove region1 suffix from column names.
        pl.col(column_name).alias(column_name[:-regions1_suffix_length])
        for column_name in df.columns
        if (
            column_name.endswith(regions1_suffix)
            and column_name not in regions1_coord_columns
        )
    ]

    # Get info column names for second set of regions
    # (all columns except coordinate columns).
    regions2_suffix_length = len(regions2_suffix)
    regions2_info_columns = [
        # Remove region2 suffix from column names if no region1 info will be displayed.
        pl.col(column_name)
        if regions1_info
        else pl.col(column_name).alias(column_name[:-regions2_suffix_length])
        for column_name in df.columns
        if (
            column_name.endswith(regions2_suffix)
            and column_name not in regions2_coord_columns
        )
    ]

    select_columns = [pl.col(["Chromosome", "Start", "End"])]
    if regions1_coord:
        select_columns.append(pl.col(regions1_coord_columns))
    if regions2_coord:
        select_columns.append(pl.col(regions2_coord_columns))

    if regions1_info:
        select_columns.extend(regions1_info_columns)
    if regions2_info:
        select_columns.extend(regions2_info_columns)

    return df.select(select_columns)
