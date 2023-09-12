import polars as pl
import pyranges as pr


class GenomicRanges:
    def __init__(self, obj):
        if isinstance(obj, dict):
            # TODO: check the dict values are all pl.DataFrame
            dfs = obj.dfs
        elif isinstance(obj, pr.PyRanges):
            # with pl.StringCache():
            dfs = {}
            for key, val in obj.dfs.items():
                dfs[key] = pl.from_pandas(val)
        else:
            raise ValueError(
                "`GenomicRanges` expects either a dictionary of polars "
                "DataFrames or a PyRanges object as input."
            )
        self.dfs = dfs

    def _conc_dfs(self, **kwargs):
        # return pl.concat(self.dfs.values(), **kwargs)
        with pl.StringCache():
            return pl.concat(pl.from_arrow(df.to_arrow()) for df in self.dfs.values())

    def __str__(self):
        # NOTE: this cats the dfs for now and we should find something more efficient
        comb = self._conc_dfs()
        return comb.__str__()

    def __repr__(self):
        # NOTE: this cats the dfs for now and we should find something more efficient
        comb = self._conc_dfs()
        return comb.__repr__()

    def to_pyranges(self):
        """
        Create PyRanges object.

        Returns
        -------
        PyRanges DataFrame.

        Examples
        --------
        # TODO: fix examples to work with class
        Read BED file to Polars DataFrame with pyarrow engine.

        >>> bed_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow")

        Create PyRanges object directly from Polars DataFrame.

        >>> bed_df_pr = create_pyranges_from_polars_df(bed_df_pl=bed_df_pl)

        """
        # Calling the PyRanges init function with a Pandas DataFrame causes too much
        # overhead as it will create categorical columns for Chromosome and Strand columns,
        # even if they are already categorical. It will also create a Pandas DataFrame per
        # chromosome-strand (stranded) combination or a Pandas DataFrame per chromosome
        # (unstranded). So instead, create the PyRanges object manually with the use of
        # Polars and pyarrow.

        # Create empty PyRanges object, which will be populated later.
        df_pr = pr.PyRanges()

        # Check if there is a "Strand" column with only "+" and/or "-"
        is_stranded = (
            set(bed_df_pl.get_column("Strand").unique().to_list()).issubset({"+", "-"})
            if "Strand" in bed_df_pl
            else False
        )

        # Create PyArrow schema for Polars DataFrame, where categorical columns are cast
        # from pa.dictionary(pa.uint32(), pa.large_string())
        # to pa.dictionary(pa.int32(), pa.large_string())
        # as for the later conversion to a Pandas DataFrame, only the latter is supported
        # by pyarrow.
        pa_schema_fixed_categoricals_list = []
        for pa_field in bed_df_pl.head(1).to_arrow().schema:
            if pa_field.type == pa.dictionary(pa.uint32(), pa.large_string()):
                # ArrowTypeError: Converting unsigned dictionary indices to Pandas not yet
                # supported, index type: uint32
                pa_schema_fixed_categoricals_list.append(
                    pa.field(
                        pa_field.name, pa.dictionary(pa.int32(), pa.large_string())
                    )
                )
            else:
                pa_schema_fixed_categoricals_list.append(
                    pa.field(pa_field.name, pa_field.type)
                )

        # Add entry for index as last column.
        pa_schema_fixed_categoricals_list.append(
            pa.field("__index_level_0__", pa.int64())
        )

        # Create pyarrow schema so categorical columns in chromosome-strand Polars
        # DataFrames or chromosome Polars DataFrames can be cast to a pyarrow supported
        # dictionary type, which can be converted to a Pandas categorical.
        pa_schema_fixed_categoricals = pa.schema(pa_schema_fixed_categoricals_list)

        # Add (row) index column to Polars DataFrame with BED entries so original row
        # indexes of BED entries can be tracked by PyRanges (not sure if pyranges uses
        # those index values or not).
        bed_with_idx_df_pl = (
            bed_df_pl
            # Add index column and cast it from UInt32 to Int64
            .with_row_count("__index_level_0__").with_columns(
                pl.col("__index_level_0__").cast(pl.Int64)
            )
            # Put index column as last column.
            .select(pl.col(pa_schema_fixed_categoricals.names))
        )
