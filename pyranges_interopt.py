from operator import itemgetter

import pandas as pd
import polars as pl
import pyarrow as pa
import pyranges as pr

# Enable Polars global string cache so all categoricals are created with the same
# string cache.
pl.enable_string_cache(True)


def create_pyranges_from_polars_df(bed_df_pl: pl.DataFrame) -> pr.PyRanges:
    """
    Create PyRanges DataFrame from Polars DataFrame.

    Parameters
    ----------
    bed_df_pl
        Polars DataFrame containing BED entries.
        e.g.: This can also be a filtered Polars DataFrame with fragments or
              TSS annotation.

    Returns
    -------
    PyRanges DataFrame.

    See Also
    --------
    pycisTopic.fragments.filter_fragments_by_cb
    pycisTopic.fragments.read_bed_to_polars_df
    pycisTopic.fragments.read_fragments_to_polars_df
    pycisTopic.gene_annotation.change_chromosome_source_in_bed

    Examples
    --------
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
                pa.field(pa_field.name, pa.dictionary(pa.int32(), pa.large_string()))
            )
        else:
            pa_schema_fixed_categoricals_list.append(
                pa.field(pa_field.name, pa_field.type)
            )

    # Add entry for index as last column.
    pa_schema_fixed_categoricals_list.append(pa.field("__index_level_0__", pa.int64()))

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

    def create_per_chrom_or_chrom_strand_df_pd(
        per_chrom_or_chrom_strand_bed_df_pl: pl.DataFrame,
    ) -> pd.DataFrame:
        """
        Create per chromosome (unstranded) or per chromosome-strand (stranded) Pandas
        DataFrame for PyRanges from equivalent Polars DataFrame.

        Parameters
        ----------
        per_chrom_or_chrom_strand_bed_df_pl
            Polars DataFrame partitioned by chromosome (unstranded) or
            chromosome-strand (stranded).

        Returns
        -------
        Pandas DataFrame partitioned by chromosome (unstranded) or
        chromosome-strand (stranded).

        """
        # Convert per chromosome (unstranded) or per chromosome-strand (stranded)
        # Polars DataFrame with BED entries to a pyarrow table and change categoricals
        # dictionary type to Pandas compatible categorical type and convert to a
        # Pandas DataFrame.
        per_chrom_or_chrom_strand_bed_df_pd = (
            per_chrom_or_chrom_strand_bed_df_pl.to_arrow()
            .cast(pa_schema_fixed_categoricals)
            .to_pandas()
        )

        # Set Pandas index inplace and remove index name.
        per_chrom_or_chrom_strand_bed_df_pd.set_index("__index_level_0__", inplace=True)
        per_chrom_or_chrom_strand_bed_df_pd.index.name = None

        return per_chrom_or_chrom_strand_bed_df_pd

    if is_stranded:
        # Populate empty PyRanges object directly with per chromosome and strand
        # Pandas DataFrames (stranded).
        df_pr.__dict__["dfs"] = {
            chrom_strand: create_per_chrom_or_chrom_strand_df_pd(
                per_chrom_or_chrom_strand_bed_df_pl
            )
            for chrom_strand, per_chrom_or_chrom_strand_bed_df_pl in sorted(
                # Partition Polars DataFrame with BED entries per chromosome-strand
                # (stranded).
                bed_with_idx_df_pl.partition_by(
                    by=["Chromosome", "Strand"], maintain_order=False, as_dict=True
                ).items(),
                key=itemgetter(0),
            )
        }
    else:
        # Populate empty PyRanges object directly with per chromosome
        # Pandas DataFrames (unstranded).
        df_pr.__dict__["dfs"] = {
            chrom: create_per_chrom_or_chrom_strand_df_pd(per_chrom_bed_df_pl)
            for chrom, per_chrom_bed_df_pl in sorted(
                # Partition Polars DataFrame with BED entries per chromosome
                # (unstranded).
                bed_with_idx_df_pl.partition_by(
                    by=["Chromosome"], maintain_order=False, as_dict=True
                ).items(),
                key=itemgetter(0),
            )
        }

    df_pr.__dict__["features"] = pr.genomicfeatures.GenomicFeaturesMethods
    df_pr.__dict__["statistics"] = pr.statistics.StatisticsMethods

    return df_pr


def create_polars_df_from_pyranges(gr: pr.PyRanges) -> pl.DataFrame:
    """
    Create Polars DataFrame from PyRanges DataFrame.

    Parameters
    ----------
    gr
        PyRanges object.

    Returns
    -------
    Polars DataFrame.

    """
    per_chrom_or_chrom_strand_df_pd = {}
    for key, per_chrom_df_pd in gr.items():
        per_chrom_or_chrom_strand_df_pd[key] = pl.DataFrame(per_chrom_df_pd)

    return pl.concat(per_chrom_or_chrom_strand_df_pd.values(), rechunk=False)
