from typing import Literal
from pathlib import Path

import gzip
import os
import pyarrow as pa
import pyarrow.csv
import polars as pl


def _normalise_filepath(path: str | Path, check_not_directory: bool = True) -> str:
    """
    Create a string path, expanding the home directory if present.

    """
    path = os.path.expanduser(path)
    if check_not_directory and os.path.exists(path) and os.path.isdir(path):
        raise IsADirectoryError(f"Expected a file path; {path!r} is a directory")
    return path


def read_bed_to_polars_df(
    bed_filename: str,
    engine: str | Literal["polars"] | Literal["pyarrow"] = "pyarrow",
    min_column_count: int = 3,
) -> pl.DataFrame:
    """
    Read BED file to a Polars DataFrame.

    Parameters
    ----------
    bed_filename
        BED filename.
    engine
        Use Polars or pyarrow to read the BED file (default: `pyarrow`).
    min_column_count
        Minimum number of required columns needed in BED file.

    Returns
    -------
    Polars DataFrame with BED entries.

    See Also
    --------
    pycisTopic.fragments.read_fragments_to_polars_df

    Examples
    --------
    Read BED file to Polars DataFrame with pyarrow engine.

    >>> bed_df_pl = read_bed_to_polars_df("test.bed", engine="pyarrow")

    Read BED file to Polars DataFrame with pyarrow engine and require that the BED
    file has at least 4 columns.

    >>> bed_with_at_least_4_columns_df_pl = read_bed_to_polars_df(
    ...     "test.bed",
    ...     engine="pyarrow",
    ...     min_column_count=4,
    ... )

    """
    bed_column_names = (
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    )

    bed_filename = _normalise_filepath(bed_filename)

    # Set the correct open function, depending upon if the fragments BED file is gzip
    # compressed or not.
    open_fn = gzip.open if bed_filename.endswith(".gz") else open

    skip_rows = 0
    column_count = 0
    with open_fn(bed_filename, "rt") as bed_fh:
        for line in bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment
                # before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                column_count = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if column_count < min_column_count:
        raise ValueError(
            f"BED file needs to have at least {min_column_count} columns. "
            f'"{bed_filename}" contains only {column_count} columns.'
        )

    if engine == "polars":
        # Read BED file with Polars.
        bed_df_pl = pl.read_csv(
            bed_filename,
            has_header=False,
            skip_rows=skip_rows,
            separator="\t",
            use_pyarrow=False,
            new_columns=bed_column_names[:column_count],
            dtypes={
                bed_column: dtype
                for bed_column, dtype in {
                    "Chromosome": pl.Categorical,
                    "Start": pl.Int32,
                    "End": pl.Int32,
                    "Name": pl.Categorical,
                    "Strand": pl.Categorical,
                }.items()
                if bed_column in bed_column_names[:column_count]
            },
        )
    elif engine == "pyarrow":
        # Read BED file with pyarrow.
        bed_df_pl = pl.from_arrow(
            pa.csv.read_csv(
                bed_filename,
                read_options=pa.csv.ReadOptions(
                    use_threads=True,
                    skip_rows=skip_rows,
                    column_names=bed_column_names[:column_count],
                ),
                parse_options=pa.csv.ParseOptions(
                    delimiter="\t",
                    quote_char=False,
                    escape_char=False,
                    newlines_in_values=False,
                ),
                convert_options=pa.csv.ConvertOptions(
                    column_types={
                        "Chromosome": pa.dictionary(pa.int32(), pa.large_string()),
                        "Start": pa.int32(),
                        "End": pa.int32(),
                        "Name": pa.dictionary(pa.int32(), pa.large_string()),
                        "Strand": pa.dictionary(pa.int32(), pa.large_string()),
                    },
                ),
            )
        )
    else:
        raise ValueError(
            f'Unsupported engine value "{engine}" (allowed: ["polars", "pyarrow"]).'
        )

    return bed_df_pl
