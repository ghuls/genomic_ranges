from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    import numpy as np
    from genomic_ranges.contig_ranges import ContigRanges


def _intersection(
    contig_ranges1: ContigRanges,
    contig_ranges2: ContigRanges,
    how: Literal["all", "containment", "first", "last"] | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get intersection between two region sets from the same contig.

    Get intersection between regions from first set and second set of regions from the
    same contig and return index positions for those overlaps in the first and second
    set of regions.

    Parameters
    ----------
    contig_ranges1
        Contig ranges for first set of regions for a certain contig.
    contig_ranges2
        Contig ranges for second set of regions for a certain contig.
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

    Returns
    -------
    (regions1_indexes, regions2_indexes)
        Tuple of indexes for regions from contig ranges Dataframe 1 and indexes for
        regions from contig ranges Dataframe 2 that have an overlap.

    """
    oncls = contig_ranges2.ncls

    if not how or how is None or how == "all":
        regions1_indexes, regions2_indexes = oncls.all_overlaps_both(
            contig_ranges1.starts, contig_ranges1.ends, contig_ranges1.indexes
        )
    elif how == "containment":
        regions1_indexes, regions2_indexes = oncls.all_containments_both(
            contig_ranges1.starts, contig_ranges1.ends, contig_ranges1.indexes
        )
    elif how == "first":
        regions1_indexes, regions2_indexes = oncls.first_overlap_both(
            contig_ranges1.starts, contig_ranges1.ends, contig_ranges1.indexes
        )
    elif how == "last":
        regions1_indexes, regions2_indexes = oncls.last_overlap_both(
            contig_ranges1.starts, contig_ranges1.ends, contig_ranges1.indexes
        )
    elif how in {"outer", "left", "right"}:
        regions1_indexes, regions2_indexes = oncls.all_overlaps_both(
            contig_ranges1.starts, contig_ranges1.ends, contig_ranges1.indexes
        )

        regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
        regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

        regions1_all_indexes = pl.arange(
            0, contig_ranges1.length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")
        regions2_all_indexes = pl.arange(
            0, contig_ranges2.length, dtype=pl.get_index_type(), eager=True
        ).alias("idx")

        regions1_missing_indexes = (
            regions1_all_indexes.to_frame()
            .join(
                regions1_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions2_missing_indexes = (
            regions2_all_indexes.to_frame()
            .join(
                regions2_indexes.to_frame(),
                on="idx",
                how="anti",
            )
            .to_series()
        )

        regions1_none_indexes = pl.repeat(
            None, regions2_missing_indexes.len(), name="idx", eager=True
        ).cast(pl.get_index_type())
        regions2_none_indexes = pl.repeat(
            None, regions1_missing_indexes.len(), name="idx", eager=True
        ).cast(pl.get_index_type())

        if how == "outer":
            regions1_indexes = pl.concat(
                [
                    regions1_indexes,
                    regions1_missing_indexes,
                    regions1_none_indexes,
                ]
            )
            regions2_indexes = pl.concat(
                [
                    regions2_indexes,
                    regions2_none_indexes,
                    regions2_missing_indexes,
                ]
            )
        elif how == "left":
            regions1_indexes = pl.concat([regions1_indexes, regions1_missing_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_none_indexes])
        elif how == "right":
            regions1_indexes = pl.concat([regions1_indexes, regions1_none_indexes])
            regions2_indexes = pl.concat([regions2_indexes, regions2_missing_indexes])

        return regions1_indexes, regions2_indexes

    regions1_indexes = pl.Series("idx", regions1_indexes, dtype=pl.get_index_type())
    regions2_indexes = pl.Series("idx", regions2_indexes, dtype=pl.get_index_type())

    return regions1_indexes, regions2_indexes
