from typing import NamedTuple

import numpy as np
import polars as pl
from ncls import NCLS


class Coordinates(NamedTuple):
    starts: np.ndarray
    ends: np.ndarray
    indexes: np.ndarray


class ContigRanges:
    def __init__(self, df: pl.DataFrame, literals: dict | None) -> None:
        assert isinstance(literals, dict)
        assert isinstance(df, pl.DataFrame)
        self._literals = literals
        self._df = df
        self._coordinates: Coordinates | None = None
        self._ncls: NCLS | None = None

    @property
    def df(self) -> pl.DataFrame:
        return self._df.with_columns(
            **{
                key: pl.lit(val, dtype=pl.Categorical)
                for key, val in self._literals.items()
            }
        ).select([*self._literals.keys(), *self._df.columns])

    @property
    def length(self):
        return self._df.height

    @property
    def starts(self) -> np.ndarray:
        return self.coordinates.starts

    @property
    def ends(self) -> np.ndarray:
        return self.coordinates.ends

    @property
    def indexes(self) -> np.ndarray:
        return self.coordinates.indexes

    @property
    def coordinates(self) -> Coordinates:
        """
        Get start, end and index positions from per chromosome Polars dataframe.

        Returns
        -------
        (starts, ends, indexes)
            Tuple of numpy arrays with starts, ends and index positions
            for the requested chromosome.

        """
        if self._coordinates is None:
            self._coordinates = Coordinates(
                *self._df.with_row_count()
                .select(
                    [
                        pl.col("Start").cast(pl.Int64),
                        pl.col("End").cast(pl.Int64),
                        pl.col("row_nr").cast(pl.Int64),
                    ]
                )
                .to_numpy()
                .T
            )
        return self._coordinates

    @property
    def ncls(self) -> None:
        if not self._ncls:
            self._ncls = NCLS(self.starts, self.ends, self.indexes)
        return self._ncls

    def __str__(self) -> str:
        return self.df.__str__()

    def __repr__(self) -> str:
        return self.df.__repr__()
