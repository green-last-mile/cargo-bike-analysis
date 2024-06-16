from typing import List

import pandas as pd
import polars as pl
import polars.selectors as psl


def dataframe_to_pandas(
    results: pl.DataFrame, index_cols: List[str,], bold_min_group: List[str]
) -> pd.DataFrame:
    drop_grouper = False
    if not bold_min_group:
        results = results.with_columns(pl.lit(1).alias("grouper"))
        bold_min_group = ["grouper"]

        drop_grouper = True

    return (
        results.with_columns(
            (pl.col("mean_coverage") - 90)
            .abs()
            .rank()
            .over(bold_min_group)
            .alias("mean_coverage_rank")
        )
        .select(
            *(
                pl.concat_str(
                    pl.concat_str(
                        [
                            pl.col(col).round(1).cast(str),
                            pl.col(f"{col}_std").round(1).cast(str),
                        ],
                        separator="\pm",
                    ),
                ).alias(col)
                for col in results.columns
                if (col not in index_cols)
                and ("std" not in col)
                and (col not in bold_min_group)
            ),
            *index_cols,
            *set(bold_min_group).difference(set(index_cols)),
            "mean_coverage_rank",
        )
        .pipe(
            lambda df: df.with_columns(
                pl.when((pl.col(col) == pl.col(col).min()) & (~pl.col("baseline")))
                .then(
                    pl.concat_str(
                        pl.lit("$"),
                        pl.lit("\\mathbf{"),
                        pl.col(col),
                        pl.lit("}"),
                        pl.lit("$"),
                    )
                )
                .otherwise(
                    pl.when((pl.col(col) == pl.col(col).min()))
                    .then(
                        pl.concat_str(
                            pl.lit("\\underline{$"),
                            pl.col(col).str.strip_chars("$"),
                            pl.lit("$}"),
                        )
                    )
                    .otherwise(
                        pl.concat_str(
                            pl.lit("$"),
                            pl.col(col),
                            pl.lit("$"),
                        )
                    )
                    # pl.concat_str(
                    #     pl.lit("$"),
                    #     pl.col(col),
                    #     pl.lit("$"),
                    # )
                )
                .over(bold_min_group)
                .alias(col)
                if col
                not in [
                    *index_cols,
                    "mean_coverage_rank",
                    "mean_coverage",
                    *bold_min_group,
                ]
                else pl.col(col)
                for col in df.columns
            )
        )
        .with_columns(
            pl.when((pl.col("mean_coverage_rank") == 1) & (~pl.col("baseline")))
            .then(
                pl.concat_str(
                    pl.lit("$\\mathbf{"),
                    pl.col("mean_coverage").str.strip_chars("$"),
                    pl.lit("}$"),
                )
            )
            .otherwise(
                pl.when(pl.col("mean_coverage_rank") == 1)
                .then(
                    pl.concat_str(
                        pl.lit("\\underline{$"),
                        pl.col("mean_coverage").str.strip_chars("$"),
                        pl.lit("$}"),
                    )
                )
                .otherwise(
                    pl.concat_str(
                        pl.lit("$"),
                        pl.col("mean_coverage"),
                        pl.lit("$"),
                    )
                )
            )
            .alias("mean_coverage")
        )
        .drop("mean_coverage_rank")
        .rename({"mean_coverage": "coverage", "mean_interval_width": "interval_width"})
        .drop("MAE_0.5", *(bold_min_group if drop_grouper else []))
        .to_pandas()
        .set_index(index_cols)
        .sort_index()
    )


def prettify_column_names(df: pd.DataFrame) -> pd.DataFrame:
    def pretty_col(name, percent=None, *args, **kwargs):
        if "coverage" in name:
            return ("Mean Interval", "Coverage (\%)")
        elif "interval" in name:
            return ("Mean Interval", "Width (s)")
        elif "crps" in name:
            return ("CRPS", "(s)")
        else:
            return (
                f"{name.title()}",
                "$P_{" + f"{int(float(percent) * 100):02d}" + "}$ (s)",
            )

    # create a multiindex for the columns
    df.columns = pd.MultiIndex.from_tuples(
        list(map(lambda x: tuple(pretty_col(*x.split("_"))), df.columns))
    )

    return (
        df
        
    )


def print_latex(df: pd.DataFrame) -> None:
    # print(
    print(
        df
        # .sort_index(
        #     axis=1,
        # )
        # # filter just for cities in ["Boston", "Seattle"]
        # .sort_index(axis=0, ascending=[True, False, True])
        
        .to_latex(
            escape=False,
            # multirow=True,
            multicolumn_format="c",
            # column_format="l",
            float_format="{:0.1f}",
            na_rep="-",
            sparsify=True,
            # highlight the min
            # highlight_min=True,
            # drop the 2nd level of the index
        )
    )
    # )
