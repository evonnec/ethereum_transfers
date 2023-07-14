"""
## Problem Statement

Use the Ethereum Logs data below to find the most transferred ERC-20 token per day. 
A list of ERC-20 tokens is also provided.

Specifically, measure transfer volume with two metrics: 
(1) count of transfers and (2) sum of transfer amounts.

The resulting dataset should be two CSVs, one for each metric, each with 3 columns 
(day, token address, transfer metric), ordered by transfer metric descending, limit 100.

Data Contents

- Ethereum Decoded Logs (Parquet)
- ERC-20 Token List (csv)
"""

import pandas as pd
import json
import logging

# BLOCK_DATE is transaction date
# sample_output1 = ["ADDRESS", "BLOCK_DATE", "count_of_transfers"]
# sample_output2 = ["ADDRESS", "BLOCK_DATE", "quantity_of_transfers"]


# if transaction addresss exists in tokens, this is included in the metric, else
# next transaction


def write_new_output(input_file: str, output_file: str) -> None:
    """
    make a smaller file for testing purposes.
    """
    df = pd.read_parquet(input_file)
    df.head(1000000).to_parquet(output_file, engine="pyarrow", compression="gzip")


def count_of_transfers(df: pd.DataFrame, metric_type: str) -> None:
    # do a count based on the number of transfers and create a new column, cast
    # group by object to dataframe
    count_df = df \
        .groupby(["ADDRESS", "BLOCK_DATE"]).size().to_frame("count_of_transfers")
    # add a row number based on count of transfers per day and order by count descending
    # group by day.
    count_df["row_num"] = count_df \
        .sort_values(by=["count_of_transfers", "BLOCK_DATE"], ascending=False) \
        .groupby("BLOCK_DATE") \
        .cumcount() + 1
    # only display the address with the highest count of transfers per day
    top_transferred_address_per_day = \
        count_df[count_df.row_num == 1].drop("row_num", axis=1)
    # order by transfer metric descending
    result = top_transferred_address_per_day.sort_values(
        by="count_of_transfers", ascending=False
    ).head(100)
    print(result)
    result.to_csv(f"{metric_type}.csv")
    logging.info("result '%s' is written", result)


def quantity_transferred(df: pd.DataFrame, metric_type: str) -> None:
    # select specific columns from the dataframe
    df = df.loc[:, ("ADDRESS", "BLOCK_DATE", "PARAMS")]
    # extract the value from the PARAMS column, rename the column
    # the other columns, we don't see what amount might be transferred
    # PARAMS column is the only column after filtering down that shows
    # something we can perhaps sum, in the json we see from, to, and `value`` keys
    # first we need to take the string and json load the string in order to pull `value``
    # https://medium.com/@eiki1212/ethereum-transaction-structure-explained-aa5a94182ad6
    df["PARAMS"] = df["PARAMS"][1]
    df["PARAMS"] = df["PARAMS"].apply(json.loads)
    df["value_transferred"] = df.PARAMS.apply(lambda x: x["value"])
    # sum the value transferred per day per address
    df["quantity_transferred"] = df \
        .groupby(["ADDRESS", "BLOCK_DATE"])["value_transferred"].transform("sum")
    # add a row number based on quantity transferred per day
    # and order by quantity transferred per day descending
    df["row_num"] = df \
        .sort_values(by=["BLOCK_DATE", "quantity_transferred"], ascending=False) \
        .groupby("BLOCK_DATE") \
        .cumcount() + 1

    # only display the address with the highest quantity transferred per day
    top_transferred_address_per_day = df[df.row_num == 1] \
        .drop("row_num", axis=1) \
        .drop("PARAMS", axis=1) \
        .drop("value_transferred", axis=1)
    # order by transfer metric descending
    result = top_transferred_address_per_day.sort_values(
        by="quantity_transferred", ascending=False
    ).head(100)
    print(result)
    result.to_csv(f"{metric_type}.csv", index=False)
    logging.info("result '%s' is written", result)


def generate_output(input_file: str, check_file: str, metric_type: str) -> None:
    """
    Generate the output file for the given input file and metric type.

    Parameters
    ----------
    input_file : str
        Path to the input file.
    metric_type : str
        Type of the metric to be generated.
    """
    df = pd.read_parquet(path=input_file)
    tokens = pd.read_csv(filepath_or_buffer=check_file).dropna(
        subset=["name", "symbol"], ignore_index=True
    )
    tokens.rename(columns={"address": "ADDRESS"}, inplace=True)
    # remove any rows with no token symbol or name: goes from 214867 to 196715 rows
    # rename column or drop duplicate column.

    df = df.merge(tokens, how="left", on="ADDRESS").dropna(
        subset=["ADDRESS", "PARAMS", "BLOCK_DATE", "name"], ignore_index=True
    )  # drop any rows that do not have a join of tokens.csv by dropping 
    # anything that does not have a name in tokens.csv

    # Be sure to have the three columns filled in.
    # Does `name` in tokens.csv have to contain `ERC-20` string to ID as token?
    # Or is each `name` in tokens.csv an ERC-20 token?
    # Assumption: each item in tokens.csv is an ERC-20 token that has a name and symbol.
    # Add a count to the df -- a count based on the number of transfers
    if metric_type == "count_of_transfers":
        count_of_transfers(df=df, metric_type=metric_type)
    elif metric_type == "quantity_transferred":
        quantity_transferred(df=df, metric_type=metric_type)
    else:
        raise ValueError(f"Invalid metric type: {metric_type}")

# make argparse arguments for these
INPUT = "eth_logs_decoded_2023_sample"
OUTPUT = "eth_logs_decoded_2023_sample_100.gzip"
VERIFY = "tokens.csv"
METRIC_TYPES = ["count_of_transfers", "quantity_transferred"]

## make a smaller file
# write_new_output(input_file=INPUT, output_file=OUTPUT)

## inspect parquet file
# import pathlib
# inspect_df = pd.read_parquet(path=pathlib.Path(OUTPUT))
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# print(inspect_df[['ADDRESS', 'PARAMS', 'BLOCK_DATE']].head(5))
# print(len(inspect_df))
# pd.reset_option("display.max_columns", "display.max_colwidth")

if __name__ == "__main__":
    # count of transfers
    generate_output(
        input_file=OUTPUT,
        check_file=VERIFY,
        metric_type=METRIC_TYPES[0],
    )

    # quantity transferred
    generate_output(
        input_file=OUTPUT,
        check_file=VERIFY,
        metric_type=METRIC_TYPES[1],
    )
