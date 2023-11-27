import sys
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import requests
import re

from udacity_inflation_forecast.data.bls_weighting_data import CPIWeights
from udacity_inflation_forecast.data.fred_api import get_series_obs


def load_data(mapping_filepath):
    """

    Loads the:
    - cpi_freddata: CPI data from FRED
    - cpi_weights_merged: Weights of the CPI compoents.
    - other_vars: Other variables used in the analysis (oil..)
    - identation: Meta data on the CPI basket

    Parameters:
        mapping_filepath (str): filepath to mapping excel. This Excel was done by hand and merges the BLS to the FRED data.

    Returns:
        dict of dataframes
    """
    # Load FRED Series
    name_to_fred_cpi_baskets = load_inflation_series_fred()
    cp = CPIWeights()

    cpi_weights = cp.get_weights_all_years()

    mapping = pd.read_excel(mapping_filepath, index_col=0)
    indentation = mapping["Identation"]
    cpi_weights_merged = merge_cpi_weights(mapping=mapping, cpi_weights=cpi_weights)

    cpi_freddata = {k: get_series_obs(v) for k, v in name_to_fred_cpi_baskets.items()}
    cpi_freddata = pd.DataFrame(cpi_freddata).resample("M").last()

    other_vars = get_additional_data()

    data = {
        "cpi_freddata": cpi_freddata,
        "identation": indentation,
        "cpi_weights_merged": cpi_weights_merged / 100,
        "other_vars": other_vars,
    }
    return data


def clean_names(x):
    x = re.sub("[^A-Z,a-z, ]", "", x.lower())
    x = re.sub("[ ]+", " ", x)
    x = x.replace(" and", "").strip()
    return x


def get_additional_data():
    """Loads additonal dataseries from FRED."""
    sers = {
        # "CPI_Urban_ex_Food_Energy_Index_M_SA": "CPILFESL",
        # "CPI_Urban_All_Index_M_SA": "CPIAUCSL",
        # "Consumer_Inflation_Expectations_UNIM_12M_Pct_M_NSA": "MICH",
        # "Sticky_Price_CPI_ex_Food_Energy_Shelder_Pct_M_SA": "CRESTKCPIXSLTRM157SFRBATL",
        # "Sticky_Price_CPI_Pct_M_SA": "STICKCPIM157SFRBATL",
        "Market_Inflation_Expectations_ClevelandFED_10YR_Pct_M_NSA": "EXPINF10YR",
        "Market_Inflation_Expectations_ClevelandFED_1YR_Pct_M_NSA": "EXPINF1YR",
        "BE_10YR_Pct_D_NSA": "T10YIE",
        "BE_5YR_Pct_D_NSA": "T5YIE",
        "Consumer_Loans_Index_M_SA": "CONSUMER",
        "Loans_Leases_Index_M_SA": "LOANS",
        "JPY_Index_D_NSA": "DEXJPUS",
        "CAD_Index_D_NSA": "DEXCAUS",
        "GBP_Index_D_NSA": "DEXUSUK",
        "Real_M1_Index_M_SA": "M1REAL",
        "M1_Index_M_SA": "M1SL",
        "M2_Index_M_SA": "M2SL",
        "Real_M2_Index_M_SA": "M2REAL",
        "M3_Index_M_SA": "MABMM301USM189S",
        "WTI_Index_M_NSA": "WTISPLC",
        "Real_Borad_Effective_Exchange_Index_M_NSA": "RBUSBIS",
        "Unemployment_Rate_Pct_M_SA": "UNRATE",
        "FFER_Pct_D_NSA": "DFF",
        "10Y_Pct_D_NSA": "GS10",
        "1M_R_Pct_M_NSA": "REAINTRATREARAT1MO",
    }
    df = {k: get_series_obs(v) for k, v in sers.items()}
    df = pd.DataFrame(df).resample("M").last()
    return df


def merge_cpi_weights(mapping, cpi_weights):
    """Takes the CPI weights from BLS and mapping FRED-> BLS
       Returns a DataFrame where the BLS weights are named
       with FRED names.

    Args:
        mapping (pd.DataFrame): Mapping FRED Name -> BLS name. One FRED Name can be mapped
                                to differnt BLS names, therefore the DataFrame contains
                                several columns (BLS name 1, BLS name 2, ...)
        cpi_weights (pd.DataFrame): DataFrame containing the CPI weights from BLS

    Returns:
        cpi_weights_clean_df (pd.DataFrame): DataFrame index=FRED names, columns=years.
    """
    cpi_weights_clean_df = pd.DataFrame(index=mapping.index, columns=cpi_weights.keys())

    for y in cpi_weights_clean_df.columns:
        w = cpi_weights[y].copy()

        w = w.rename(index=clean_names)

        for x in filter(lambda x: "BLS name" in x, mapping.columns):
            m = mapping.reset_index()[["FRED Name", x]].dropna()
            m = m.applymap(clean_names)
            m = m.set_index(x)["FRED Name"]
            m += "_"

            w = w.rename(m)

        w = w.rename(lambda x: x[:-1])

        for item in cpi_weights_clean_df.index:
            try:
                a = w.loc[clean_names(item)]
                if isinstance(a, pd.Series):
                    a = a.max()
                cpi_weights_clean_df.loc[item, y] = a
            except:
                print(f"Item {item} missing in year {y}")

    return cpi_weights_clean_df


def load_inflation_series_fred():
    """Scraps the FRED homepage to search all
      the subcomponents of the CPI.
      Returns a mapping name -> FRED Ticker
      for all the inflation_components.

    Returns:
        pd.Series: Mapping name -> Fred Ticker
    """

    def clean_names(x):
        x = re.sub("[^A-Z,a-z, ]", "", x)
        x = re.sub("[ ]+", " ", x)
        return x

    url = "https://fred.stlouisfed.org/release/tables?rid=10&eid=34483&od=#"

    req = requests.get(url)
    names = re.findall('(?<= target="_blank">)[^<]+(?=</a>)', req.text)
    names = [clean_names(i) for i in names]
    tickers = re.findall('(?<=href="/series/)[^"]+(?=")', req.text)
    assert len(names) == len(tickers)

    names = [n.replace("&#039;", "'") for n in names]
    return pd.Series(index=names, data=tickers)


def save_data(database_filename, data_to_store):
    """
    Saves the dataframe to a sql lite database.
    The data is saved in different dataframe.
    Data is passed as as dictionary with keys = table_name and value=data to store
    If the table already exists it is replaced.

    Parameters:
        data_to_store (dict): dictionary with data to story (key=table names)
        database_filename (str): the name of the sql lite database file

    Returns:
        None
    """
    ### Save the clean dataset into an sqlite database.
    engine = create_engine("sqlite:///" + database_filename)
    for k, df in data_to_store.items():
        df.to_sql(k, engine, index=True, if_exists="replace")


def main():
    if len(sys.argv) == 2:
        filepath = sys.argv[1]

        mapping_filepath = os.path.join(filepath, "mapping_cpi_comp.xlsx")

        data = load_data(mapping_filepath)

        database_filepath = os.path.join(filepath, "cpi_database.db")
        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(database_filepath, data)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths filepath of the database to save the cleaned data "
        )


if __name__ == "__main__":
    main()
