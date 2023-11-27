import os
from dotenv import load_dotenv
import pandas as pd
import requests
import numpy as np
from cache_to_disk import cache_to_disk
import time

load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if FRED_KEY == "insert_key_here":
    raise Exception(
        "Data download requires an API key. Add the FRED API key submitted in the observations of the project submission to the .env file"
    )


# Example Url: https://api.stlouisfed.org/fred/series?series_id={series_id}&api_key={FRED_KEY}&file_type=json
BASE_URL = "https://api.stlouisfed.org/fred/{api}?{kwargs}&file_type=json"

# The default kwargs passed to the API
DEFAULT_KWARGS = {
    "api_key": FRED_KEY,
}


@cache_to_disk(3650)
def get_series_obs(series_id):
    """Downloads the series series_id from the FRED API.
        see FredApi/get_series_obs for more details
        Data is cached


    Args:
        series_id (str): Fred Series ID

    Returns:
        pd.Series: data downloaded from FRED

    """
    fred = FredApi()
    # retry is necessary as unstable API was observed
    for i in range(5):
        try:
            time.sleep(i * 2)
            return fred.get_series_obs(series_id)
        except:
            pass
    return fred.get_series_obs(series_id)


class FredApi(object):
    """Object to communicate with the FRED API.
    The FRED API allows download of Economic data series and is well documented
    here:
    https://fred.stlouisfed.org/docs/api/fred/#API
    https://fred.stlouisfed.org/docs/api/fred/series_observations.html#aggregation_method


    Usage:
    fred = FredApi()
    series_id = "GNPCA"
    ser = fred.get_series_obs(series_id)
    ser = fred.get_series_meta(series_id)
    """

    def __init__(self):
        pass

    def _dict_to_str(self, kwargs_dict):
        """Translates a dictionary a string that can be passed
            to the FRED API

            Example:
            kwargs_dict = {'key1': 'value1',
                            'key2': 'value2'}
            -> Returns 'key1=value1&key2=value2


        Args:
            kwargs_dict (dict): dictionary key -> values

        Returns:
            str: String that can be passed to the FRED API
        """

        str = "&".join([f"{k}={v}" for k, v in kwargs_dict.items()])
        return str

    def _add_default_kwargs(self, kwargs_dict):
        """Adds the default values (DEFAULT_KWARGS) to the kwargs dictionary
             if they do not exist in the kwargs dictionary

        Args:
            kwargs_dict (dict): Dictionary with the values

        Returns
            dict: kwargs_dict but extended with DEFAULT_KWARGS
        """
        for key, default_value in DEFAULT_KWARGS.items():
            if not (key in kwargs_dict.keys()):
                kwargs_dict[key] = default_value

        return kwargs_dict

    def get_series_obs(self, series_id="GNPCA", **kwargs):
        """Downloads the series series_id from the FRED API.

            Returns a data series with index=dates
            https://fred.stlouisfed.org/docs/api/fred/series_observations.html

        Args:
            series_id (str, optional): Fred Series ID

        Returns:
            pd.Series: data downloaded from FRED

        """
        api = "series/observations"
        kwargs["series_id"] = series_id
        kwargs = self._add_default_kwargs(kwargs)
        kwargs_str = self._dict_to_str(kwargs)

        url = BASE_URL.format(api=api, kwargs=kwargs_str)
        response = requests.get(url)  #

        df = pd.DataFrame(response.json()["observations"])

        def parse_float(x):
            if isinstance(x, float):
                return x
            if x == ".":
                return np.nan
            return float(x)

        # FRED returns . for nan
        df["value"] = df["value"].apply(parse_float)

        # parse values to floats
        ser = df.set_index("date")["value"].astype(float)

        # parse dates to timestamps
        ser.rename(lambda x: pd.Timestamp(x), inplace=True)

        try:
            name = self.get_series_meta(series_id=series_id)["title"]
            ser.name = name
        except:
            pass

        return ser

    def get_series_meta(self, series_id="GNPCA", **kwargs):
        """Returns the metadata of the Series 'series_id'
           meta data is described here:
           https://fred.stlouisfed.org/docs/api/fred/series.html

        Args:
            series_id (str, optional): Fred Series ID

        Returns:
            dict: metadata downloaded from FRED
        """

        api = "series"
        kwargs["series_id"] = series_id
        kwargs = self._add_default_kwargs(kwargs)
        kwargs_str = self._dict_to_str(kwargs)

        url = BASE_URL.format(api=api, kwargs=kwargs_str)
        print(url)
        response = requests.get(url)  #

        return response.json()["seriess"][0]
