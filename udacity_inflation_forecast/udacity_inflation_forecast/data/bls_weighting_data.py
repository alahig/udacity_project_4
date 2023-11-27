# Import the zipfile module
import zipfile
import shutil
import os
import re
import pandas as pd
import requests


class CPIWeights(object):
    """Class to load the CPI weights from BLS
    The class takes care of downloading the data from the BLS homepage and parsing it.
    Usage:

    folder = 'raw_data_temp'
    cpi_weigths = CPIWeights(folder)
    cpi_weights.get_weights(1960)


    """

    def __init__(self, folder="tmp", download_data=True):
        try:
            os.mkdir(folder)

        except:
            pass

        self.folder = folder
        self.valid_years = range(1952, 2023)

        if download_data:
            self._download_data()
            self._extract_zip_files()

    def _extract_zip_files(self):
        for zip_file in filter(lambda x: x.endswith(".zip"), os.listdir(self.folder)):
            extract_txt_files(os.path.join(self.folder, zip_file), self.folder)

    def _download_data(self):
        headers = {
            "Host": "www.bls.gov",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/118.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "de,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "TE": "trailers",
        }
        files_to_download = [
            "https://www.bls.gov/cpi/tables/relative-importance/2022.xlsx",
            "https://www.bls.gov/cpi/tables/relative-importance/2021.xlsx",
            "https://www.bls.gov/cpi/tables/relative-importance/2020.xlsx",
            # download data 2010-2017
            # here we are provided with zip files containing .txt files
            "https://www.bls.gov/cpi/tables/relative-importance/ri-archive-2010-2019.zip",
            "https://www.bls.gov/cpi/tables/relative-importance/ri-archive-2000-2009.zip",
            "https://www.bls.gov/cpi/tables/relative-importance/ri-archive-1990-1999.zip",
            "https://www.bls.gov/cpi/tables/relative-importance/ri-archive-1987-1989.zip",
            # older data
            # here we are Historical relative importance 1947-1986
            "https://www.bls.gov/cpi/tables/relative-importance/historical-relative-importance-1947-1986.xlsx",
        ]
        for url in files_to_download:
            response = requests.get(url, headers=headers)
            target_file = os.path.join(self.folder, url.split("/")[-1])
            open(target_file, "wb").write(response.content)

    def get_weights(self, year):
        """Returns the CPI weights for a given year

        Args:
            year (int): Year we want the CPI weights

        Returns:
            pd.Series: CPI weights
        """
        assert year in self.valid_years
        if year in [2020, 2021, 2022]:
            file = os.path.join(self.folder, f"{year}.xlsx")
            return parse_excel_file(file)

        if year in range(1997, 2020):
            file = os.path.join(self.folder, f"{year}.txt")

            return parse_txt_file_format_1(file)

        if year in range(1987, 1997):
            file = os.path.join(self.folder, f"{year}.txt")
            return parse_txt_file_format_2(file)

        if year in range(1952, 1987):
            file = os.path.join(
                self.folder, "historical-relative-importance-1947-1986.xlsx"
            )
            return parse_excel_file_old_data(
                file,
                year,
            )

    def get_weights_all_years(self):
        """Returns the weights for all available years

        Returns:
            dict: dict (key = year)-> CPI weights
        """
        w = dict()
        for i in self.valid_years:
            w[i] = self.get_weights(i)

        return w


def parse_excel_file_old_data(file, year):
    """Parses the Excel File in the Format that
       BLS used between 1952 and 1986

    Args:
        file (str): Filepath to the file to be parsed
        year (int): Target year we are searching
    Returns:
        pd.Series: Mapping category -> weights
    """

    assert year in range(1952, 1987)

    def clean_spaces(t):
        """Remove spaces"""
        t = t.strip()
        return re.sub("\s+", " ", t)

    if year in [1986]:
        a = (
            pd.read_excel(file, skiprows=2, sheet_name=f"{year}").iloc[:, :3].dropna()
        )  # .set_index(['Indent Level', 'Item and Group'])
        a = a.drop_duplicates().set_index("Item and Group").iloc[:, 0]

        return a.rename(clean_spaces)

    if year in range(1982, 1986):
        a = (
            pd.read_excel(file, skiprows=1, sheet_name=f"{year}").iloc[:, :3].dropna()
        )  # .set_index(['Indent Level', 'Item and Group'])
        a = a.drop_duplicates().set_index("Item and Group").iloc[:, 0]

        return a.rename(clean_spaces)

    if year in [1980, 1981]:
        name = "1980-1981"
        a = (
            pd.read_excel(file, skiprows=1, sheet_name=name).iloc[:, :3].dropna()
        )  # .set_index(['Indent Level', 'Item and Group'])
        a = a.drop_duplicates().set_index("Item and Group").iloc[:, year - 1980]

        return a.rename(clean_spaces)

    if year in [1977, 1978, 1979]:
        name = "1977-1979"
        a = (
            pd.read_excel(file, skiprows=1, sheet_name=name).iloc[:, :].dropna()
        )  # .set_index(['Indent Level', 'Item and Group'])
        a = a.drop_duplicates().set_index("Item and Group")
        a = a.rename(index=clean_spaces, columns=clean_spaces)
        return a[f"All Urban Consumers {year}"]

    if year in range(1961, 1977):
        if year in [1961, 1962]:  # This year is missing
            year = 1963

        name = "1961-1976"
        a = pd.read_excel(file, skiprows=1, sheet_name=name).iloc[:, :]
        a = a.dropna(subset=a.columns[:2])
        # .set_index(['Indent Level', 'Item and Group'])
        a = a.drop_duplicates().set_index("Item and Group")
        a = a.rename(index=clean_spaces, columns=clean_spaces)
        return a[f"December {year}"].dropna()

    if year in range(1952, 1961):
        name = "1952-1960"
        a = pd.read_excel(file, skiprows=1, sheet_name=name).iloc[:, :]
        a = a.dropna(subset=a.columns[:2])
        # .set_index(['Indent Level', 'Item and Group'])
        a = a.drop_duplicates().set_index("Item and Group")
        a = a.rename(index=clean_spaces, columns=clean_spaces)
        return a[f"December {year}"].dropna()


def parse_excel_file(file):
    """Parses a excel file as provided by BLS.

    Args:
        file (str): Filepath to the file to be parsed
    Returns:
        pd.Series: Mapping category -> weights
    """
    a = (
        pd.read_excel(file, skiprows=3).iloc[:, :3].dropna()
    )  # .set_index(['Indent Level', 'Item and Group'])

    a = a.drop_duplicates().set_index("Item and Group")["U.S. City Average"]
    return a


def parse_txt_file_format_2(file):
    """Parses a weighting file as provided by BLS.
        The assumed format is the one used by BLS up to (including) 1996

        The data in this type of files comes as follows:

        _____
        SAC      COMMODITIES                                    45.088    49.025
        SACE     ENERGY COMMODITIES                              3.581     4.260
        SACL1    COMMODITIES LESS FOOD                          28.770    30.910
        SACL1E   COMMODITIES LESS FOOD AND ENERGY               25.188    26.649
        SACL1E4  COMMODITIES LESS FOOD, ENERGY, & UCR           23.952    24.438
        SACL11   COMMODITIES LESS FOOD & BEVERAGES              27.223    29.257
        SAD      DURABLES                                       11.256    12.323
        SAN      NONDURABLES                                    33.832    36.702
        SANL1    NONDURABLES LESS FOOD                          17.513    18.586
        SANL11   NONDURABLES LESS FOOD AND BEVERAGES            15.967    16.934
        SANL113  NONDURABLES LESS FOOD, BVRGS, APPAREL          10.394    11.319
        _____

    Args:
        file (str): Filepath to the file to be parsed
    Returns:
        pd.Series: Mapping category -> weights
    """

    def parse_line(l):
        # Try to see if line contains weights, i.e. if line looks as follows
        # SAC      COMMODITIES                                    45.088    49.025
        l = l.strip()
        weights = re.findall("[0-9]{1,3}[.][0-9]{3}", l)
        if len(weights) == 2:
            # line contains weights
            w = float(weights[0])  # we want CPI-U

            # Remove the weights, we now search the name
            l = re.sub("[0-9]{1,3}[.][0-9]{3}", "", l)

            # The first word is the item code, which we want to strip.
            m = re.search("[A-Z,0-9]+", l)
            span = m.span()
            assert span[0] == 0
            l = l[span[1] :]
            k = l.strip()

            return (k, w)
        else:
            return None

    di = dict()
    lines = open(file).readlines()
    for l in lines:
        p = parse_line(l)
        if not (p is None):
            di[p[0]] = p[1]
    return pd.Series(di)


def parse_txt_file_format_1(file):
    """Parses a weighting file as provided by BLS.
       The assumed format is the one used by BLS since 1997

        The data in this type of files comes as follows:
        _____
     All items............................................   100.000        100.000

      Food and beverages..................................    15.256         15.940
       Food...............................................    14.308         15.071

        Food at home......................................     8.638          9.460
         Cereals and bakery products......................     1.242          1.338
          Cereals and cereal products.....................      .482           .533

    _____


        Args:
            file (str): Filepath to the file to be parsed

        Returns:
            pd.Series: Mapping category -> weights
    """
    with open(file) as f:
        t = f.read()
    names = re.findall("\n[ ,A-Z,a-z,']+[.]+[ ]+[0-9,.]+", t)
    di = dict()
    for n in names:
        n_ = re.sub("[ ]*[.]{1,}[ ]+", "/", n)

        sp = n_.split("/")
        assert len(sp) == 2
        k = sp[0].strip()
        v = float(sp[1])

        di[k] = v

    return pd.Series(di)


def extract_txt_files(zip_file, dest_folder):
    """The BLS delivers some data as ZIP file
    containing subfolders.
    The folders do not only contain the cpi weights
    in .txt format but also other information, that we do not need.
    This function extracts all the .txt files (the ones we are interested in)


    Args:
        zip_file (str): filepath to the zip file
        dest_folder (str): filepath destination folder
    """
    # Open the zip file in read mode
    with zipfile.ZipFile(zip_file, "r") as z:
        # Loop through the file names in the zip file

        for file_name in z.namelist():
            # Check if the file name ends with .txt
            if file_name.endswith(".txt"):
                # Extract the file to the destination folder
                z.extract(file_name, dest_folder)
                # Move the file out of the nested structure used by BLS
                shutil.move(
                    os.path.join(dest_folder, file_name),
                    os.path.join(dest_folder, file_name.split("/")[-1]),
                )
