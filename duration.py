import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from goggles import analysis_of_variance

now = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger("colour")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'results/duration_{now}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

if __name__ == '__main__':
    red_bucket_column = "R bucket"
    red_helmet_column = "R helmet + face"
    red_jacket_column = "R jacket"
    yellow_bucket_column = "Y bucket"
    yellow_helmet_column = "Y helmet + face"
    yellow_bag_column = "Y bag"

    columns = [
        red_jacket_column,
        red_helmet_column,
        red_bucket_column,
        yellow_bucket_column,
        yellow_bag_column,
        yellow_helmet_column,
    ]
    data_file_path = Path(__file__).parent.joinpath('data', '20241217', 'data.ods')
    factor = 'TIME'

    sheets = {
        'Transparent': "T_TIME",
        'Yellow': "Y_TIME",
        'Red': "R_TIME",
    }

    samples = {
        group_name: pd.read_excel(
            data_file_path,
            sheet_name=sheet_name,
            usecols=[2],
            engine='odf',
            skiprows=1,
        )['Time [s]']
        for group_name, sheet_name in sheets.items()
    }

    output_folder = Path(__file__).parent.joinpath('results', factor)
    output_folder.mkdir(exist_ok=True, parents=True)

    analysis_of_variance(factor, samples, output_folder, factor.lower())
    logger.debug("--------------------------------------------------------------------------\n")

