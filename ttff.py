import logging
from pathlib import Path

from goggles import evaluate_differences_in_means

logger = logging.getLogger("colour")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('results/ttff.log')
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
    factor = 'TTFF'

    sheets = {
        'Transparent': "TTFF_T_Time_to_First_Fixation",
        'Yellow': "TTFF_Y_Time_to_First_Fixation",
        'Red': "TTFF_R_Time_to_First_Fixation",
    }

    evaluate_differences_in_means(data_file_path, columns, factor, sheets)
