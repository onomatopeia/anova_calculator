import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from goggles import assumptions, parametric

now = datetime.now().strftime("%Y%m%d_%H%M%S")

logger = logging.getLogger("colour")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'results/experience_{now}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def evaluate(df, experience_column, results_folder, columns):
    df_experienced = df.loc[df[experience_column] == 'Tak']
    df_inexperienced = df.loc[df[experience_column] == 'Nie']

    for col in columns:
        logger.debug(f"Variable: {col}")
        output_folder = results_folder.joinpath(col)
        output_folder.mkdir(exist_ok=True, parents=True)
        exp = df_experienced[col].dropna()
        inexp = df_inexperienced[col].dropna()
        equal_var = assumptions.equal_variances(exp, inexp)
        logger.debug(f'Experienced mean: {exp.mean()}')
        logger.debug(f'Inexperienced mean: {inexp.mean()}')
        parametric.paired_t_test(exp, inexp, equal_var)
        logger.debug("--------------------------------------------------------------------------\n")


if __name__ == '__main__':
    experience_column = "experience"
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
    data_file_path = Path(__file__).parent.joinpath('data', '20241229', 'experience.ods')

    df = pd.read_excel(
        data_file_path,
        sheet_name="TFD_TYR_ALL",
        usecols=range(1, 9),
        engine='odf',
    )
    results_folder = Path(__file__).parents[1].joinpath('results', "TFD_experience")
    logger.debug("TFD")
    evaluate(df, experience_column, results_folder, columns)

    df = pd.read_excel(
        data_file_path,
        sheet_name="TTFF_TYR_ALL",
        usecols=range(1, 9),
        engine='odf',
    )
    results_folder = Path(__file__).parents[1].joinpath('results', "TTFF_experience")
    logger.debug("TTFF")
    evaluate(df, experience_column, results_folder, columns)
