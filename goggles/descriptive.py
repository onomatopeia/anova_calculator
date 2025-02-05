import logging

import pandas as pd

logger = logging.getLogger("colour")


def describe(samples: dict[str, pd.Series]) -> None:
    for group_name, group in samples.items():
        logger.debug(group_name)
        logger.debug(f'[{group.min()}, {group.max()}]')
        logger.debug(f'{group.median()}')
        logger.debug(f'{group.mean()} +- {group.std()}')
