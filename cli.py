from pathlib import Path

import typer
import sys
import os
from enum import IntEnum
import logging

from rich.logging import RichHandler
import absl.logging

from typing_extensions import Annotated

from index.index import Indexer

sys.path.insert(1, os.path.join(sys.path[0], "../.."))


app = typer.Typer(help="CLI for preprocessing the dataset and  train  triplet_network")


class VerboseMode(IntEnum):
    """
    Enum for logging verbosity
    """

    WARNING = 0
    INFO = 1
    DEBUG = 2

    def log_level(self) -> str:
        """
        Returns the string representation of the enum
        """
        return {0: "WARNING", 1: "INFO", 2: "DEBUG"}.get(self.value, "INFO")


@app.command()
def index_white_addresses(
        save_path: Annotated[
            Path,
            typer.Option(
                help="Abs path to save dataset"
            )
        ],
        start_block: Annotated[
            int,
            typer.Option(
                help="Block number to start indexing"
            )
        ],
        end_block: Annotated[
            int, typer.Option(
                help="Block number to end indexing"
            )
        ],
        step: Annotated[
            int,
            typer.Option(
                help="Step number to start indexing"
            )
        ],
        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Insert new transactions in dataset
    ### Args:
        See the help for each argument
    """

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        indexer = Indexer()
        indexer.index_white(save_path, list(range(start_block, end_block, step)), 10)

    except Exception as e:
        logger.exception("Error while indexing new transactions. Aborting...")
        return

    logger.info("Done! Exiting...")


@app.command()
def index_marked_addresses(
        save_path: Annotated[
            Path,
            typer.Option(
                help="Abs path to save dataset"
            )
        ],
        csv_path: Annotated[
            Path,
            typer.Option(
                help="Path of csv file with all black addresses"
            )
        ],

        verbose: Annotated[
            int,
            typer.Option(
                help="Whether to print the logs. 0 to set WARNING level only, 1 for INFO, 2 for showing triplet_network summary and debug"
            ),
        ] = 1,
) -> None:
    """
    Insert new transactions in dataset
    ### Args:
        See the help for each argument
    """

    logger = _get_logger(level=VerboseMode(verbose).log_level())

    try:
        indexer = Indexer()
        indexer.index_black_addresses(save_path, csv_path)

    except Exception as e:
        logger.exception("Error while indexing new transactions. Aborting...")
        return

    logger.info("Done! Exiting...")


def _get_logger(level="INFO") -> logging.Logger:
    """
    Configures the logging, and returns the logger instance

    ### Args:
    - level (str, optional): Logging level. Defaults to 'INFO'.

    ### Returns:
        logging.Logger: Instance of the logger
    """

    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    absl.logging.set_verbosity(absl.logging.ERROR)  # Disabling the TensorFlow warnings

    return logging.getLogger("rich")


if __name__ == "__main__":
    app()