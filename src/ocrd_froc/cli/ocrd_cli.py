"""
OCR-D conformant command line interface
"""
import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ..processor import FROCProcessor

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    """
    Classify typegroups and get transcriptions
    """
    return ocrd_cli_wrap_processor(FROCProcessor, *args, **kwargs)
