# pylint: disable=import-error, unused-import, missing-docstring
from ocrd.processor.base import run_processor
from ocrd_utils import MIMETYPE_PAGE

from ocrd_froc.processor import FROCProcessor

def test_froc(processor_kwargs):
    ws  = processor_kwargs['workspace']
    pagexml_before = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
    run_processor(
        FROCProcessor,
        input_file_grp='OCR-D-IMG',
        output_file_grp='FROC',
        parameter={},
        **processor_kwargs
    )
    ws.reload_mets()
    pagexml_after = len(list(ws.mets.find_files(mimetype=MIMETYPE_PAGE)))
    assert 0, next(ws.mets.find_files(fileGrp='FROC'))
