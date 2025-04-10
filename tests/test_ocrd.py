# pylint: disable=import-error, unused-import, missing-docstring
from ocrd.processor.base import page_from_file, run_processor
from ocrd_models.constants import NAMESPACES
from ocrd_utils import MIMETYPE_PAGE

from ocrd_froc.processor import FROCProcessor

def test_froc_defaults(processor_kwargs):
    ws  = processor_kwargs['workspace']
    run_processor(
        FROCProcessor,
        input_file_grp="OCR-D-SEG-LINE-tesseract-ocropy",
        output_file_grp='FROC',
        parameter={},
        **processor_kwargs
    )
    ws.reload_mets()
    result_page = page_from_file(next(ws.mets.find_files(ID="FROC_0001")))
    assert not result_page.etree.xpath('//page:TextEquiv', namespaces=NAMESPACES), "No OCR with ocr_method='none'"
    textstyle = result_page.etree.xpath('//page:TextStyle', namespaces=NAMESPACES)
    assert textstyle, 'TextStyle was created'
    textstyle0 = textstyle[0]
    assert textstyle0.get('fontFamily') == 'fraktur:100', 'title in fraktur'

def test_froc_ocr(processor_kwargs):
    ws  = processor_kwargs['workspace']
    # TODO pytest parameterize
    for ocr_method in ['SelOCR', 'adaptive', 'COCR']:
        file_grp = f'FROC_{ocr_method}'
        run_processor(
            FROCProcessor,
            input_file_grp="OCR-D-SEG-LINE-tesseract-ocropy",
            output_file_grp=file_grp,
            parameter={
                "ocr_method": ocr_method
            },
            **processor_kwargs
        )
        ws.reload_mets()
        result_page = page_from_file(next(ws.mets.find_files(ID=f"{file_grp}_0001")))
        assert result_page.etree.xpath('//page:TextEquiv', namespaces=NAMESPACES), "TextEquivs were created"
        text0 = result_page.etree.xpath('//page:TextEquiv/page:Unicode/text()', namespaces=NAMESPACES)[0]
        assert 'erlini' in text0, '"Berlinische Monatsschrift" or similar'
