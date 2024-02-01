"""
Wrap FROC as an ocrd.Processor
"""
import os

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_utils import getLogger, make_file_id, MIMETYPE_PAGE
from ocrd_models.ocrd_page import (
    to_xml,
    TextStyleType,
    TextEquivType,
)
from json import loads
from ocrd_utils import resource_filename, resource_string
from ocrd_modelfactory import page_from_file
from .froc import Froc

OCRD_TOOL = loads(resource_string(__name__, 'ocrd-tool.json'))

class FROCProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-froc-recognize']
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):

        if 'network' not in self.parameter:
            self.parameter['network'] = str(resource_filename(f'ocrd_froc.models', 'default.froc'))

        network_file = self.resolve_resource(self.parameter['network'])
        self.froc = Froc.load(network_file)

    def _process_segment(self, segment, image):
        textStyle = segment.get_TextStyle()
        if textStyle and self.parameter['replace_textstyle']:
            textStyle = None
            segment.set_TextStyle(textStyle)
        if not textStyle:
            textStyle = TextStyleType()
            segment.set_TextStyle(textStyle)

        ocr_method = self.parameter['ocr_method']

        result = {}

        if ocr_method != 'COCR':

            result = self.froc.classify(image)
            classification_result = ''

            font_class_priors = self.parameter['font_class_priors']
            output_font = True

            if font_class_priors:
                if 'other' in font_class_priors:
                    for typegroup in self.froc.classMap.cl2id:
                        result[typegroup] = 0
                    result['all'] = 1
                    output_font = False
                else:
                    result_sum = 0
                    for typegroup in self.froc.classMap.cl2id:
                        if typegroup not in font_class_priors:
                            result[typegroup] = 0
                        else:
                            result_sum += result[typegroup]
                    if result_sum == 0:
                        result['all'] = 1
                        output_font = False
                    else:
                        for typegroup in self.froc.classMap.cl2id:
                            result[typegroup] /= result_sum

            for typegroup in self.froc.classMap.cl2id:
                score = result[typegroup]
                score = round(100 * score)
                if score <= 0:
                    continue
                if classification_result != '':
                    classification_result += ', '
                classification_result += '%s:%d' % (typegroup, score)

            if output_font:
                textStyle.set_fontFamily(classification_result)


        if ocr_method == 'COCR':
            fast_cocr = self.parameter['fast_cocr']
            transcription, score = self.froc.run(image,
                                          method=ocr_method,
                                          fast_cocr=fast_cocr)
        elif ocr_method == 'SelOCR':
            transcription, score = self.froc.run(image,
                                          method=ocr_method,
                                          classification_result=result)
        else:
            fast_cocr = self.parameter['fast_cocr']
            adaptive_treshold = self.parameter['adaptive_treshold']
            transcription, score = self.froc.run(image,
                                          method=ocr_method,
                                          classification_result=result,
                                          fast_cocr=fast_cocr,
                                          adaptive_treshold=adaptive_treshold)
        segment.set_TextEquiv([TextEquivType(Unicode=transcription, conf=score)])


    def process(self):  # type: ignore
        """Perform font classification and text recognition (in one step) on historic documents.

        Open and deserialize PAGE input files and their respective images,
        iterating over the element hierarchy down to the text line level.

        Then for each line, retrieve the raw image and feed it to the font
        classifier (optionally) and the OCR.

        Annotate font predictions by name and score as a comma-separated
        list under ``./TextStyle/@fontFamily``, if any.

        Annotate the text prediction as a string under ``./TextEquiv``.

        If ``method`` is `adaptive`, then use `SelOCR` if font classification is confident
        enough, otherwise use `COCR`.

        Finally, produce a new PAGE output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('ocrd_froc')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info('Processing: %d / %s', n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            page_image, page_coords, _ = self.workspace.image_from_page(
                page, page_id,
                # prefer raw image (to meet expectation of the models, which
                # have been trained on RGB images with both geometry and color
                # transform random augmentation)
                # maybe even: dewarped,deskewed ?
                feature_filter='binarized,normalized,grayscale_normalized,despeckled')

            for line in page.get_AllTextLines():
                line_image, line_coords = self.workspace.image_from_segment(
                    line, page_image, page_coords,
                    feature_filter='binarized,normalized,grayscale_normalized,despeckled')
                self._process_segment(line, line_image)

            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts)
            )
