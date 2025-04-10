"""
Wrap FROC as an ocrd.Processor
"""
from functools import cached_property
from typing import List, Optional, Tuple

from ocrd import OcrdPage, OcrdPageResult, Processor
from ocrd_models.ocrd_page import (
    TextStyleType,
    TextEquivType,
)
from ocrd_utils import resource_filename
from .froc import Froc

class FROCProcessor(Processor):

    max_workers = 1 # Torch CUDA context cannot be shared across fork mp

    @cached_property
    def executable(self):
        return 'ocrd-froc-recognize'

    @property
    def moduledir(self):
        return resource_filename(self.module, 'models')

    def setup(self):
        assert self.parameter
        model = self.resolve_resource(self.parameter['model'])
        self.froc: Froc = Froc.load(model)

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
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
        assert input_pcgts[0]
        pcgts = input_pcgts[0]
        result = OcrdPageResult(pcgts)
        page = pcgts.get_Page()
        page_image, page_coords, _ = self.workspace.image_from_page(
            page, page_id,
            # prefer raw image (to meet expectation of the models, which
            # have been trained on RGB images with both geometry and color
            # transform random augmentation)
            # maybe even: dewarped,deskewed ?
            feature_filter='binarized,normalized,grayscale_normalized,despeckled')

        for line in page.get_AllTextLines():
            line_image, _ = self.workspace.image_from_segment(
                line, page_image, page_coords,
                feature_filter='binarized,normalized,grayscale_normalized,despeckled')
            self._process_segment(line, line_image)
        return result


    def _process_segment(self, segment, image):
        assert self.parameter
        ocr_method = self.parameter['ocr_method']

        result = {}

        if ocr_method != 'COCR':

            result = self.froc.classify(image)
            fonts_detected : List[Tuple[str, float]] = []

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
                fonts_detected.append((typegroup, score))

            classification_result = ', '.join([
                f'{family}:{score}' \
                for family, score in fonts_detected \
                if score > self.parameter['min_score_style']
            ])

            if output_font:
                textStyle = segment.get_TextStyle()
                if not textStyle or self.parameter['overwrite_style']:
                    if not textStyle:
                        textStyle = TextStyleType()
                        segment.set_TextStyle(textStyle)
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
            adaptive_threshold = self.parameter['adaptive_threshold']
            transcription, score = self.froc.run(image,
                                          method=ocr_method,
                                          classification_result=result,
                                          fast_cocr=fast_cocr,
                                          adaptive_threshold=adaptive_threshold)

        if self.parameter['overwrite_text']:
            segment.set_TextEquiv([TextEquivType(Unicode=transcription, conf=score)])
        else:
            segment.add_TextEquiv(TextEquivType(Unicode=transcription, conf=score))
