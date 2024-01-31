# ocrd_froc

Perform font classification and text recognition (in one step) on historic documents.

    > Open and deserialize PAGE input files and their respective images,
    > iterating over the element hierarchy down to the text line level.

    > Then for each line, retrieve the raw image and feed it to the font
    > classifier and/or the  OCR.

    > Annotate font predictions by name and score as a comma-separated
    > list under ``./TextStyle/@fontFamily``, if any.

    > Annotate the text prediction as a string under ``./TextEquiv``.

    > If ``method`` is `adaptive`, then use `SelOCR` if font classification is confident
    > enough, otherwise use `COCR`.

    > Finally, produce a new PAGE output file by serialising the resulting hierarchy.

## Installation

## Models

### Default
The default.froc model is composed of a SelOCR network and a COCR 
architecture, and is trained to classify and OCR textlines on the following 12 classes:

- Antiqua

- Bastarda

- Fraktur

- Textura

- Schwabacher

- *Greek* \*

- Italic

- *Hebrew* \*

- Gotico-antiqua

- *Manuscript* \*

- Rotunda

- No class/Ignore

\* Greek, Hebrew and Manuscript font groups do not currently provide good 
results due to a lack of training data.


## Usage

OCR-D processor interface ocrd-froc

To be used with PAGE-XML documents in an OCR-D annotation workflow.

```
Parameters:

   "ocr_method" [string - "none"]
    The method to use for text recognition
    Possible values: ["none", "SelOCR", "COCR", "adaptive"]
   "replace_textstyle" [bool - true]
    Whether to replace existing textStyle
   "network" [string]
    The file name of the neural network to use, including sufficient path
    information. Defaults to the model bundled with ocrd_froc.
   "fast_cocr" [boolean - true]
    Whether to use optimization steps on the COCR strategy
   "adaptive_treshold" [number - 95]
    Treshold of certitude needed to use SelOCR when using the adaptive
    strategy
   "font_class_priors" [array - []]
    List of font classes which are known to be present on the data when
    using the adaptive/SelOCR strategies. When this option is specified,
    every font classes not included will be ignored. If 'other' is
    included in the list, font classification will not be outputted and
    a generic model will be used for transcriptions.
```
