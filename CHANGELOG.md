Change Log
==========

Versioned according to [Semantic Versioning](http://semver.org/).

## Unreleased

## [1.1.0] - 2025-05-06

Changed:

  * `make docker`: Overrideable `$(DOCKER)` command and base on `latest` OCR-D/core, #22

## [1.0.1] - 2025-04-11

Fixed:

  * `Dockerfile`: Also build `ocrd-all-tool.json`, document XDG variables, #20

## [1.0.0] - 2025-04-11

Changed:

  * :fire: port processor to v3 OCR-D/core API, #19
  * switch to `src-layout`, #19

Added:

  * Unit tests, CI and CD, #19

Fixed:

  * Typos corrected, #16
  * `ocr_method="none"` actually skips OCR, ran `adaptive` before, #19
  * typo corrected that prevented adaptive OCR, #19

## [0.6.0] - 2024-04-22

Changed:

* rename parameter `replace_textstyle` -> `overwrite_style`, #8, #9
* rename parameter `network` -> `model`, #8, #9
* parameter `overwrite_text`: If true, replace all existing textequivs, else (default) just add a textequiv, #8, #9
* parameter `overwrite_style`: if true (default), replace the `fontFamily` attribute of existing textstyle or create new style if non exists., #8, #9
* parameter `min_score_style`: Score between 0 and 100, font classification results below this score will not be serialized or used for OCR, default: 0, #8, #9

Fixed:

  * Require OCR-D/core v2.64.1+ with proper support for `importlib{.,_}metadata`, #10
  * CI: Use most recent actions, #15
  * missing top-level `__init__.py`, #12

## [0.5.2] - 2024-02-01

Fixed:

  - typo in ocrd-tool.json: `ocrd-froc` -> `ocrd-froc-recognize`, again

## [0.5.1] - 2024-02-01

Fixed:

  - typo in ocrd-tool.json: `ocrd-froc` -> `ocrd-froc-recognize`

## [0.5.0] - 2024-01-30

- First release in ocrd_all

<!-- link-labels -->
[1.1.0]: ../../compare/v1.1.0...v1.0.1
[1.0.1]: ../../compare/v1.0.1...v1.0.0
[1.0.0]: ../../compare/v1.0.0...v0.6.1
[0.6.1]: ../../compare/v0.6.1...v0.6.0
[0.6.0]: ../../compare/v0.6.0...v0.5.2
[0.5.2]: ../../compare/v0.5.2...v0.5.1
[0.5.1]: ../../compare/v0.5.1...v0.5.0
[0.5.0]: ../../compare/v0.5.0...HEAD
