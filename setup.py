# -*- coding: utf-8 -*-
import json

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()
with open('./ocrd-tool.json', 'r') as f:
    version = json.load(f)['version']

setup(
    name='ocrd_froc',
    version=version,
    description='font recognition and OCR',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Guillaume Carriere, Matthias Seuret, Konstantin Baierer',
    author_email='',
    url='https://github.com/MegaloPat/ocrd_froc',
    license='Apache License 2.0',
    #packages=find_packages(exclude=('tests', 'docs')),
    #include_package_data=True,
    install_requires=open('requirements.txt').read().split('\n'),
    package_data={
        '': ['*.json', '*.tgc'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-froc-recognize=ocrd_froc.cli.ocrd_cli:cli',
        ]
    },
)
