from io import open
from setuptools import find_packages, setup

with open('requirements.txt') as f_in:
    install_requires = f_in.read()

setup(
    name='sbb-ocr-postcorrection',
    version='0.0.1',
    author='Robin Schaefer, The Qurator Team',
    author_email='robin.schaefer@sbb.spk-berlin.de, qurator@sbb.spk-berlin.de',
    description='An OCR Postcorrection Tool',
    long_description=open('README.md', mode='r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    keywords='qurator ocr postcorrection',
    license='Apache',
    namespace_packages=['qurator'],
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            #cli_preproc.py
            'align-sequences=qurator.sbb_ocr_correction.preprocessing.cli_preproc:align_sequences',
            'apply-sliding-window=qurator.sbb_ocr_correction.preprocessing.cli_preproc:apply_sliding_window',
            'create-detector-targets=qurator.sbb_ocr_correction.preprocessing.cli_preproc:create_detector_targets',
            'filter-language=qurator.sbb_ocr_correction.preprocessing.cli_preproc:filter_language',
            'parse-xml=qurator.sbb_ocr_correction.preprocessing.cli_preproc:parse_xml',
            'split-dataset=qurator.sbb_ocr_correction.preprocessing.cli_preproc:split_dataset',
            'split-dataset-sliding-window=qurator.sbb_ocr_correction.preprocessing.cli_preproc:split_dataset_sliding_window',
            #cli_feature.py
            'create-encoding-mapping=qurator.sbb_ocr_correction.feature_extraction.cli_feature:create_encoding_mapping',
            'encode-features=qurator.sbb_ocr_correction.feature_extraction.cli_feature:encode_features',
            'encode-features-hack=qurator.sbb_ocr_correction.feature_extraction.cli_feature:encode_features_hack',
            #cli_correct.py
            'evaluate-detector=qurator.sbb_ocr_correction.mt.cli_correct:evaluate_detector',
            'evaluate-translator=qurator.sbb_ocr_correction.mt.cli_correct:evaluate_translator',
            'predict-detector=qurator.sbb_ocr_correction.mt.cli_correct:predict_detector',
            'predict-translator=qurator.sbb_ocr_correction.mt.cli_correct:predict_translator',
            'train-detector=qurator.sbb_ocr_correction.mt.cli_correct:train_detector',
            'train-translator=qurator.sbb_ocr_correction.mt.cli_correct:train_translator'
        ]
    }
)
