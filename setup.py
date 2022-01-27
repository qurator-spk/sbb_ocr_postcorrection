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
            'align-sequences=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:align_sequences',
            'apply-sliding-window=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:apply_sliding_window',
            'create-detector-targets=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:create_detector_targets',
            'create-ocr-json-of-single-page=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:create_ocr_json_of_single_page',
            'filter-language=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:filter_language',
            'parse-xml=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:parse_xml',
            'split-dataset=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:split_dataset',
            'split-dataset-sliding-window=qurator.sbb_ocr_postcorrection.preprocessing.cli_preproc:split_dataset_sliding_window',
            #cli_feature.py
            'create-encoding-mapping=qurator.sbb_ocr_postcorrection.feature_extraction.cli_feature:create_encoding_mapping',
            'encode-features-for-single-page=qurator.sbb_ocr_postcorrection.feature_extraction.cli_feature:encode_features_for_single_page',
            'encode-features-for-splitted-data=qurator.sbb_ocr_postcorrection.feature_extraction.cli_feature:encode_features_for_splitted_data',
            'encode-features-hack=qurator.sbb_ocr_postcorrection.feature_extraction.cli_feature:encode_features_hack',
            #cli_correct.py
            'evaluate-detector=qurator.sbb_ocr_postcorrection.mt.cli_correct:evaluate_detector',
            'evaluate-translator=qurator.sbb_ocr_postcorrection.mt.cli_correct:evaluate_translator',
            'predict-argmax-converter=qurator.sbb_ocr_postcorrection.mt.cli_correct:predict_argmax_converter',
            'predict-detector=qurator.sbb_ocr_postcorrection.mt.cli_correct:predict_detector',
            'predict-translator=qurator.sbb_ocr_postcorrection.mt.cli_correct:predict_translator',
            'reconstruct-single-page-line-boundaries=qurator.sbb_ocr_postcorrection.mt.cli_correct:reconstruct_single_page_line_boundaries',
            'run-two-step-pipeline=qurator.sbb_ocr_postcorrection.mt.cli_correct:run_two_step_pipeline',
            'run-two-step-pipeline-on-single-page=qurator.sbb_ocr_postcorrection.mt.cli_correct:run_two_step_pipeline_on_single_page',
            'train-detector=qurator.sbb_ocr_postcorrection.mt.cli_correct:train_detector',
            'train-argmax-converter=qurator.sbb_ocr_postcorrection.mt.cli_correct:train_argmax_converter',
            'train-translator=qurator.sbb_ocr_postcorrection.mt.cli_correct:train_translator'
        ]
    }
)
