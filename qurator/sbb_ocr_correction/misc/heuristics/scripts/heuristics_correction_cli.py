import click

from data import load_fulltexts
from dictionaries import combine_dicts, load_aspell_dict, \
    load_character_patterns, load_hist_dict, load_morph_dict
from heuristics import HeuristicsCorrector


@click.command()
@click.argument('f_path', type=click.Path(exists=True))
def main(f_path):
    '''
    Run OCR post-correction using heuristics.

    F_PATH is the location of the data file to be corrected.
    '''
    # Path definitions
    aspell_path_common = '../resources/de-common.wl'
    aspell_path_at = '../resources/de_AT-only.wl'
    aspell_path_ch = '../resources/de_CH-only.wl'
    aspell_path_de = '../resources/de_DE-only.wl'
    morph_path = '../resources/DE_morph_dict.txt'
    hist_path = '../resources/101110_GermanLexicon.xml'
    # variants_path = '../resources/variants.txt'
    ocr_errors_path = '../resources/ocr_errors.txt'

    click.echo('\n#####ERROR POST-CORRECTION USING HEURISTICS#####')
    click.echo('1. Start data loading.')

    # Load data
    data = load_fulltexts(f_path, size=10000)

    # Load dictionaries
    aspell_common = load_aspell_dict(aspell_path_common)
    aspell_at = load_aspell_dict(aspell_path_at)
    aspell_ch = load_aspell_dict(aspell_path_ch)
    aspell_de = load_aspell_dict(aspell_path_de)
    morph = load_morph_dict(morph_path)
    modern_dict = combine_dicts([aspell_common, aspell_at, aspell_ch,
        aspell_de, morph])
    hist_dict = load_hist_dict(hist_path)

    # variants = load_character_patterns(variants_path)
    ocr_errors = load_character_patterns(ocr_errors_path)

    click.echo('2. Start error correction.')

    # Run correction procedure
    c = HeuristicsCorrector(data, modern_dict, hist_dict, ocr_errors)
    c.run_correction()
    c.run_frequency_calculation()

    click.echo('3. Error correction is finished.')
    click.echo('################################################\n')


if __name__ == '__main__':
    main()
