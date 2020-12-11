from collections import defaultdict
from lxml import etree as ET

from helpers import get_file_paths


def parse_page(f_path, conf_threshold):
    '''
    '''
    tree = ET.parse(f_path)
    root = tree.getroot()

    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

    def get_textlines(region):
        textlines = []
        for textline in region:
            try:
                if conf_threshold:
                    if reject_small_confidences(textline):
                        textlines.append(reject_small_confidences(textline))
                else:
                    textlines.append(textline.find('./page:TextEquiv/page:Unicode', namespaces=ns).text)
            except AttributeError:
                pass
        return textlines

    def reject_small_confidences(textline):
        conf = textline.find('./page:TextEquiv', namespaces=ns).attrib['conf']
        if float(conf) >= conf_threshold:
            return textline.find('./page:TextEquiv/page:Unicode', namespaces=ns).text
        else:
            return None

    textlines = []
    reading_order = root.find('.//page:ReadingOrder', ns)
    for group in reading_order.iterfind('./*', ns):
        if ET.QName(group.tag).localname == 'OrderedGroup':
            region_ref_indexeds = group.findall('./page:RegionRefIndexed', ns)
            for region_ref_indexed in sorted(region_ref_indexeds, key=lambda r: int(r.attrib['index'])):
                region_id = region_ref_indexed.attrib['regionRef']
                region = root.find('.//page:TextRegion[@id="%s"]' % region_id, ns)
                textlines += get_textlines(region)
    return textlines


def clean_page(page):
    '''
    '''
    page = [line for line in page if line is not None]
    return page


def extract_ocr_fulltext(ocr_path, conf_threshold=None):
    try:
        f_paths = get_file_paths(ocr_path)
        pages = defaultdict(list)
        f_paths = [f_path for f_path in f_paths if f_path.endswith('.xml')]
        for f_path in f_paths:
            lines = parse_page(f_path, conf_threshold)
            lines = clean_page(lines)
            page_id = f_path.split('.')[-2].split('_')[-1]
            pages[page_id] = lines
        return pages
    except FileNotFoundError as fe:
        print(fe)
