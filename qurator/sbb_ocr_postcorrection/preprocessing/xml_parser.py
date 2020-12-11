from collections import defaultdict
from lxml import etree as ET
import re
import xml.sax

from qurator.sbb_ocr_postcorrection.helpers import get_file_paths


#####################################
#                                   #
#  TEI File Parsing (with xml sax)  #
#                                   #
#####################################

class TEIHandler(xml.sax.ContentHandler):
    '''A xml sax handler for TEI XML processing.'''

    def __init__(self):
        self.page_number = None
        self.pages = defaultdict(list)
        self.is_id_tag = False
        self.dta_id = None

    def startElement(self, tag, attributes):
        if tag == 'pb':
            self.page_number = re.search('[0-9]+', attributes['facs']).group(0)
            self.pages[self.page_number] = []
        elif tag == 'lb':
            if self.page_number:
                self.pages[self.page_number].append('<lb>')
        elif tag == 'idno':
            if attributes.getLength() > 0:
                if attributes['type'] == 'DTAID':
                    self.is_id_tag = True

    def endElement(self, tag):
        if self.is_id_tag:
            self.is_id_tag = False

    def characters(self, content):
        if self.is_id_tag:
            self.dta_id = content.strip()
        if self.page_number:
            #index = len(self.pages[self.page_number]) - 1
            #print(index)
            #if index >= 0:
            #    if self.pages[self.page_number][index] == 'lb':
            #        self.pages[self.page_number].append(content.strip())
            #    else:
            #        self.pages[self.page_number].append(' ' + content.strip())
            if len(content.strip()) > 0:
                self.pages[self.page_number].append(content.strip())


def clean_tei(page):
    '''
    Clean TEI XML page.

    Strip whitespace, separate by line break and drop last element (empty).

    Keyword arguments:
    page (list) -- the page content

    Outputs:
    page (list) -- the cleaned page
    '''

    # strip whitespace
    page = [e for e in page if len(e) > 0]
    # separate by line break
    page = [e.strip() for e in ' '.join(page).split('<lb>')]
    page = [re.sub(r'\s[\/]', r'/', line) for line in page]
    del page[-1]  # drops last element which seems to be always empty
    page = [line for line in page if line is not None]
    return page


##################################
#                                #
#  Page File Parsing (with lxml) #
#                                #
##################################

def parse_page(path, conf_threshold):
    '''
    Parse PAGE XML.

    Applies textline extraction (using lxml) and optional small confidence
    rejection.

    Keyword arguments:
    path (str) -- the path to the PAGE XML
    conf_threshold (float or None) -- optional confidence value for sequence
                                      quality

    Outputs:
    textlines (list) -- the extracted textlines
    '''

    tree = ET.parse(path)
    root = tree.getroot()

    ns = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}

    def get_textlines(region):
        '''
        Extract textlines.

        Optionally checks for textline confidence if threshold is given.

        Keyword arguments:
        region (list) -- the textlines in region

        Outputs:
        textlines (list) -- the extracted textlines
        '''

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
        '''
        Reject confidences below confidence threshold.

        Keyword arguments:
        textline (xml) -- the textline XML object

        Outputs:
        The textline text (str) -- if confidence is above threshold
        None -- if confidence is below threshold
        '''

        conf = textline.find('./page:TextEquiv', namespaces=ns).attrib['conf']
        if float(conf) >= conf_threshold:
            return textline.find('./page:TextEquiv/page:Unicode', namespaces=ns).text
        else:
            return None

    textlines = []
    reading_order = root.find('.//page:ReadingOrder', ns)
    try:
        for group in reading_order.iterfind('./*', ns):
            if ET.QName(group.tag).localname == 'OrderedGroup':
                region_ref_indexeds = group.findall('./page:RegionRefIndexed', ns)
                for region_ref_indexed in sorted(region_ref_indexeds, key=lambda r: int(r.attrib['index'])):
                    region_id = region_ref_indexed.attrib['regionRef']
                    region = root.find('.//page:TextRegion[@id="%s"]' % region_id, ns)
                    textlines += get_textlines(region)
    except:
        pass

    if len(textlines) < 1:
        textlines.append(None)

    return textlines


def clean_page(page):
    '''
    Clean PAGE XML page.

    Keyword arguments:
    page (list) -- the page content

    Outputs:
    page (list) -- the cleaned page
    '''

    page = [line for line in page if line is not None]
    return page


def extract_page_fulltext(path, id_mapping, conf_threshold=None):
    '''
    Extract fulltext of PAGE XML file.

    Parses PAGE content and cleans it, afterward.

    Keyword arguments:
    path (str) -- path to PAGE XML
    conf_threshold (float) -- optional confidence value for sequence quality
                              (default: None)

    Outputs:
    pages (dict) -- the extracted PAGE content per page id
    '''
    try:
        f_paths = get_file_paths(path)
        pages = defaultdict(list)
        f_paths = [f_path for f_path in f_paths if not f_path.endswith('mets.xml')]
        for f_path in f_paths:
            lines = parse_page(f_path, conf_threshold)
            if lines[0] is not None:
                lines = clean_page(lines)
            try:
                page_id = id_mapping[f_path.split('.')[-2].split('_')[-1]]
                pages[page_id] = lines
            except:
                pass
        return pages, f_paths
    except FileNotFoundError as fe:
        print(fe)


###################################
#                                 #
#  METS File Parsing (with lxml)  #
#                                 #
###################################

def create_ocr_gt_id_mappings(path, ocr_type='OCR-CALAMARI'):
    '''
    Create mapping of OCR/ GT IDs and Page IDs, respectively.

    This function makes use extract_file_idx_from_mets, which extracts the
    needed ID links from the METS file.

    Keyword arguments:
    path (str) -- path to METS file
    ocr_type (str) -- the OCR data to be used (default: 'OCR-CALAMARI')

    Outputs:
    ocr_id_mapping (dict) -- the OCR ID - Page ID mapping
    gt_id_mapping (dict) -- the GT ID - Page ID mapping
    '''
    mets_idx = extract_file_idx_from_mets(path)

    ocr_id_mapping = {}
    gt_id_mapping = {}

    for page_id, idx in mets_idx.items():
        try:
            ocr_id_mapping[idx[ocr_type]] = page_id
            gt_id_mapping[idx['IMG']] = page_id
        except:
            pass

    return ocr_id_mapping, gt_id_mapping


def extract_file_idx_from_mets(path):
    '''
    Extract file ID links from METS file.

    Keyword arguments:
    path (str) -- path to METS file

    Outputs:
    mets_idx (dict) -- contains the ID links
    '''

    tree = ET.parse(path)
    root = tree.getroot()

    ns = {'mets': 'http://www.loc.gov/METS/'}

    mets_idx = {}

    for div in root.find('.//mets:div', ns):
        page_id = [items[1] for items in div.items() if items[0] == 'ID'].pop()
        page_id_dict = {}
        for child in div:
            id_text = child.items()[0][1]
            if re.search('^\d+$', id_text):
                page_id_dict['IMG'] = re.search('^\d+$', id_text).group(0)
            else:
                id_name_match = re.search('OCR-D-(.*)_\d', id_text)
                id_value_match = re.search('\d+', id_text)

                try:
                    id_name = id_name_match.group(1)
                    id_value = id_value_match.group(0)
                    page_id_dict[id_name] = id_value
                except:
                    pass
        mets_idx[page_id] = page_id_dict
    return mets_idx


def convert_to_page_id(parsed_doc, id_mapping):
    '''

    '''
    converted_doc = defaultdict()

    for page, content in parsed_doc.items():
        try:
            converted_doc[id_mapping[page]] = content
        except:
            pass

    return converted_doc
