from collections import defaultdict
import re
import xml.sax


class PageHandler(xml.sax.ContentHandler):
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


def clean_page(page):

    # strip whitespace
    page = [e for e in page if len(e) > 0]
    # separate by line break
    page = [e.strip() for e in ' '.join(page).split('<lb>')]
    page = [re.sub(r'\s[\/]', r'/', line) for line in page]
    del page[-1]  # drops last element which seems to be always empty
    page = [line for line in page if line is not None]
    return page
