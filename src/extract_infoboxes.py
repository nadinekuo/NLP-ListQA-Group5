import mwparserfromhell
import xml.etree.ElementTree as ET
import bz2
import json

def extract_infoboxes(wiki_dump_path, output_file, max_infoboxes):
    infoboxes = []
    page_count = 0
    infobox_count = 0

    with bz2.open(wiki_dump_path, 'r') as f:
        for event, elem in ET.iterparse(f, events=('end',)):
            if elem.tag == "{http://www.mediawiki.org/xml/export-0.10/}page":
                page_count += 1
                if page_count % 100 == 0:
                    print(f"Processed {page_count} pages, found {infobox_count} infoboxes.")

                title = elem.find("{http://www.mediawiki.org/xml/export-0.10/}title").text
                text = elem.find(".//{http://www.mediawiki.org/xml/export-0.10/}text").text

                if text:
                    wikicode = mwparserfromhell.parse(text)
                    templates = wikicode.filter_templates()

                    for template in templates:
                        if template.name.matches("Infobox"):
                            infoboxes.append({"title": title, "infobox": str(template)})
                            infobox_count += 1
                            if infobox_count >= max_infoboxes:
                                print(f"Reached {max_infoboxes} infoboxes. Stopping extraction.")
                                with open(output_file, 'w') as outfile:
                                    json.dump(infoboxes, outfile)
                                return

                elem.clear()  # Clear the element to save memory

    with open(output_file, 'w') as outfile:
        json.dump(infoboxes, outfile)
    print(f"Extraction complete: Processed {page_count} pages, found {infobox_count} infoboxes.")

# Update with your Wikipedia dump file path and desired output file path
wiki_dump_path = r"C:\Users\vpoch\Desktop\DELFT\Courses\Natural Language Processing\Projects\enwiki-latest-pages-articles.xml.bz2"
output_file = "extracted_infoboxes.json"
max_infoboxes = 100  # Number of infoboxes to extract
extract_infoboxes(wiki_dump_path, output_file, max_infoboxes)