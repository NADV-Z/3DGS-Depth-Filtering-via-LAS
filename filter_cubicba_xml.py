import argparse
import os
import xml.etree.ElementTree as ET
from copy import deepcopy


def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    keep_names = {
        f.lower()
        for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    print(f"Images whitelist: {len(keep_names)}")

    tree = ET.parse(args.xml)
    root = tree.getroot()

    total_photos = 0
    kept_photos = 0

    for photogroup in root.findall(".//Photogroup"):
        photos = photogroup.findall("Photo")
        total_photos += len(photos)

        for photo in list(photos):
            image_path_node = photo.find("ImagePath")
            if image_path_node is None or not image_path_node.text:
                photogroup.remove(photo)
                continue

            image_name = os.path.basename(image_path_node.text).lower()

            if image_name in keep_names:
                kept_photos += 1
            else:
                photogroup.remove(photo)

    missing_in_xml = keep_names.copy()
    for photo in root.findall(".//Photo"):
        image_path_node = photo.find("ImagePath")
        if image_path_node is not None and image_path_node.text:
            missing_in_xml.discard(os.path.basename(image_path_node.text).lower())

    indent(root)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)

    print(f"Original Photo count: {total_photos}")
    print(f"Kept Photo count: {kept_photos}")
    print(f"Images not found in XML: {len(missing_in_xml)}")
    print(f"Output: {args.output}")

    if missing_in_xml:
        print("First missing examples:")
        for name in sorted(missing_in_xml)[:20]:
            print("  ", name)


if __name__ == "__main__":
    main()
