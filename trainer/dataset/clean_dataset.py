import json
import os
import re
import argparse
import html5lib
from lxml import etree

def parse_html(html_str):
    try:
        doc = html5lib.parse(html_str, namespaceHTMLElements=False)
        return doc.find(".//body")
    except Exception as e:
        return None

def bbox_from_attr(el):
    bbox_str = el.get("data-bbox")
    if not bbox_str: return None
    parts = bbox_str.strip().split()
    if len(parts) != 4: return None
    try:
        return [float(x) for x in parts]
    except:
        return None

def clean_html(body):
    """
    Remove noisy elements from the body.
    Returns: bool (whether changes were made)
    """
    modified = False
    elements_to_remove = []
    
    for el in body.iter():
        if el.tag is etree.Comment:
            continue
            
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        bbox = bbox_from_attr(el)
        if not bbox: continue
        
        x1, y1, x2, y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        text = "".join(el.itertext()).strip()
        
        # 1. Clean noisy formulas
        if tag == "div" and "formula" in el.get("class", ""):
            # Anomalous aspect ratio (e.g., vertical sliver or huge block without much text)
            if h > w * 3: # Too tall and thin for a typical formula
                elements_to_remove.append(el)
            elif w < 10 or h < 10: # Too small
                elements_to_remove.append(el)
            elif not text and not el.get("data-latex"):
                # Empty formula block
                elements_to_remove.append(el)
                
        # 2. Clean noisy titles
        if tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            # Titles shouldn't be massive boxes (e.g. > 1/3 of typical page width/height at once if very sparse)
            if h > 500 and len(text) < 10:
                elements_to_remove.append(el)
            elif w < 10 or h < 10: 
                elements_to_remove.append(el)
    
    for el in elements_to_remove:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)
            modified = True
            
    return modified

def serialize_html(body):
    # Returns inner HTML of body
    inner = ""
    for child in body:
        inner += etree.tostring(child, encoding='unicode', method='html')
    return inner.strip()

def process_file(in_path, out_path):
    print(f"Cleaning {in_path} -> {out_path}")
    with open(in_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        total = 0
        cleaned = 0
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            total += 1
            
            html_content = data.get("answer", "")
            if html_content:
                # Wrap in body for parsing
                wrapped = f"<body>{html_content}</body>"
                body = parse_html(wrapped)
                if body is not None:
                    if clean_html(body):
                        data["answer"] = serialize_html(body)
                        cleaned += 1
            
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    print(f"Processed {total} items, modified/cleaned {cleaned} items.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    process_file(args.input, args.output)
