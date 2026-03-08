#!/usr/bin/env python3
import argparse
import difflib
import json
import math
import re
from collections import defaultdict
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from scipy.optimize import linear_sum_assignment

EVAL_CLASSES = ["title", "paragraph", "figure", "chart", "table", "formula"]
TEXT_CLASSES = {"title", "paragraph"}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def parse_bbox(raw_value: str) -> Optional[List[float]]:
    if not raw_value:
        return None
    numbers = re.findall(r"-?\d+(?:\.\d+)?", raw_value)
    if len(numbers) < 4:
        return None
    x1, y1, x2, y2 = map(float, numbers[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def map_class(tag: str, attrs: Dict[str, str]) -> str:
    cls = (attrs.get("class") or "").lower()
    tag_name = tag.lower()

    if "header" in cls:
        return "header"
    if "footer" in cls:
        return "footer"
    if "page_number" in cls or "pagenumber" in cls:
        return "page_number"

    if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return "title"
    if tag_name == "p":
        return "paragraph"
    if tag_name == "table":
        return "table"
    if tag_name in ("img", "figure"):
        return "figure"
    if tag_name == "chart":
        return "chart"

    if tag_name == "div":
        if "formula" in cls or "equation" in cls or "math" in cls:
            return "formula"
        if "chart" in cls:
            return "chart"
        if "figure" in cls or "image" in cls:
            return "figure"

    if tag_name in ("math", "mrow", "msup", "mi", "mn", "mo"):
        return "formula"

    return "unknown"


class ElementParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[Dict[str, Any]] = []
        self.elements: List[Dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attrs_dict = {k: (v or "") for k, v in attrs}
        node = {
            "tag": tag.lower(),
            "attrs": attrs_dict,
            "bbox": parse_bbox(attrs_dict.get("data-bbox", "")),
            "text": "",
            "inner": "",
            "capture": parse_bbox(attrs_dict.get("data-bbox", "")) is not None,
        }
        self.stack.append(node)

    def handle_endtag(self, tag: str):
        if not self.stack:
            return
        node = self.stack.pop()
        rendered = f"<{tag}>{node['inner']}</{tag}>"
        if self.stack:
            self.stack[-1]["inner"] += rendered
        if node["capture"]:
            self.elements.append(
                {
                    "cls": map_class(node["tag"], node["attrs"]),
                    "bbox": node["bbox"],
                    "text": normalize_space(node["text"]),
                    "html": rendered,
                }
            )

    def handle_data(self, data: str):
        if not self.stack:
            return
        self.stack[-1]["text"] += data
        self.stack[-1]["inner"] += data


def parse_html_elements(html: str) -> List[Dict[str, Any]]:
    parser = ElementParser()
    try:
        parser.feed(html or "")
    except Exception:
        return []
    return parser.elements


def iou(box_a: List[float], box_b: List[float]) -> float:
    if not box_a or not box_b:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def hungarian_match(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]], threshold: float = 0.5):
    if not pred or not gt:
        return []

    import numpy as np

    matrix = np.zeros((len(pred), len(gt)), dtype=float)
    for pred_idx, pred_item in enumerate(pred):
        for gt_idx, gt_item in enumerate(gt):
            matrix[pred_idx, gt_idx] = iou(pred_item["bbox"], gt_item["bbox"])

    row_ind, col_ind = linear_sum_assignment(1.0 - matrix)
    matches = []
    for pred_idx, gt_idx in zip(row_ind, col_ind):
        if matrix[pred_idx, gt_idx] >= threshold:
            matches.append((int(pred_idx), int(gt_idx), float(matrix[pred_idx, gt_idx])))
    return matches


def bleu4(pred_text: str, gt_text: str) -> float:
    pred_chars = list(normalize_space(pred_text))
    gt_chars = list(normalize_space(gt_text))
    pred_len = len(pred_chars)
    gt_len = len(gt_chars)
    if pred_len == 0 or gt_len == 0:
        return 0.0

    import collections

    precisions: List[float] = []
    for ngram_size in range(1, 5):
        if pred_len < ngram_size:
            precisions.append(0.0)
            continue
        pred_counter = collections.Counter(
            tuple(pred_chars[idx : idx + ngram_size]) for idx in range(pred_len - ngram_size + 1)
        )
        gt_counter = collections.Counter(
            tuple(gt_chars[idx : idx + ngram_size]) for idx in range(gt_len - ngram_size + 1)
        )
        overlap = sum(min(value, gt_counter.get(key, 0)) for key, value in pred_counter.items())
        total = sum(pred_counter.values())
        precisions.append(overlap / total if total else 0.0)

    if min(precisions) <= 0:
        return 0.0

    brevity_penalty = 1.0 if pred_len > gt_len else math.exp(1.0 - gt_len / pred_len)
    return brevity_penalty * math.exp(sum(0.25 * math.log(value) for value in precisions))


def teds_proxy(pred_html: str, gt_html: str) -> float:
    pred = normalize_space(re.sub(r">\s+<", "><", pred_html or ""))
    gt = normalize_space(re.sub(r">\s+<", "><", gt_html or ""))
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return difflib.SequenceMatcher(None, pred, gt).ratio()


def load_predictions(debug_jsonl: str) -> Dict[str, str]:
    output: Dict[str, str] = {}
    with open(debug_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "_summary" in row:
                continue
            output[row["image"]] = row.get("answer_html", "")
    return output


def score_dataset(gt_jsonl: str, pred_html_by_image: Dict[str, str]) -> Dict[str, Any]:
    samples = []
    with open(gt_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            gt_row = json.loads(line)
            image = gt_row.get("image")
            if image in pred_html_by_image:
                samples.append((image, gt_row.get("suffix", ""), pred_html_by_image[image]))

    class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    f1_tp = 0
    f1_fp = 0
    f1_fn = 0
    text_bleu_scores: List[float] = []
    formula_bleu_scores: List[float] = []
    table_teds_scores: List[float] = []
    ktds_scores: List[float] = []

    for _, gt_html, pred_html in samples:
        gt_items = [item for item in parse_html_elements(gt_html) if item["cls"] in EVAL_CLASSES]
        pred_items = [item for item in parse_html_elements(pred_html) if item["cls"] in EVAL_CLASSES]

        for cls_name in EVAL_CLASSES:
            gt_cls = [item for item in gt_items if item["cls"] == cls_name]
            pred_cls = [item for item in pred_items if item["cls"] == cls_name]
            matched = hungarian_match(pred_cls, gt_cls, threshold=0.5)
            class_counts[cls_name]["tp"] += len(matched)
            class_counts[cls_name]["fp"] += len(pred_cls) - len(matched)
            class_counts[cls_name]["fn"] += len(gt_cls) - len(matched)

        global_matches = hungarian_match(pred_items, gt_items, threshold=0.5)
        matched_pred = {pred_idx for pred_idx, _, _ in global_matches}
        matched_gt = {gt_idx for _, gt_idx, _ in global_matches}

        for pred_idx, gt_idx, _ in global_matches:
            if pred_items[pred_idx]["cls"] == gt_items[gt_idx]["cls"]:
                f1_tp += 1
            else:
                f1_fp += 1
                f1_fn += 1
        f1_fp += len(pred_items) - len(matched_pred)
        f1_fn += len(gt_items) - len(matched_gt)

        class_match_map: Dict[Tuple[str, int], int] = {}
        for cls_name in ("title", "paragraph", "formula", "table"):
            gt_cls_indexed = [(idx, item) for idx, item in enumerate(gt_items) if item["cls"] == cls_name]
            pred_cls_indexed = [(idx, item) for idx, item in enumerate(pred_items) if item["cls"] == cls_name]
            matched = hungarian_match(
                [item for _, item in pred_cls_indexed],
                [item for _, item in gt_cls_indexed],
                threshold=0.5,
            )
            for pred_local_idx, gt_local_idx, _ in matched:
                class_match_map[(cls_name, gt_cls_indexed[gt_local_idx][0])] = pred_cls_indexed[pred_local_idx][0]

        for gt_idx, gt_item in enumerate(gt_items):
            cls_name = gt_item["cls"]
            if cls_name in TEXT_CLASSES:
                pred_idx = class_match_map.get((cls_name, gt_idx))
                score = bleu4(pred_items[pred_idx]["text"], gt_item["text"]) if pred_idx is not None else 0.0
                text_bleu_scores.append(score)
            elif cls_name == "formula":
                pred_idx = class_match_map.get((cls_name, gt_idx))
                score = bleu4(pred_items[pred_idx]["text"], gt_item["text"]) if pred_idx is not None else 0.0
                formula_bleu_scores.append(score)
            elif cls_name == "table":
                pred_idx = class_match_map.get((cls_name, gt_idx))
                score = teds_proxy(pred_items[pred_idx]["html"], gt_item["html"]) if pred_idx is not None else 0.0
                table_teds_scores.append(score)

        same_class_pairs = [
            (pred_idx, gt_idx)
            for pred_idx, gt_idx, _ in hungarian_match(pred_items, gt_items, threshold=0.5)
            if pred_items[pred_idx]["cls"] == gt_items[gt_idx]["cls"]
        ]
        if len(same_class_pairs) < 2:
            ktds_scores.append(1.0)
        else:
            same_class_pairs.sort(key=lambda pair: pair[1])
            pred_order = [pair[0] for pair in same_class_pairs]
            discordant = 0
            pair_count = len(pred_order)
            for left in range(pair_count):
                for right in range(left + 1, pair_count):
                    if pred_order[left] > pred_order[right]:
                        discordant += 1
            ktds = max(0.0, 1.0 - (2.0 * discordant) / (pair_count * (pair_count - 1)))
            ktds_scores.append(ktds)

    ap_values = []
    per_class_ap: Dict[str, Optional[float]] = {}
    for cls_name in EVAL_CLASSES:
        counts = class_counts[cls_name]
        denom = counts["tp"] + counts["fp"] + counts["fn"]
        ap = (counts["tp"] / denom) if denom > 0 else None
        per_class_ap[cls_name] = ap
        if ap is not None:
            ap_values.append(ap)

    mean_ap = sum(ap_values) / len(ap_values) if ap_values else 0.0
    precision = f1_tp / (f1_tp + f1_fp) if (f1_tp + f1_fp) > 0 else 0.0
    recall = f1_tp / (f1_tp + f1_fn) if (f1_tp + f1_fn) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    text_bleu = sum(text_bleu_scores) / len(text_bleu_scores) if text_bleu_scores else 0.0
    formula_bleu = sum(formula_bleu_scores) / len(formula_bleu_scores) if formula_bleu_scores else 0.0
    table_teds = sum(table_teds_scores) / len(table_teds_scores) if table_teds_scores else 0.0
    ktds = sum(ktds_scores) / len(ktds_scores) if ktds_scores else 0.0

    weighted_score = (
        0.3 * mean_ap
        + 0.1 * f1_score
        + 0.2 * text_bleu
        + 0.15 * formula_bleu
        + 0.15 * table_teds
        + 0.1 * ktds
    )

    return {
        "samples": len(samples),
        "mAP": mean_ap,
        "F1": f1_score,
        "B_text": text_bleu,
        "B_formula": formula_bleu,
        "T_table": table_teds,
        "K_order": ktds,
        "weighted_score": weighted_score,
        "ap_by_class": per_class_ap,
        "note": "proxy metrics without official contest evaluator",
    }


def main():
    parser = argparse.ArgumentParser(description="Proxy scorer for OCR challenge weighted score")
    parser.add_argument("--gt", default="datasets/label/eval.jsonl", help="Ground-truth jsonl")
    parser.add_argument("--debug", required=True, help="Prediction debug jsonl from eval.py")
    args = parser.parse_args()

    predictions = load_predictions(args.debug)
    report = score_dataset(args.gt, predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
