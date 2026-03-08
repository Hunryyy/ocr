"""
Sanity check for table HTML rendering.

Run from the project root with:
    python -m unittest tests.test_table_html
"""

import sys
import os
import unittest

# Ensure the project root is on the path when running this file directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import only the functions we need to test (import-safe subset)
from eval import _render_table_cell, _render_table_content, _render_block


class TestRenderTableCell(unittest.TestCase):
    def test_simple_cell_no_span(self):
        cell = {"bbox": [0, 0, 50, 20], "text": "Hello", "rowspan": 1, "colspan": 1}
        html = _render_table_cell(cell)
        self.assertEqual(html, "<td>Hello</td>")
        self.assertNotIn("data-bbox", html)
        self.assertNotIn("rowspan", html)
        self.assertNotIn("colspan", html)

    def test_cell_with_rowspan(self):
        cell = {"bbox": [0, 0, 50, 40], "text": "A", "rowspan": 2, "colspan": 1}
        html = _render_table_cell(cell)
        self.assertIn('rowspan="2"', html)
        self.assertNotIn("colspan", html)
        self.assertNotIn("data-bbox", html)

    def test_cell_with_colspan(self):
        cell = {"bbox": [0, 0, 100, 20], "text": "B", "rowspan": 1, "colspan": 3}
        html = _render_table_cell(cell)
        self.assertIn('colspan="3"', html)
        self.assertNotIn("rowspan", html)
        self.assertNotIn("data-bbox", html)

    def test_cell_with_both_spans(self):
        cell = {"bbox": [0, 0, 100, 40], "text": "C", "rowspan": 2, "colspan": 2}
        html = _render_table_cell(cell)
        self.assertIn('rowspan="2"', html)
        self.assertIn('colspan="2"', html)
        self.assertNotIn("data-bbox", html)

    def test_no_rowspan_1_or_colspan_1(self):
        """rowspan=1 and colspan=1 must NOT appear in output."""
        cell = {"bbox": [0, 0, 50, 20], "text": "X", "rowspan": 1, "colspan": 1}
        html = _render_table_cell(cell)
        self.assertNotIn('rowspan="1"', html)
        self.assertNotIn('colspan="1"', html)


class TestRenderTableContent(unittest.TestCase):
    def test_empty_table(self):
        html = _render_table_content({})
        self.assertIn("<table>", html)
        self.assertIn("</table>", html)
        self.assertNotIn("<thead>", html)
        self.assertNotIn("<tbody>", html)

    def test_simple_table(self):
        table_obj = {
            "rows": [
                [{"text": "H1", "rowspan": 1, "colspan": 1}, {"text": "H2", "rowspan": 1, "colspan": 1}],
                [{"text": "A", "rowspan": 1, "colspan": 1}, {"text": "B", "rowspan": 1, "colspan": 1}],
            ]
        }
        html = _render_table_content(table_obj)
        self.assertIn("<table>", html)
        self.assertIn("</table>", html)
        self.assertNotIn("<thead>", html)
        self.assertNotIn("<tbody>", html)
        self.assertIn("<tr>", html)
        self.assertIn("<td>H1</td>", html)
        self.assertIn("<td>H2</td>", html)
        self.assertIn("<td>A</td>", html)
        self.assertIn("<td>B</td>", html)
        self.assertNotIn("data-bbox", html)

    def test_table_no_data_bbox_in_cells(self):
        table_obj = {
            "rows": [
                [{"bbox": [0, 0, 50, 20], "text": "X", "rowspan": 1, "colspan": 1}],
            ]
        }
        html = _render_table_content(table_obj)
        self.assertNotIn("data-bbox", html)


class TestRenderBlock(unittest.TestCase):
    def test_table_block_outer_div(self):
        block = {"type": "table", "bbox": [10, 20, 200, 100], "id": 1}
        table_obj = {
            "id": 1,
            "rows": [
                [{"text": "Cell", "rowspan": 1, "colspan": 1}],
            ]
        }
        html = _render_block(block, None, table_obj)
        self.assertTrue(html.startswith('<div class="table" data-bbox="'))
        self.assertIn("<table>", html)
        self.assertIn("</table>", html)
        self.assertIn("</div>", html)
        self.assertNotIn("<thead>", html)
        self.assertNotIn("<tbody>", html)
        # td must not have data-bbox
        self.assertNotIn('<td data-bbox', html)

    def test_page_number_block(self):
        block = {"type": "page_number", "bbox": [100, 900, 180, 930], "text": "-2-"}
        html = _render_block(block, None, None)
        self.assertIn('<div class="page_number" data-bbox="100 900 180 930">-2-</div>', html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
