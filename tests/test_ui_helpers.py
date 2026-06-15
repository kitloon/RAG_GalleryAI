import unittest

import ui_helpers


class UiHelpersTest(unittest.TestCase):
    def test_admin_headers_are_empty_until_key_is_configured(self):
        self.assertEqual(ui_helpers.admin_headers(""), {})
        self.assertEqual(ui_helpers.admin_headers("secret"), {"X-Admin-Key": "secret"})

    def test_confidence_label_summarizes_score_for_business_users(self):
        self.assertEqual(ui_helpers.confidence_label(0.82), "High confidence · 82%")
        self.assertEqual(ui_helpers.confidence_label(0.63), "Medium confidence · 63%")
        self.assertEqual(ui_helpers.confidence_label(0.31), "Low confidence · 31%")

    def test_render_citations_html_includes_evidence_and_escapes_snippets(self):
        html = ui_helpers.render_citations_html(
            [
                {
                    "source": "manual.pdf",
                    "page": 5,
                    "snippet": "Supported <script>alert('x')</script> evidence.",
                    "score": 0.91,
                }
            ]
        )

        self.assertIn("manual.pdf · p.5 · 91%", html)
        self.assertIn("Supported &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; evidence.", html)
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
