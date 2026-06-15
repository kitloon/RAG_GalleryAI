import unittest
from types import SimpleNamespace

import rag_engine


class RagQualityHelpersTest(unittest.TestCase):
    def test_build_citations_preserves_source_page_snippet_and_score(self):
        doc = SimpleNamespace(
            page_content=(
                "The Porsche 911 has a distinctive rear-engine layout. "
                "It is one of the brand's most recognizable sports cars."
            ),
            metadata={"source": "Porsche-Range.pdf", "page": 2},
        )

        citations = rag_engine.build_citations([(doc, 0.87321)], snippet_chars=70)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].source, "Porsche-Range.pdf")
        self.assertEqual(citations[0].page, 3)
        self.assertEqual(citations[0].score, 0.873)
        self.assertIn("rear-engine layout", citations[0].snippet)
        self.assertLessEqual(len(citations[0].snippet), 70)

    def test_quality_gate_rejects_empty_and_low_confidence_retrieval(self):
        self.assertTrue(rag_engine.needs_more_context([], confidence=0.0, threshold=0.45))
        self.assertTrue(rag_engine.needs_more_context(["manual.pdf"], confidence=0.2, threshold=0.45))
        self.assertFalse(rag_engine.needs_more_context(["manual.pdf"], confidence=0.81, threshold=0.45))

    def test_response_language_follows_user_question(self):
        self.assertEqual(rag_engine.detect_response_language("这台车有什么安全功能？"), "Chinese")
        self.assertEqual(rag_engine.detect_response_language("What safety features are mentioned?"), "English")

    def test_persist_directory_is_read_from_current_environment(self):
        old_value = rag_engine.os.environ.get("PERSIST_DIRECTORY")
        try:
            rag_engine.os.environ["PERSIST_DIRECTORY"] = "/tmp/gallery-ai-test-db"
            self.assertEqual(rag_engine.get_persist_directory(), "/tmp/gallery-ai-test-db")
        finally:
            if old_value is None:
                rag_engine.os.environ.pop("PERSIST_DIRECTORY", None)
            else:
                rag_engine.os.environ["PERSIST_DIRECTORY"] = old_value


if __name__ == "__main__":
    unittest.main()
