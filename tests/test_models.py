import unittest

from models import QueryResponse


class QueryResponseContractTest(unittest.TestCase):
    def test_query_response_keeps_legacy_fields_and_adds_quality_metadata(self):
        response = QueryResponse(
            answer="The 911 has rear-mounted engine heritage.",
            topic="Product History",
            sources=["porsche.pdf"],
        )

        self.assertEqual(response.answer, "The 911 has rear-mounted engine heritage.")
        self.assertEqual(response.topic, "Product History")
        self.assertEqual(response.sources, ["porsche.pdf"])
        self.assertEqual(response.confidence, 0.0)
        self.assertEqual(response.citations, [])
        self.assertEqual(response.refined_question, "")
        self.assertFalse(response.needs_more_context)

    def test_query_response_accepts_citations_with_page_snippet_and_score(self):
        response = QueryResponse(
            answer="The Taycan is Porsche's electric sports sedan.",
            topic="Product Info",
            sources=["range.pdf"],
            confidence=0.82,
            citations=[
                {
                    "source": "range.pdf",
                    "page": 4,
                    "snippet": "Taycan models are fully electric sports cars.",
                    "score": 0.82,
                }
            ],
            refined_question="Taycan electric model information",
            needs_more_context=False,
        )

        citation = response.citations[0]
        self.assertEqual(citation.source, "range.pdf")
        self.assertEqual(citation.page, 4)
        self.assertEqual(citation.snippet, "Taycan models are fully electric sports cars.")
        self.assertEqual(citation.score, 0.82)


if __name__ == "__main__":
    unittest.main()
