import importlib
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from models import QueryResponse


class FakeEngine:
    def query(self, question):
        return QueryResponse(
            answer=f"Answered: {question}",
            topic="Test Topic",
            sources=["manual.pdf"],
            confidence=0.91,
            citations=[
                {
                    "source": "manual.pdf",
                    "page": 2,
                    "snippet": "The answer is supported by this manual excerpt.",
                    "score": 0.91,
                }
            ],
            refined_question="standalone test question",
            needs_more_context=False,
        )

    def ingest_url(self, url):
        return 3

    def ingest_pdf(self, path):
        return 4

    def list_sources(self):
        return ["manual.pdf"]

    def delete_source(self, source_path):
        return True, f"Deleted {source_path}"


class ApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["PERSIST_DIRECTORY"] = cls.tmpdir.name
        cls.main = importlib.import_module("main")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_query_endpoint_returns_commercial_quality_fields(self):
        app = self.main.create_app(engine=FakeEngine(), admin_api_key=None)
        client = TestClient(app)

        response = client.post("/query", json={"question": "What is mentioned?"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Answered: What is mentioned?")
        self.assertEqual(payload["confidence"], 0.91)
        self.assertEqual(payload["citations"][0]["source"], "manual.pdf")
        self.assertEqual(payload["citations"][0]["page"], 2)
        self.assertEqual(payload["refined_question"], "standalone test question")
        self.assertFalse(payload["needs_more_context"])

    def test_admin_endpoints_require_key_when_configured(self):
        app = self.main.create_app(engine=FakeEngine(), admin_api_key="secret")
        client = TestClient(app)

        unauthorized = client.post("/admin/ingest-url", json={"url": "https://example.com/docs"})
        authorized = client.post(
            "/admin/ingest-url",
            json={"url": "https://example.com/docs"},
            headers={"X-Admin-Key": "secret"},
        )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertEqual(authorized.json()["chunks_ingested"], 3)


if __name__ == "__main__":
    unittest.main()
