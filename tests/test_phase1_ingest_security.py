import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from flask import Flask
from app import app
from config import INGEST_ALLOWED_ROOT
import ingest

class TestPhase1IngestSecurity(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_ingest_requires_auth_when_key_set(self):
        """Assert that /api/ingest is rejected without X-Api-Key if API_SECRET_KEY is configured."""
        with patch("app.API_SECRET_KEY", "super-secret"):
            response = self.app.post("/api/ingest", json={"kind": "txt", "target": "somefile.txt"})
            self.assertEqual(response.status_code, 401)
            self.assertIn("Unauthorized", response.get_json()["message"])

    def test_upload_requires_auth_when_key_set(self):
        """Assert that /api/upload is rejected without X-Api-Key if API_SECRET_KEY is configured."""
        with patch("app.API_SECRET_KEY", "super-secret"):
            response = self.app.post("/api/upload", data={"file": (None, "")})
            self.assertEqual(response.status_code, 401)
            self.assertIn("Unauthorized", response.get_json()["message"])

    def test_ingest_path_outside_allowed_root_rejected(self):
        """Assert that a path outside INGEST_ALLOWED_ROOT raises a ValueError during document ingestion."""
        outside_path = Path("/etc/passwd").resolve()
        with self.assertRaises(ValueError) as context:
            ingest.ingest_txt(str(outside_path))
        self.assertIn("not within the allowed ingest root directory", str(context.exception))

    def test_ingest_url_private_ip_rejected(self):
        """Assert that a URL resolving to local/private IP is rejected to prevent SSRF."""
        for target_url in ["http://127.0.0.1/metadata", "http://localhost/test", "http://192.168.1.1/config"]:
            with self.assertRaises(ValueError) as context:
                ingest.ingest_url(target_url)
            self.assertIn("resolves to a restricted IP address", str(context.exception))

    def test_ingest_url_private_ip_override(self):
        """Assert that private IP restriction can be bypassed with ALLOW_INTERNAL_INGEST_URLS=1."""
        with patch.dict(os.environ, {"ALLOW_INTERNAL_INGEST_URLS": "1"}):
            with patch("requests.Session.get") as mock_get:
                mock_resp = MagicMock()
                mock_resp.text = "<html><body><p>Some text</p></body></html>"
                mock_get.return_value = mock_resp
                
                # Mock _append_documents to bypass database additions
                with patch("ingest._append_documents", return_value=1):
                    count = ingest.ingest_url("http://127.0.0.1/health", force=True)
                    self.assertEqual(count, 1)

if __name__ == "__main__":
    unittest.main()
