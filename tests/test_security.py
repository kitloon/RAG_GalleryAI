import unittest

import security


class SecurityHelpersTest(unittest.TestCase):
    def test_safe_upload_filename_strips_paths_and_blocks_empty_names(self):
        self.assertEqual(security.safe_upload_filename("../../secret.pdf"), "secret.pdf")

        with self.assertRaises(ValueError):
            security.safe_upload_filename("../../")

    def test_validate_public_url_blocks_local_network_targets(self):
        self.assertEqual(security.validate_public_url("https://example.com/docs"), "https://example.com/docs")

        with self.assertRaises(ValueError):
            security.validate_public_url("http://127.0.0.1:8000/admin")

        with self.assertRaises(ValueError):
            security.validate_public_url("file:///etc/passwd")

    def test_admin_authorization_is_optional_but_strict_when_configured(self):
        self.assertTrue(security.is_admin_authorized(None, None))
        self.assertFalse(security.is_admin_authorized(None, "secret"))
        self.assertFalse(security.is_admin_authorized("wrong", "secret"))
        self.assertTrue(security.is_admin_authorized("secret", "secret"))


if __name__ == "__main__":
    unittest.main()
