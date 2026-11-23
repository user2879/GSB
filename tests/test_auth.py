"""Signature auth tests."""

import unittest

from workbench.authn import (REASON_EXPIRED, REASON_MISMATCH, REASON_OK,
                             REASON_UNKNOWN_VERSION, expected_signature,
                             verify_signature)
from workbench.spec import KeyVersion

KEYS = (
    KeyVersion(version=1, secret="s1", retire_ms=1000),
    KeyVersion(version=2, secret="s2", retire_ms=5000),
    KeyVersion(version=3, secret="s3", retire_ms=-1),
)
WINDOW = 1000


def sig(version, seq=7, client="c1", svc="svc-a"):
    secret = {1: "s1", 2: "s2", 3: "s3"}[version]
    return expected_signature(secret, client, svc, seq, version)


class AuthTest(unittest.TestCase):
    def verify(self, version, signature, arrival):
        return verify_signature(KEYS, WINDOW, version, signature,
                                "c1", "svc-a", 7, arrival)

    def test_current_version_ok(self):
        self.assertEqual(self.verify(3, sig(3), 99999),
                         (True, REASON_OK))

    def test_signature_mismatch(self):
        self.assertEqual(self.verify(3, "deadbeef", 0),
                         (False, REASON_MISMATCH))

    def test_old_version_inside_window(self):
        self.assertEqual(self.verify(1, sig(1), 2000), (True, REASON_OK))

    def test_old_version_expired(self):
        self.assertEqual(self.verify(1, sig(1), 2001),
                         (False, REASON_EXPIRED))

    def test_expired_checked_before_signature(self):
        self.assertEqual(self.verify(1, "deadbeef", 99999),
                         (False, REASON_EXPIRED))

    def test_unknown_version(self):
        self.assertEqual(self.verify(9, "deadbeef", 0),
                         (False, REASON_UNKNOWN_VERSION))

    def test_mid_version_window(self):
        self.assertEqual(self.verify(2, sig(2), 6000), (True, REASON_OK))
        self.assertEqual(self.verify(2, sig(2), 6001),
                         (False, REASON_EXPIRED))

    def test_deterministic_across_runs(self):
        results = {self.verify(3, sig(3), 100) for _ in range(50)}
        self.assertEqual(results, {(True, REASON_OK)})


if __name__ == "__main__":
    unittest.main()
