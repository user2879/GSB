from __future__ import annotations

import unittest

from edgecache.content import derive_shard, manifest_digest
from edgecache.model import ContentSpec, Manifest, NodeSpec, Config


class ContentDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node1 = NodeSpec("n1", "127.0.0.1", 19001, 2, 2, 11, 1)
        self.node2 = NodeSpec("n2", "127.0.0.1", 19002, 2, 2, 22, 1)
        self.content = ContentSpec("clip", 4096, 4, 99)

    def test_repeated_derivation_is_byte_identical(self) -> None:
        first = derive_shard(self.node1, self.content, 2, 64)
        second = derive_shard(self.node1, self.content, 2, 64)
        self.assertEqual(first.data, second.data)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(len(first.data), 1024)

    def test_digest_binds_node_content_and_shard(self) -> None:
        base = derive_shard(self.node1, self.content, 0, 64)
        other_node = derive_shard(self.node2, self.content, 0, 64)
        other_shard = derive_shard(self.node1, self.content, 1, 64)
        other_content = derive_shard(self.node1, ContentSpec("other", 4096, 4, 99), 0, 64)
        self.assertEqual(len({base.digest, other_node.digest, other_shard.digest, other_content.digest}), 4)

    def test_declaration_order_does_not_change_shards(self) -> None:
        config = Config(64, 1000, 2, 8192, 1000, 1000, 2, 1000)
        left = Manifest(1, config, (self.node1, self.node2), (self.content,), ())
        right = Manifest(1, config, (self.node2, self.node1), (self.content,), ())
        for shard in range(self.content.shard_count):
            a = derive_shard(left.node_map["n1"], left.content_map["clip"], shard, 64)
            b = derive_shard(right.node_map["n1"], right.content_map["clip"], shard, 64)
            self.assertEqual(a.data, b.data)
            self.assertEqual(a.digest, b.digest)
        self.assertEqual(manifest_digest(left), manifest_digest(right))

    def test_effective_scale_changes_fixed_point_result(self) -> None:
        first = derive_shard(self.node1, self.content, 0, 1)
        second = derive_shard(self.node1, self.content, 0, 128)
        self.assertNotEqual(first.data, second.data)
        self.assertNotEqual(first.checksum, second.checksum)


if __name__ == "__main__":
    unittest.main()
