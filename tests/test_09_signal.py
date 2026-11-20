"""业务要求 8：可处理退出信号下完成当前图像提交后以退出码 0 退出。

使用线程内 Event 模拟信号（跨平台、不依赖真实 OS 信号投递时机），
验证停止后账本仍完整提交且可恢复。
"""

from __future__ import annotations

import os
import threading

from vrep.config import Config
from vrep.engine import Engine
from vrep.injection import InjectionManager

from ._base import SampleTestBase


class SignalExitTests(SampleTestBase):
    def test_stop_event_finalizes_current_batch(self) -> None:
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        stop_event = threading.Event()
        engine = Engine(
            os.path.join(self.temp_root, "signal-engine"),
            InjectionManager(None),
            stop_event=stop_event,
        )
        engine.seed_library(self.samples)

        # 在第一批计算开始后立即请求停止：引擎必须完成当前批次提交
        stop_event.set()
        engine.run_batch(config, batch_no=1)

        # 账本已完整落盘，且恢复点与当前账本一致
        on_disk = engine.store.load_ledger()
        self.assertEqual(on_disk["global_version"], 1)
        self.assertGreaterEqual(len(on_disk["images"]), 1)

        restarted = Engine(
            os.path.join(self.temp_root, "signal-engine"), InjectionManager(None)
        )
        manifest = restarted.recover()
        # 若提交间隔未触发快照，则没有恢复点；已落盘账本依然完整
        if manifest is not None:
            self.assertEqual(
                manifest["global_version"], on_disk["global_version"]
            )

    def test_stop_before_next_image_keeps_submitted_state(self) -> None:
        config = Config(
            **{
                **Config.load(self.config_path(), image_size=self.IMAGE_SIZE).to_dict(),
                "snapshot_commit_interval": 1,
            }
        )
        stop_event = threading.Event()
        engine = Engine(
            os.path.join(self.temp_root, "signal-snapshot"),
            InjectionManager(None),
            stop_event=stop_event,
        )
        engine.seed_library(self.samples)
        stop_event.set()
        engine.run_batch(config, batch_no=1)

        manifest = engine.store.recover()[1]
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["quant_level"], config.quant_level)
        self.assertIn("stage_versions", manifest)
