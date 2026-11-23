import unittest

from arbworkbench import gui


class GuiGuardTests(unittest.TestCase):
    def test_launch_raises_guidance_when_tk_missing(self):
        if gui.TK_AVAILABLE:
            self.skipTest("Tk is available in this environment")
        with self.assertRaises(gui.HeadlessUnavailable) as ctx:
            gui.launch_gui("x.json", "y.json", "z.sqlite")
        self.assertIn("Tkinter", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
