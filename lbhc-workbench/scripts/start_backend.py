"""Start one simulated backend instance by hand (debug helper).

Usage: python scripts/start_backend.py --port 18601 --behavior ok
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lbhc.backend import main  # noqa: E402

if __name__ == "__main__":
    main()
