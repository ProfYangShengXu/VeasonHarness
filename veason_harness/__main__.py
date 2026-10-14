"""零安装入口：``python -m veason_harness <cmd>``。

clone 下来不装任何东西就能跑这条路（`veason` 命令需要 pip install -e .）。
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
