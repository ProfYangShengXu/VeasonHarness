"""pytest 与 ``tests/test_vrs.py`` 的接缝。

``tests/test_vrs.py`` 是**脚本式**测试：每个用例收一个位置参数 ``tmp: Path``，
由它自己的 ``main()`` 传进来；顶层不用 pytest，所以没有 fixture 的概念。
一跑 ``pytest tests/``，pytest 就把 ``tmp`` 当成一个 fixture 去找 ——
找不到，15 个用例全 ERROR（实测：``7 passed, 15 errors``）。

★ 为什么在这里补一个 ``tmp`` fixture，而不是把 ``test_vrs.py`` 改成 pytest 风格：
  那个文件的价值在于「每条断言都曾被观察到失败」，改写法会动到所有断言的
  上下文；而这个接缝只有一个语义 —— **给每个用例一个独立的临时目录**。
  补在最外层，两边的调用方式都不用改。

★ 用 ``tmp_path_factory`` 而不是 ``tmpdir``/``tmp_path``：后者也是 ``Path``，
  但作用域与 ``test_vrs.py`` 的 per-case「用完即删」语义更接近的是
  function-scope 的一次性目录。这里显式建/删，行为与脚本的
  ``tempfile.mkdtemp`` + ``finally: shutil.rmtree`` 一致。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp():
    """``tests/test_vrs.py`` 要的那个 ``tmp: Path``。用例结束即删。"""
    path = Path(tempfile.mkdtemp(prefix="vrs-test-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
