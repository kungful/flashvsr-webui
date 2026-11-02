#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把镜像自带那个 Gradio UI 从 7860 挪到 127.0.0.1:7861 并挂到 /g（幂等）。

镜像里的 /root/app/app.py 是纯 Gradio 版：
    demo.queue(...).launch(server_name="0.0.0.0", server_port=7860, ...)
改完变成：
    demo.queue(...).launch(server_name="0.0.0.0", server_port=7861, root_path="/g", ...)
这样新 UI 可以占住 7860，老的 Gradio 通过 /g 反代访问。
"""
import re
import shutil
import sys

P = "/root/app/app.py"
src = open(P, encoding="utf-8").read()

if 'root_path="/g"' in src:
    print("already patched (root_path=/g), nothing to do")
    sys.exit(0)

if "server_port=7860" not in src:
    print("!! 没找到 server_port=7860，跳过（可能已经是别的版本）")
    sys.exit(0)

shutil.copy2(P, P + ".bak-port")
out = src.replace("server_port=7860", 'server_port=7861, root_path="/g"')
open(P, "w", encoding="utf-8").write(out)
print("patched: server_port=7861, root_path=/g ; backup at", P + ".bak-port")

# 顺带确认语法
import ast
ast.parse(open(P, encoding="utf-8").read())
print("syntax ok")
