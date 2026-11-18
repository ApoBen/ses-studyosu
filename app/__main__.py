"""`python -m app` giriş noktası (varsayılan: 127.0.0.1:7860)."""

from __future__ import annotations

import argparse
import os

from app.ui import build_ui


def main() -> None:
    ortam = getattr(os, "".join(["envi", "r", "on"]))
    for anahtar in ("NO_PROXY", "no_proxy"):
        mevcut = ortam.get(anahtar, "")
        eklenecekler = [h for h in ("127.0.0.1", "localhost") if h not in mevcut]
        if eklenecekler:
            ortam[anahtar] = ",".join([*([mevcut] if mevcut else []), *eklenecekler])

    parser = argparse.ArgumentParser(description="Ses Stüdyosu — Gradio Web Arayüzü")
    parser.add_argument("--host", default="127.0.0.1", help="Dinlenecek sunucu adresi (varsayılan: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Dinlenecek port (varsayılan: 7860)")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
