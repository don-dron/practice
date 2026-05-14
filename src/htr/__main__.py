"""Запуск из корня без установки: PYTHONPATH=src python -m htr infer … | train …"""

from __future__ import annotations

import sys


def _usage(exit_code: int = 0) -> None:
    prog = "python -m htr"
    print(
        f"Использование:\n"
        f"  {prog} infer  [...]   — см. htr.cli.infer_main (или установленная команда htr-infer)\n"
        f"  {prog} train [...]   — см. htr.cli.train_main (или htr-train)\n"
        f"\n"
        f"Из корня репозитория без pip install:\n"
        f"  PYTHONPATH=src {prog} infer --checkpoint training/registry_snapshots/T0_latest.pt --device cpu img.jpg\n",
        file=sys.stderr,
    )
    raise SystemExit(exit_code)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _usage(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "infer":
        from htr.cli import infer_main

        sys.argv = ["htr-infer"] + rest
        infer_main()
    elif cmd == "train":
        from htr.cli import train_main

        sys.argv = ["htr-train"] + rest
        train_main()
    else:
        print(f"неизвестная подкоманда: {cmd!r}\n", file=sys.stderr)
        _usage(2)


if __name__ == "__main__":
    main()
