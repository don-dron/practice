"""python -m htr → справка по подкомандам train / infer."""

import sys


def main() -> None:
    print(
        "используйте entrypoints htr-train и htr-infer или модуль:",
        "`python -m htr.cli train_main ...`, `python -m htr.cli infer_main ...`",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
