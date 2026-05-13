"""CUDA probe for train_registry_all.cmd (avoids brittle cmd.exe -c quoting)."""
from __future__ import annotations

import sys


def main() -> int:
    short = len(sys.argv) > 1 and sys.argv[1].lower() == "short"

    import torch  # noqa: PLC0415

    if short:
        c = torch.cuda.is_available()
        print("torch", torch.__version__, "cuda.is_available", c)
        return 0 if c else 1

    c = torch.cuda.is_available()
    vc = getattr(torch.version, "cuda", None)
    nk = "none_or_cpu_wheel"
    print("torch", torch.__version__, "cuda.is_available=", c)
    print("torch.version.cuda", vc or nk)
    print("torch.__file__=", torch.__file__)
    return 0 if c else 1


if __name__ == "__main__":
    raise SystemExit(main())
