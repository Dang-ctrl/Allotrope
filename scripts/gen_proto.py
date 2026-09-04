"""Regenerate the gRPC stubs from allotrope/rpc/allotrope.proto.

Run this after editing the .proto file:

    python scripts/gen_proto.py

The generated files are committed to the repository (so `pip install -e .` and
`pytest` work without a protoc step), but they are generated -- edit the .proto,
not allotrope_pb2*.py directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RPC_DIR = ROOT / "allotrope" / "rpc"


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{RPC_DIR}",
            f"--python_out={RPC_DIR}",
            f"--grpc_python_out={RPC_DIR}",
            str(RPC_DIR / "allotrope.proto"),
        ],
        check=True,
    )

    # grpc_tools emits a bare `import allotrope_pb2`, which only resolves when
    # the generated directory is on sys.path directly -- not true once these
    # files live inside the allotrope.rpc package. Rewrite it to a package-
    # relative import so `from allotrope.rpc import allotrope_pb2_grpc` works.
    grpc_stub = RPC_DIR / "allotrope_pb2_grpc.py"
    text = grpc_stub.read_text(encoding="utf-8")
    text = text.replace(
        "import allotrope_pb2 as allotrope__pb2",
        "from allotrope.rpc import allotrope_pb2 as allotrope__pb2",
    )
    grpc_stub.write_text(text, encoding="utf-8")
    print(f"regenerated {grpc_stub.relative_to(ROOT)} and its pb2 module")


if __name__ == "__main__":
    main()
