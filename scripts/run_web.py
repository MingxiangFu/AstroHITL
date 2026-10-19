"""Start the local feature-mapping UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hitl_features.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="HITL custom-feature web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--sessions", default="web_sessions")
    args = parser.parse_args()
    app = create_app(ROOT / args.sessions)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
