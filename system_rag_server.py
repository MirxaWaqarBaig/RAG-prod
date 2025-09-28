#!/usr/bin/env python3
"""Thin CLI wrapper for SystemRagService."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system_rag_service import SystemRagService


def serve() -> None:
    service = SystemRagService()
    import zmq

    service_name = "text-to-text"
    zmq_url = os.environ.get("ZMQ_BACKEND_ROUTER_URL", "tcp://localhost:5560")
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(zmq_url)
    socket.send_multipart([service_name.encode()])
    print(f"[system_rag_server] Registered with ZMQ at {zmq_url}")
    print("[system_rag_server] Admin commands: !reload, !clear, !stats, !detailed")

    while True:
        message = socket.recv_multipart()
        try:
            prompt = message[3].decode()
        except Exception:
            prompt = ""

        response = service.handle_query(prompt)
        excerpt = response if len(response) <= 200 else response[:200] + "…"
        print(f"[system_rag_server] Prompt: {prompt}")
        print(f"[system_rag_server] Response: {excerpt}")

        socket.send_multipart(message[:3] + [response.encode()])


def main() -> None:
    parser = argparse.ArgumentParser(description="System Prompt RAG server CLI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the System Prompt RAG server")
    args = parser.parse_args()

    if args.command == "serve":
        serve()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
