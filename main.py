"""CLI entry point for RagBase.

Usage
-----
Ingest documents from a directory::

    python main.py ingest ./path/to/docs

Ask a question against the existing vector store::

    python main.py ask "What is the main topic of the documents?"
"""

import argparse
import sys
from pathlib import Path

from ragbase.chain import create_chain
from ragbase.config import Config
from ragbase.ingestor import Ingestor
from ragbase.io.file_discovery import discover_files
from ragbase.model import create_llm
from ragbase.retriever import create_retriever

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg"}


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest all supported documents found in *data_dir* into the vector store."""
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"Error: '{data_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    file_paths = discover_files(data_dir, exts=SUPPORTED_EXTENSIONS)

    if not file_paths:
        print(
            f"Error: no supported files ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) "
            f"found in '{data_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(file_paths)} file(s). Ingesting...")
    Ingestor().ingest(list(file_paths))
    print("Ingestion complete.")


def cmd_ask(args: argparse.Namespace) -> None:
    """Query the vector store and print the answer to stdout."""
    question: str = args.question

    llm = create_llm()
    retriever = create_retriever(llm)
    chain = create_chain(llm, retriever)

    result = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": "cli"}},
    )

    print("=== ANSWER ===")
    # result is an AIMessage when the chain ends with an LLM
    from langchain_core.messages import BaseMessage

    content = result.content if isinstance(result, BaseMessage) else str(result)
    print(content)


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="RagBase CLI — ingest documents and ask questions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest sub-command ---
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents from a directory into the vector store.",
    )
    ingest_parser.add_argument(
        "data_dir",
        help="Path to the directory containing documents to ingest.",
    )

    # --- ask sub-command ---
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question against the existing vector store.",
    )
    ask_parser.add_argument("question", help="The question to ask.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)


if __name__ == "__main__":
    main()
