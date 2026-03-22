"""
Smoke tests for main.py CLI — argument parsing and core function dispatch.

External services (Ingestor, create_llm, create_retriever, create_chain) are
monkeypatched so these tests run without any model downloads or API keys.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import build_parser, cmd_ask, cmd_ingest


# ---------------------------------------------------------------------------
# discover_files / file_discovery helpers
# ---------------------------------------------------------------------------


def test_discover_files_returns_sorted_paths(tmp_path):
    """discover_files returns only matching extensions, sorted."""
    from ragbase.io.file_discovery import discover_files

    (tmp_path / "b.pdf").write_bytes(b"")
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "ignore.txt").write_bytes(b"")

    result = discover_files(tmp_path, exts={".pdf"})

    assert [p.name for p in result] == ["a.pdf", "b.pdf"]


def test_discover_files_recursive(tmp_path):
    """discover_files descends into sub-directories when recursive=True."""
    from ragbase.io.file_discovery import discover_files

    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.pdf").write_bytes(b"")
    (sub / "nested.pdf").write_bytes(b"")

    result = discover_files(tmp_path, exts={".pdf"}, recursive=True)

    assert len(result) == 2


def test_discover_files_non_recursive(tmp_path):
    """discover_files does NOT descend when recursive=False."""
    from ragbase.io.file_discovery import discover_files

    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "root.pdf").write_bytes(b"")
    (sub / "nested.pdf").write_bytes(b"")

    result = discover_files(tmp_path, exts={".pdf"}, recursive=False)

    assert len(result) == 1
    assert result[0].name == "root.pdf"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_parser_ingest_subcommand():
    """Parser correctly parses the 'ingest' subcommand."""
    parser = build_parser()
    args = parser.parse_args(["ingest", "./docs"])

    assert args.command == "ingest"
    assert args.data_dir == "./docs"


def test_parser_ask_subcommand():
    """Parser correctly parses the 'ask' subcommand."""
    parser = build_parser()
    args = parser.parse_args(["ask", "What is RAG?"])

    assert args.command == "ask"
    assert args.question == "What is RAG?"


def test_parser_requires_subcommand():
    """Parser exits with error when no subcommand is given."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------------------------------------------------------------------------
# cmd_ingest
# ---------------------------------------------------------------------------


def test_cmd_ingest_missing_directory(tmp_path, capsys):
    """cmd_ingest exits non-zero when the directory does not exist."""
    args = argparse.Namespace(data_dir=str(tmp_path / "nonexistent"))

    with pytest.raises(SystemExit) as exc_info:
        cmd_ingest(args)

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_cmd_ingest_no_supported_files(tmp_path, capsys):
    """cmd_ingest exits non-zero when no supported files are found."""
    (tmp_path / "readme.txt").write_text("hello")
    args = argparse.Namespace(data_dir=str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        cmd_ingest(args)

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "no supported files" in captured.err


def test_cmd_ingest_calls_ingestor(tmp_path, capsys):
    """cmd_ingest calls Ingestor().ingest() with the discovered file paths."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    args = argparse.Namespace(data_dir=str(tmp_path))

    mock_ingestor_instance = MagicMock()
    with patch("main.Ingestor") as MockIngestor:
        MockIngestor.return_value = mock_ingestor_instance
        cmd_ingest(args)

    mock_ingestor_instance.ingest.assert_called_once()
    call_paths = mock_ingestor_instance.ingest.call_args[0][0]
    assert any(p.name == "sample.pdf" for p in call_paths)

    captured = capsys.readouterr()
    assert "Ingestion complete" in captured.out


# ---------------------------------------------------------------------------
# cmd_ask
# ---------------------------------------------------------------------------


def test_cmd_ask_prints_answer(capsys):
    """cmd_ask invokes the chain and prints the answer with the header."""
    from langchain_core.messages import AIMessage

    fake_result = AIMessage(content="RAG stands for Retrieval-Augmented Generation.")

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_result

    args = argparse.Namespace(question="What is RAG?")

    with (
        patch("main.create_llm", return_value=MagicMock()),
        patch("main.create_retriever", return_value=MagicMock()),
        patch("main.create_chain", return_value=fake_chain),
    ):
        cmd_ask(args)

    captured = capsys.readouterr()
    assert "=== ANSWER ===" in captured.out
    assert "Retrieval-Augmented Generation" in captured.out


def test_cmd_ask_uses_cli_session(capsys):
    """cmd_ask passes session_id='cli' to chain.invoke."""
    from langchain_core.messages import AIMessage

    fake_result = AIMessage(content="answer")

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = fake_result

    args = argparse.Namespace(question="test?")

    with (
        patch("main.create_llm", return_value=MagicMock()),
        patch("main.create_retriever", return_value=MagicMock()),
        patch("main.create_chain", return_value=fake_chain),
    ):
        cmd_ask(args)

    call_args = fake_chain.invoke.call_args
    config = call_args.kwargs.get("config") or call_args.args[1]
    assert config["configurable"]["session_id"] == "cli"
