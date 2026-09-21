"""The command line, exercised against the offline corpus."""

from __future__ import annotations

import json

import pytest

from research.cli.main import main


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "research.sqlite3")


def run(db: str, *args: str) -> int:
    return main(["--db", db, *args])


class TestInvestigationLifecycle:
    def test_new_and_list(self, db, capsys) -> None:
        assert run(db, "new", "are SMRs competitive?", "--tag", "energy") == 0
        assert "investigation:1" in capsys.readouterr().out

        assert run(db, "--json", "list") == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows[0]["id"] == "investigation:1"
        assert rows[0]["documents"] == 0

    def test_search_persists_evidence(self, db, capsys) -> None:
        run(db, "new", "are SMRs competitive?")
        capsys.readouterr()
        assert run(
            db, "--offline", "search", "investigation:1", "reactor cost", "--family", "academic"
        ) == 0
        output = capsys.readouterr().out
        assert "new" in output and "evidence:1" in output

        assert run(db, "--json", "evidence", "investigation:1") == 0
        rows = json.loads(capsys.readouterr().out)
        assert rows and rows[0]["independence"] == "independent"

    def test_stop_records_a_reason(self, db, capsys) -> None:
        run(db, "new", "q")
        capsys.readouterr()
        assert run(db, "stop", "investigation:1", "--reason", "evidence_sufficient") == 0
        assert "evidence_sufficient" in capsys.readouterr().out

    def test_unknown_document_is_a_clean_failure(self, db, capsys) -> None:
        assert run(db, "show", "evidence:999") == 2
        assert "not found" in capsys.readouterr().err


class TestDemoAndInspection:
    @pytest.fixture
    def demo_db(self, db, capsys) -> str:
        assert run(db, "demo") == 0
        capsys.readouterr()
        return db

    def test_demo_builds_a_mixed_corpus(self, db, capsys) -> None:
        assert run(db, "demo") == 0
        output = capsys.readouterr().out
        assert "academic" in output and "news" in output
        assert "independent sources" in output
        assert "syndicated_copy" in output

    def test_show_traces_a_document_to_its_origin(self, demo_db, capsys) -> None:
        assert run(demo_db, "show", "evidence:1") == 0
        output = capsys.readouterr().out
        assert "provenance" in output
        assert "content hash" in output
        assert "external, untrusted" in output

    def test_show_explains_why_a_copy_is_not_independent(self, demo_db, capsys) -> None:
        run(demo_db, "--json", "evidence", "investigation:1")
        rows = json.loads(capsys.readouterr().out)
        copies = [row for row in rows if row["independence"] != "independent"]
        assert copies
        assert run(demo_db, "show", copies[0]["id"]) == 0
        output = capsys.readouterr().out
        assert "NOT independent" in output
        assert "reason" in output

    def test_independence_report_collapses_copies(self, demo_db, capsys) -> None:
        assert run(demo_db, "independence", "investigation:1") == 0
        output = capsys.readouterr().out
        assert "independent sources" in output
        assert "overstate corroboration" in output

    def test_graph_reports_the_citation_structure(self, demo_db, capsys) -> None:
        assert run(demo_db, "--json", "graph", "investigation:1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["stats"]["edges"] > 0
        assert payload["most_cited"]

    def test_activity_shows_searches_and_fetches(self, demo_db, capsys) -> None:
        assert run(demo_db, "--json", "activity", "investigation:1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["searches"] and payload["fetches"]
        assert all(search["status"] == "ok" for search in payload["searches"])

    def test_budget_reports_what_was_spent(self, demo_db, capsys) -> None:
        assert run(demo_db, "--json", "budget", "investigation:1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["searches"]["used"] >= 4
        assert payload["documents"]["remaining"] < payload["documents"]["limit"]

    def test_citations_command_is_bounded(self, demo_db, capsys) -> None:
        assert run(
            demo_db, "--offline", "--json", "citations", "investigation:1", "evidence:1",
            "--direction", "backward", "--depth", "1", "--max-documents", "2",
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["max_depth"] == 1
        assert payload["documents_added"] <= 2


class TestSources:
    def test_sources_lists_capabilities(self, db, capsys) -> None:
        assert run(db, "--json", "sources") == 0
        rows = json.loads(capsys.readouterr().out)
        names = {row["name"] for row in rows}
        assert {"openalex", "crossref", "arxiv", "semantic_scholar", "gdelt", "fetch"} <= names
        openalex = next(row for row in rows if row["name"] == "openalex")
        assert openalex["citing"] and openalex["references"]
        assert not openalex["needs_key"]

    def test_offline_mode_uses_the_corpus_sources(self, db, capsys) -> None:
        assert run(db, "--offline", "--json", "sources") == 0
        names = {row["name"] for row in json.loads(capsys.readouterr().out)}
        assert names == {"offline_academic", "offline_web", "fetch"}

    def test_there_is_no_command_that_executes_anything(self) -> None:
        from research.cli.main import build_parser

        parser = build_parser()
        commands = set(parser._subparsers._group_actions[0].choices)  # type: ignore[attr-defined]
        assert not commands & {"run", "exec", "shell", "eval", "python"}
