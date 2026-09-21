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


class TestClaimsAndGraph:
    """The Milestone 4 surface: claims, entities, events, timelines."""

    @pytest.fixture
    def demo_db(self, db, capsys) -> str:
        assert run(db, "demo") == 0
        capsys.readouterr()
        return db

    def test_claim_lifecycle_through_the_cli(self, demo_db, capsys) -> None:
        assert run(
            demo_db, "claim", "new", "investigation:1", "SMR costs have risen above projections"
        ) == 0
        assert "claim:1" in capsys.readouterr().out

        assert run(
            demo_db, "claim", "link", "investigation:1", "claim:1", "evidence:8",
            "--stance", "supports",
        ) == 0
        assert "linked evidence:8" in capsys.readouterr().out

        assert run(demo_db, "--json", "claim", "show", "claim:1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["status"] in {"supported", "insufficient_evidence"}
        assert payload["support"]["independent_sources"] == 1

    def test_a_fabricated_excerpt_is_refused_at_the_command_line(self, demo_db, capsys) -> None:
        run(demo_db, "claim", "new", "investigation:1", "A claim about costs")
        capsys.readouterr()
        assert run(
            demo_db, "claim", "link", "investigation:1", "claim:1", "evidence:8",
            "--excerpt", "costs have tripled according to leaked documents",
        ) == 1
        assert "does not appear" in capsys.readouterr().err

    def test_a_verified_excerpt_is_accepted(self, demo_db, capsys) -> None:
        run(demo_db, "claim", "new", "investigation:1", "The target price rose")
        capsys.readouterr()
        assert run(
            demo_db, "claim", "link", "investigation:1", "claim:1", "evidence:8",
            "--excerpt", "The target price for power from the plant had risen",
        ) == 0
        assert "excerpt checked" in capsys.readouterr().out

    def test_syndicated_support_is_reported_as_one_source(self, demo_db, capsys) -> None:
        run(demo_db, "claim", "new", "investigation:1", "The project was terminated")
        capsys.readouterr()
        run(demo_db, "claim", "link", "investigation:1", "claim:1", "evidence:8")
        run(demo_db, "claim", "link", "investigation:1", "claim:1", "evidence:9")
        output = capsys.readouterr().out
        assert "adds no independent weight" in output

        assert run(demo_db, "--json", "claim", "show", "claim:1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["support"]["documents"]) == 2
        assert payload["support"]["independent_sources"] == 1

    def test_open_questions_name_what_is_missing(self, demo_db, capsys) -> None:
        run(demo_db, "claim", "new", "investigation:1", "The project was terminated")
        run(demo_db, "claim", "link", "investigation:1", "claim:1", "evidence:8")
        capsys.readouterr()
        assert run(demo_db, "--json", "questions", "investigation:1") == 0
        questions = json.loads(capsys.readouterr().out)
        assert questions[0]["claim_id"] == "claim:1"
        assert any("counterevidence" in gap for gap in questions[0]["gaps"])

    def test_entities_are_registered_from_document_metadata(self, demo_db, capsys) -> None:
        assert run(demo_db, "entities", "investigation:1", "--from-documents") == 0
        output = capsys.readouterr().out
        assert "registered" in output
        assert "person" in output and "organization" in output

    def test_events_require_evidence(self, demo_db, capsys) -> None:
        assert run(
            demo_db, "event", "investigation:1", "Something happened",
            "--evidence", "evidence:8", "--date", "2023-11-08", "--precision", "day",
        ) == 0
        assert "event:1" in capsys.readouterr().out

        assert run(
            demo_db, "event", "investigation:1", "Undated but precise",
            "--evidence", "evidence:8", "--precision", "day",
        ) == 1
        assert "refused" in capsys.readouterr().err

    def test_timeline_collapses_copies_into_one_entry(self, demo_db, capsys) -> None:
        assert run(demo_db, "--json", "timeline", "investigation:1", "--publications") == 0
        entries = json.loads(capsys.readouterr().out)
        assert entries
        assert all(entry["independent_sources"] == 1 for entry in entries)
        grouped = [entry for entry in entries if len(entry["evidence"]) > 1]
        assert grouped, "the syndicated copy shares an entry with its original"


class TestAutonomousRun:
    """The Milestone 5-6 surface: plan, work, recurse, stop."""

    @pytest.fixture
    def investigated(self, db, capsys) -> str:
        assert run(
            db, "--offline", "investigate",
            "Are small modular reactors competitive for AI data centres?",
            "--max-tasks", "3", "--steps", "5",
        ) == 0
        capsys.readouterr()
        return db

    def test_an_investigation_plans_works_and_stops(self, db, capsys) -> None:
        assert run(
            db, "--offline", "investigate", "Are SMRs competitive for data centres?",
            "--max-tasks", "3", "--steps", "5",
        ) == 0
        output = capsys.readouterr().out
        assert "plan:" in output
        assert "academic:" in output
        assert "stopped:" in output
        assert "independent sources" in output

    def test_the_run_is_reported_as_structured_data(self, db, capsys) -> None:
        assert run(
            db, "--offline", "--json", "investigate",
            "Are SMRs competitive for data centres?", "--max-tasks", "2", "--steps", "4",
        ) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["tasks_run"] == 2
        assert payload["stop_reason"]
        assert payload["tasks"][0]["role"]

    def test_the_task_tree_records_who_spawned_what(self, investigated, capsys) -> None:
        assert run(investigated, "--json", "tasks", "investigation:1") == 0
        tasks = json.loads(capsys.readouterr().out)
        assert tasks
        assert all(task["status"] in {"completed", "failed", "skipped", "pending"} for task in tasks)
        assert any(task["summary"] for task in tasks)

    def test_status_explains_where_the_run_stands(self, investigated, capsys) -> None:
        assert run(investigated, "--json", "status", "investigation:1") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["stop_reason"]
        assert payload["documents"] > 0
        assert payload["independent_sources"] <= payload["documents"]
        assert set(payload["stopping_rules"]) == {
            "budget", "runtime", "open_tasks", "sources", "diminishing_returns", "evidence"
        }

    def test_claims_made_autonomously_are_inspectable(self, investigated, capsys) -> None:
        assert run(investigated, "--json", "claim", "list", "investigation:1") == 0
        claims = json.loads(capsys.readouterr().out)
        assert claims
        assert all(claim["explanation"] for claim in claims)

    def test_a_missing_credential_is_reported_not_worked_around(self, db, capsys) -> None:
        # Without --offline and without a configured key, the run declines to
        # start rather than silently choosing another model.
        assert run(db, "investigate", "a question") == 1
        assert "no model available" in capsys.readouterr().err


class TestReportCommand:
    def test_a_report_is_written_from_the_record(self, db, capsys) -> None:
        assert run(
            db, "--offline", "investigate", "Are SMRs competitive for data centres?",
            "--max-tasks", "3", "--steps", "5",
        ) == 0
        capsys.readouterr()
        assert run(db, "report", "investigation:1", "--no-summary") == 0
        report = capsys.readouterr().out
        assert "## Key findings" in report
        assert "## Sources" in report
        assert "independent sources" in report

    def test_the_report_can_be_written_to_a_file(self, db, tmp_path, capsys) -> None:
        run(db, "--offline", "investigate", "Are SMRs competitive for data centres?",
            "--max-tasks", "2", "--steps", "4")
        capsys.readouterr()
        destination = tmp_path / "report.md"
        assert run(db, "report", "investigation:1", "--no-summary", "--output", str(destination)) == 0
        assert destination.read_text().startswith("# ")

    def test_report_data_is_available_as_json(self, db, capsys) -> None:
        run(db, "--offline", "investigate", "Are SMRs competitive for data centres?",
            "--max-tasks", "2", "--steps", "4")
        capsys.readouterr()
        assert run(db, "--json", "report", "investigation:1", "--no-summary") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["counts"]["independent_sources"] <= payload["counts"]["documents"]
        assert payload["sources"]
        assert all(source["provenance"] for source in payload["sources"])
