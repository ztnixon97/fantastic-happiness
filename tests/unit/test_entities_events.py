"""Entity resolution and evidence-backed events."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research.acquisition.pipeline import EvidenceAcquirer
from research.errors import IntegrityError
from research.graph.entities import EntityRegistry, is_weak_name
from research.graph.timelines import Timeline
from research.models.common import Provenance, SourceFamily, SourceType
from research.models.entity import EntityIdentifier, EntityType, MatchConfidence
from research.models.event import DatePrecision
from research.normalize.document import build_document
from research.operations.timeline import TimelineOperations

ORCID = EntityIdentifier("orcid", "0000-0002-1825-0097")
OTHER_ORCID = EntityIdentifier("orcid", "0000-0003-1111-2222")


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


@pytest.fixture
def registry(store, investigation) -> EntityRegistry:
    return EntityRegistry(store, investigation.id, task_id="task:1")


@pytest.fixture
def documents(store, investigation):
    acquirer = EvidenceAcquirer(store)

    def add(title: str, text: str, **kwargs):
        document = build_document(
            provider="test",
            source_type=kwargs.pop("source_type", SourceType.ACADEMIC_PEER_REVIEWED),
            source_family=kwargs.pop("family", SourceFamily.ACADEMIC),
            provenance=Provenance(provider="test"),
            title=title,
            text=text,
            url=kwargs.pop("url", f"https://example.test/{abs(hash(title)) % 9973}"),
            **kwargs,
        )
        return acquirer.persist(document, investigation_id=investigation.id).document

    return add


class TestEntityResolution:
    def test_an_authoritative_identifier_decides_identity(self, registry) -> None:
        first = registry.resolve("Rachel Okafor", EntityType.PERSON, identifiers=[ORCID])
        second = registry.resolve("R. Okafor", EntityType.PERSON, identifiers=[ORCID])
        assert first.created and first.confidence is MatchConfidence.DETERMINISTIC
        assert not second.created
        assert second.confidence is MatchConfidence.DETERMINISTIC
        assert second.entity.id == first.entity.id

    def test_conflicting_identifiers_mean_different_entities(self, registry) -> None:
        first = registry.resolve("Rachel Okafor", EntityType.PERSON, identifiers=[ORCID])
        second = registry.resolve(
            "Rachel Okafor", EntityType.PERSON, identifiers=[OTHER_ORCID]
        )
        assert second.created
        assert second.entity.id != first.entity.id

    def test_names_match_heuristically_and_say_so(self, registry) -> None:
        first = registry.resolve("NuScale Power", EntityType.ORGANIZATION)
        second = registry.resolve("nuscale power", EntityType.ORGANIZATION)
        assert second.entity.id == first.entity.id
        assert second.confidence is MatchConfidence.HEURISTIC

    def test_aliases_resolve_to_the_same_entity(self, registry) -> None:
        entity = registry.resolve(
            "NuScale Power Corporation", EntityType.ORGANIZATION, aliases=["NuScale"]
        )
        again = registry.resolve("NuScale", EntityType.ORGANIZATION)
        assert again.entity.id == entity.entity.id

    def test_ambiguous_matches_are_not_merged(self, store, investigation, registry) -> None:
        # Two distinct entities that share an alias: the registry must refuse
        # to choose rather than pick one.
        first = store.entities.create(
            investigation.id, EntityType.ORGANIZATION, "Acme Energy", aliases=["Acme"]
        )
        second = store.entities.create(
            investigation.id, EntityType.ORGANIZATION, "Acme Nuclear", aliases=["Acme"]
        )
        resolution = registry.resolve("Acme", EntityType.ORGANIZATION, create=False)
        assert resolution.confidence is MatchConfidence.AMBIGUOUS
        assert resolution.entity is None
        assert {candidate.id for candidate in resolution.candidates} == {first.id, second.id}

    def test_a_name_only_match_on_initials_is_flagged_for_review(self, registry) -> None:
        registry.resolve("J. Smith", EntityType.PERSON)
        registry.resolve("J. Smith", EntityType.PERSON)
        flagged = registry.needs_review()
        assert [entity.name for entity in flagged] == ["J. Smith"]

    def test_full_names_are_not_flagged(self, registry) -> None:
        registry.resolve("Rachel Okafor", EntityType.PERSON)
        registry.resolve("Rachel Okafor", EntityType.PERSON)
        assert registry.needs_review() == []

    @pytest.mark.parametrize(
        "name,weak", [("J. Smith", True), ("Smith", True), ("Rachel Okafor", False)]
    )
    def test_weak_name_detection(self, name: str, weak: bool) -> None:
        assert is_weak_name(name) is weak

    def test_entities_are_scoped_to_an_investigation(self, store) -> None:
        first = store.investigations.create("one")
        second = store.investigations.create("two")
        EntityRegistry(store, first.id).resolve("NuScale", EntityType.ORGANIZATION)
        resolution = EntityRegistry(store, second.id).resolve(
            "NuScale", EntityType.ORGANIZATION
        )
        assert resolution.created


class TestDocumentEntities:
    def test_authors_and_venue_are_registered_with_provenance(
        self, store, investigation, registry, documents
    ) -> None:
        document = documents(
            "Levelized cost projections",
            "We review 42 published cost estimates for small modular reactors.",
            authors=["R. Okafor", "L. Meng"],
            publisher="Energy Policy",
        )
        resolutions = registry.register_document(document)
        names = {resolution.entity.name for resolution in resolutions}
        assert names == {"R. Okafor", "L. Meng", "Energy Policy"}
        author = next(r.entity for r in resolutions if r.entity.name == "L. Meng")
        assert registry.documents_for(author.id) == [document.id]
        edges = store.relationships.for_object(investigation.id, author.id)
        assert edges[0]["predicate"] == "AUTHORED_BY"
        assert edges[0]["evidence_document_id"] == document.id

    def test_orcids_in_metadata_are_used_when_present(self, registry, documents) -> None:
        document = documents(
            "A paper",
            "Body text about reactor economics and construction costs over time.",
            authors=["Rachel Okafor"],
            metadata={"author_orcids": {"Rachel Okafor": "https://orcid.org/0000-0002-1825-0097"}},
        )
        resolution = registry.register_document(document)[0]
        assert resolution.confidence is MatchConfidence.DETERMINISTIC
        assert resolution.entity.identifier("orcid") == "0000-0002-1825-0097"


class TestEvents:
    def test_an_event_needs_evidence(self, store, investigation) -> None:
        operations = TimelineOperations(store, investigation_id=investigation.id)
        with pytest.raises(IntegrityError) as exc:
            operations.record_event("Something happened", evidence_ids=[])
        assert "at least one evidence document" in str(exc.value)

    def test_a_dated_event_must_state_its_precision(self, store, investigation, documents) -> None:
        document = documents("Report", "The commission issued its approval on Thursday.")
        operations = TimelineOperations(store, investigation_id=investigation.id)
        with pytest.raises(IntegrityError):
            operations.record_event(
                "Approval issued",
                evidence_ids=[document.id],
                date_start=dt(2025, 5, 29),
                date_precision=DatePrecision.UNKNOWN,
            )

    def test_events_cannot_end_before_they_start(self, store, investigation, documents) -> None:
        document = documents("Report", "The project ran from 2015 until it was cancelled.")
        operations = TimelineOperations(store, investigation_id=investigation.id)
        with pytest.raises(IntegrityError):
            operations.record_event(
                "Project ran",
                evidence_ids=[document.id],
                date_start=dt(2020),
                date_end=dt(2015),
                date_precision=DatePrecision.YEAR,
            )

    def test_recording_an_event_links_entities_and_evidence(
        self, store, investigation, registry, documents
    ) -> None:
        document = documents("Report", "The commission issued a standard design approval.")
        entity = registry.resolve("US NRC", EntityType.ORGANIZATION).entity
        operations = TimelineOperations(store, investigation_id=investigation.id, task_id="task:9")
        event = operations.record_event(
            "NRC issues standard design approval",
            evidence_ids=[document.id],
            date_start=dt(2025, 5, 29),
            date_precision=DatePrecision.DAY,
            entity_ids=[entity.id],
        )
        stored = store.events.get(event.id)
        assert stored.evidence_ids == [document.id]
        assert stored.entity_ids == [entity.id]
        predicates = {
            row["predicate"] for row in store.relationships.list(investigation.id)
        }
        assert {"PARTICIPATED_IN", "ABOUT"} <= predicates


class TestTimelines:
    def test_undated_events_sort_last_and_are_labelled(
        self, store, investigation, documents
    ) -> None:
        document = documents("Report", "The commission issued a standard design approval.")
        operations = TimelineOperations(store, investigation_id=investigation.id)
        operations.record_event(
            "Approval issued",
            evidence_ids=[document.id],
            date_start=dt(2025, 5, 29),
            date_precision=DatePrecision.DAY,
        )
        operations.record_event("Cost estimate revised at some point", evidence_ids=[document.id])
        entries = operations.build_timeline()
        assert [entry.date is None for entry in entries] == [False, True]
        assert entries[1].precision is DatePrecision.UNKNOWN

    def test_publication_entries_count_one_per_independent_source(
        self, store, investigation, documents
    ) -> None:
        wire = (
            "WASHINGTON (Reuters) - The utility group said on Wednesday it had agreed to "
            "terminate the flagship reactor project after subscriptions fell short."
        )
        documents(
            "Project terminated",
            wire,
            source_type=SourceType.ORIGINAL_NEWS_REPORTING,
            family=SourceFamily.NEWS,
            url="https://www.reuters.test/a",
            published_at=dt(2023, 11, 8),
        )
        documents(
            "Project terminated",
            wire,
            source_type=SourceType.SECONDARY_NEWS_REPORTING,
            family=SourceFamily.NEWS,
            url="https://www.gazette.test/a",
            published_at=dt(2023, 11, 9),
        )
        timeline = Timeline(store, investigation.id)
        entries = [
            entry for entry in timeline.build(include_publications=True)
            if entry.kind == "publication"
        ]
        assert len(entries) == 1, "a syndicated copy is not a second point in time"
        assert entries[0].date == dt(2023, 11, 8), "dated by the first copy to appear"
        assert "further cop" in entries[0].description

    def test_first_appearance_follows_the_independence_group(
        self, store, investigation, documents
    ) -> None:
        wire = (
            "LONDON (Reuters) - Regulators opened a hearing into who pays for the "
            "additional generation capacity required by new data centre load."
        )
        original = documents(
            "Regulators open hearing",
            wire,
            source_type=SourceType.ORIGINAL_NEWS_REPORTING,
            family=SourceFamily.NEWS,
            url="https://www.reuters.test/hearing",
            published_at=dt(2025, 3, 1),
        )
        copy = documents(
            "Regulators open hearing",
            wire,
            source_type=SourceType.SECONDARY_NEWS_REPORTING,
            family=SourceFamily.NEWS,
            url="https://www.gazette.test/hearing",
            published_at=dt(2025, 3, 4),
        )
        timeline = Timeline(store, investigation.id)
        appearance = timeline.first_appearance(copy.id)
        assert appearance is not None
        assert appearance.date == dt(2025, 3, 1)
        assert appearance.evidence_ids == [original.id]
