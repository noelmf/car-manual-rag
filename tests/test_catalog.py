"""The catalogue is the picker: its ordering and its collisions are the risk."""

import pytest

from car_manual_rag.ingest import catalog

ENTRIES = [
    {"brand": "SEAT", "model": ["Ibiza"], "year": ["2023"], "edition": ["11.22"]},
    {"brand": "SEAT", "model": ["Ibiza"], "year": ["2023"], "edition": ["06.22"]},
    {
        "brand": "SEAT",
        "model": ["Leon", "Leon Sportstourer"],
        "year": ["2026"],
        "edition": ["11.25"],
    },
]


@pytest.fixture
def manuals():
    entries = [dict(e) for e in ENTRIES]
    for e in entries:
        e["manual_id"] = catalog.manual_id(e)
    return entries


class TestManualId:
    def test_a_space_becomes_a_dash(self):
        assert catalog.slug("Leon SC") == "Leon-SC"

    def test_several_models_share_one_id(self):
        assert catalog.manual_id(ENTRIES[2]) == "SEAT_Leon+Leon-Sportstourer_11.25"

    def test_the_id_is_stable_for_the_same_entry(self):
        assert catalog.manual_id(ENTRIES[0]) == catalog.manual_id(dict(ENTRIES[0]))


class TestOrdering:
    def test_editions_sort_by_date_not_alphabetically(self):
        # 'MM.YY': 11.22 is newer than 06.22, and a text sort would disagree.
        assert catalog.newest_first(["06.22", "11.22"]) == ["11.22", "06.22"]

    def test_a_year_boundary_is_respected(self):
        assert catalog.newest_first(["11.22", "06.23"]) == ["06.23", "11.22"]

    def test_an_unexpected_format_does_not_raise(self):
        assert len(catalog.newest_first(["sin formato", "11.22"])) == 2


class TestPicker:
    def test_a_multi_model_manual_appears_under_every_model(self, manuals):
        assert catalog.resolve(manuals, "SEAT", "Leon", "2026", "11.25") == catalog.resolve(
            manuals, "SEAT", "Leon Sportstourer", "2026", "11.25"
        )

    def test_more_paths_than_manuals(self, manuals):
        assert len(catalog.paths(manuals)) == 4 and len(manuals) == 3

    def test_options_walks_down_the_levels(self, manuals):
        assert catalog.options(manuals) == ["SEAT"]
        assert "Ibiza" in catalog.options(manuals, "SEAT")
        assert catalog.options(manuals, "SEAT", "Ibiza") == ["2023"]
        assert catalog.options(manuals, "SEAT", "Ibiza", "2023") == ["11.22", "06.22"]

    def test_an_unknown_choice_lists_the_valid_ones(self, manuals):
        with pytest.raises(LookupError, match="Panda"):
            catalog.options(manuals, "SEAT", "Panda")

    def test_an_incomplete_selection_has_no_manual(self, manuals):
        with pytest.raises(LookupError):
            catalog.resolve(manuals, "SEAT", "Ibiza", "1999", "01.01")


class TestValidate:
    def test_a_sound_catalogue_with_its_chunks_has_no_problems(self, manuals, tmp_path):
        for m in manuals:
            (tmp_path / f"{m['manual_id']}.jsonl").write_text("{}", encoding="utf-8")
        assert catalog.validate(manuals, tmp_path) == []

    def test_a_missing_chunk_file_is_reported(self, manuals, tmp_path):
        problems = catalog.validate(manuals, tmp_path)
        assert len(problems) == 3 and "crag-chunk" in problems[0]

    def test_a_chunk_file_with_no_catalogue_entry_is_reported(self, manuals, tmp_path):
        for m in manuals:
            (tmp_path / f"{m['manual_id']}.jsonl").write_text("{}", encoding="utf-8")
        (tmp_path / "SEAT_Fantasma_01.01.jsonl").write_text("{}", encoding="utf-8")
        assert any("Fantasma" in p for p in catalog.validate(manuals, tmp_path))

    def test_two_manuals_on_one_filter_path_collide(self, manuals, tmp_path):
        clash = dict(ENTRIES[0], edition=["11.22"], manual_id="SEAT_Ibiza_otro")
        assert any("matches 2 manuals" in p for p in catalog.validate([*manuals, clash], tmp_path))
