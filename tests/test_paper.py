"""The technical report: every number traceable to a file (kortyn.paper), generated tables current, README."""

from __future__ import annotations

from pathlib import Path

import pytest

from kortyn import paper

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
SUMMARY = ROOT / "paper" / "summary.md"


@pytest.fixture(scope="module")
def build() -> paper.Build:
    return paper.build()


@pytest.fixture(scope="module")
def report() -> str:
    return paper.REPORT.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------- formatting
def test_formatting() -> None:
    assert paper.pct(0.473) == "47%"
    assert paper.pct(0.3166, 1) == "31.7%"
    assert paper.signed(-1.56) == "−1.6"
    assert paper.signed(3.4) == "+3.4"
    assert paper.signed(0.04) == "±0.0"
    assert paper.pp(-0.0156) == "−1.6 pp"
    assert paper.pp(0.03, 0) == "+3 pp"
    assert paper.pp_interval(-0.061, 0.034) == "[−6.1, +3.4] pp"
    assert paper.ms(1250) == "1,250 ms"
    assert paper.p_value(0.126) == "p = 0.13"
    assert paper.p_value(0.004) == "p < 0.01"


def test_facts_reject_duplicate_keys() -> None:
    facts = paper.Facts()
    facts.add("a", "1%", "results/x.json", "field")
    with pytest.raises(paper.PaperError):
        facts.add("a", "2%", "results/x.json", "field")
    assert facts.texts() == {"1%"}


# --------------------------------------------------------------------------------------------- checks
def test_unbacked_finds_numbers_without_a_source() -> None:
    facts = paper.Facts()
    facts.add("x", "47%", "results/x.json", "rate")
    facts.add("r", "5–7 pp", "results/x.json", "range")
    text = "Success was 47% and fell by 5–7 pp; it lost 7 pp and needed 30 ms.\n"
    assert paper.unbacked(text, facts) == ["7 pp", "30 ms"]


def test_unbacked_ignores_tables_code_and_references() -> None:
    facts = paper.Facts()
    text = (
        "Prose without numbers.\n<!-- table:t -->\n| 5% |\n<!-- /table:t -->\n"
        "```bash\nsleep 30 s\n```\n## References\n[1] 2023, 12 pp.\n"
    )
    assert paper.unbacked(text, facts) == []


def test_line_breaks_inside_a_quantity_are_normalized() -> None:
    facts = paper.Facts()
    facts.add("c", "1,250 ms", "kortyn/experiments/x.yaml", "budget")
    assert paper.unbacked("a budget of 1,250\nms.", facts) == []
    assert paper.unquoted("a budget of 1,250\nms.", facts) == []


def test_replace_tables_fills_an_empty_block_and_rejects_an_open_one() -> None:
    filled = paper.replace_tables("a\n<!-- table:t -->\n<!-- /table:t -->\nb\n", {"t": "| x |"})
    assert filled == "a\n<!-- table:t -->\n| x |\n<!-- /table:t -->\nb\n"
    with pytest.raises(paper.PaperError):
        paper.replace_tables("a\n<!-- table:t -->\nno end\n", {"t": "| x |"})


def test_replace_tables_is_idempotent() -> None:
    text = "intro\n<!-- table:t -->\nold\n<!-- /table:t -->\nend\n"
    once = paper.replace_tables(text, {"t": "| new |"})
    assert "| new |" in once
    assert "old" not in once
    assert paper.replace_tables(once, {"t": "| new |"}) == once
    assert paper.table_names(once) == ["t"]


# --------------------------------------------------------------------------------------------- the report
def test_report_is_up_to_date(capsys: pytest.CaptureFixture[str]) -> None:
    assert paper.main(["--check"]) == 0, "run python -m kortyn.paper"
    capsys.readouterr()


def test_report_quotes_every_fact(build: paper.Build, report: str) -> None:
    assert paper.unquoted(report, build.facts) == []


def test_every_number_in_the_report_has_a_source(build: paper.Build, report: str) -> None:
    assert paper.unbacked(report, build.facts) == []


def test_every_fact_names_an_existing_file(build: paper.Build) -> None:
    for fact in build.facts.items.values():
        assert (ROOT / fact.source).exists(), fact


def test_report_holds_every_generated_table(build: paper.Build, report: str) -> None:
    assert set(paper.table_names(report)) == set(build.tables)


def test_report_shows_every_figure(build: paper.Build, report: str) -> None:
    for name in build.figures:
        assert (paper.FIGURES / name).exists(), name
        assert f"figures/{name}" in report, name


def test_readme_numbers_have_a_source(build: paper.Build) -> None:
    """The README quotes the report's numbers; each must be a fact of the report."""
    assert paper.unbacked(README.read_text(encoding="utf-8"), build.facts) == []


def test_readme_links_exist() -> None:
    import re

    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+)\)", text):
        if target.startswith("http"):
            continue
        assert (ROOT / target).exists(), target


def test_summary_is_short_and_its_numbers_have_a_source(build: paper.Build) -> None:
    """paper/summary.md is shared next to the link: at most 200 words, every number a fact of the report."""
    import re

    text = SUMMARY.read_text(encoding="utf-8")
    body = text.split("\n", 1)[1]  # without the title
    words = re.findall(r"[^\s<>]+", re.sub(r"<https?://[^>]+>", "", body))
    assert len([w for w in words if w != "-"]) <= 200
    assert paper.unbacked(text, build.facts) == []
