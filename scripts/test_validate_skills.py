#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Unit tests for `validate_skills`.

Run via:

    uv run scripts/test_validate_skills.py

Each test creates a throwaway skill in a tempdir, runs `validate_skill`, and
asserts on the resulting report. No external test framework needed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from validate_skills import (
    MAX_BODY_LINES,
    MAX_DESCRIPTION_LEN,
    MAX_NAME_LEN,
    validate_skill,
)


def _content(
    *,
    name: str = "good-skill",
    description: str = "Does the thing. Use when the user wants the thing done.",
    body: str = "# Heading\n\nMinimal body content.\n",
) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description)}\n"
        "---\n\n"
        f"{body}"
    )


def _make_skill(parent: Path, dir_name: str, content: str | None) -> Path:
    skill_dir = parent / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    if content is not None:
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


class ValidateSkillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skills-test-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertHasError(self, errors: list[str], pattern: str) -> None:
        self.assertTrue(
            any(pattern in err for err in errors),
            f"expected an error containing {pattern!r}; got: {errors}",
        )

    def test_accepts_well_formed_skill(self) -> None:
        skill = _make_skill(self.tmp, "good-skill", _content())
        self.assertEqual(validate_skill(skill).errors, [])

    def test_flags_missing_skill_md(self) -> None:
        skill = _make_skill(self.tmp, "no-skill-md", None)
        report = validate_skill(skill)
        self.assertEqual(len(report.errors), 1)
        self.assertHasError(report.errors, "Missing SKILL.md")

    def test_flags_missing_frontmatter(self) -> None:
        skill = _make_skill(self.tmp, "no-frontmatter", "# Body only\n")
        report = validate_skill(skill)
        self.assertHasError(report.errors, "YAML frontmatter block")

    def test_flags_invalid_yaml(self) -> None:
        broken = "---\nname: : oops\n---\n\nbody\n"
        skill = _make_skill(self.tmp, "broken-yaml", broken)
        report = validate_skill(skill)
        self.assertHasError(report.errors, "YAML frontmatter is invalid")

    def test_flags_missing_name(self) -> None:
        skill = _make_skill(
            self.tmp,
            "no-name",
            '---\ndescription: "Does X. Use when Y."\n---\n\nbody\n',
        )
        report = validate_skill(skill)
        self.assertHasError(report.errors, "`name` is missing")

    def test_flags_missing_description(self) -> None:
        skill = _make_skill(
            self.tmp, "no-description", "---\nname: no-description\n---\n\nbody\n"
        )
        report = validate_skill(skill)
        self.assertHasError(report.errors, "`description` is missing")

    def test_rejects_invalid_name_format(self) -> None:
        skill = _make_skill(self.tmp, "Bad_Name", _content(name="Bad_Name"))
        report = validate_skill(skill)
        self.assertHasError(report.errors, "lowercase-with-hyphens")

    def test_rejects_too_long_name(self) -> None:
        long_name = "a" * (MAX_NAME_LEN + 1)
        skill = _make_skill(self.tmp, long_name, _content(name=long_name))
        report = validate_skill(skill)
        self.assertHasError(report.errors, "exceeds")

    def test_rejects_reserved_name_substrings(self) -> None:
        for reserved in ("claude-helper", "anthropic-tool"):
            with self.subTest(name=reserved):
                skill = _make_skill(self.tmp, reserved, _content(name=reserved))
                report = validate_skill(skill)
                self.assertHasError(report.errors, "may not contain")
                shutil.rmtree(skill)

    def test_rejects_name_directory_mismatch(self) -> None:
        skill = _make_skill(
            self.tmp, "actual-dir", _content(name="different-name")
        )
        report = validate_skill(skill)
        self.assertHasError(report.errors, "must match the skill directory")

    def test_rejects_too_long_description(self) -> None:
        skill = _make_skill(
            self.tmp,
            "verbose",
            _content(name="verbose", description="x" * (MAX_DESCRIPTION_LEN + 1)),
        )
        report = validate_skill(skill)
        self.assertHasError(report.errors, "`description` length")

    def test_rejects_too_long_body(self) -> None:
        body = "\n".join(f"line {i}" for i in range(MAX_BODY_LINES + 1)) + "\n"
        skill = _make_skill(
            self.tmp, "long-body", _content(name="long-body", body=body)
        )
        report = validate_skill(skill)
        self.assertHasError(report.errors, "body is")

    def test_accepts_body_at_exact_limit(self) -> None:
        body = "\n".join(f"line {i}" for i in range(MAX_BODY_LINES)) + "\n"
        skill = _make_skill(
            self.tmp, "at-limit", _content(name="at-limit", body=body)
        )
        self.assertEqual(validate_skill(skill).errors, [])

    def test_ignores_blank_lines_around_body(self) -> None:
        skill = _make_skill(
            self.tmp,
            "padded-body",
            _content(name="padded-body", body="\n\n\n# Title\n\n\n\n"),
        )
        self.assertEqual(validate_skill(skill).errors, [])

    def test_handles_crlf_line_endings(self) -> None:
        skill = _make_skill(
            self.tmp, "good-skill", _content().replace("\n", "\r\n")
        )
        self.assertEqual(validate_skill(skill).errors, [])


if __name__ == "__main__":
    unittest.main()
