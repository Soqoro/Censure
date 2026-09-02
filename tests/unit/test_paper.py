from __future__ import annotations

import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PAPER_PATH = REPOSITORY_ROOT / "paper" / "censure_iclr2027.tex"
ESTIMATOR_PAPER_PATH = REPOSITORY_ROOT / "paper" / "censure_estimator.tex"
PHASE2_PLACEHOLDER_PATH = REPOSITORY_ROOT / "paper" / "phase2_results_placeholder.tex"
BIBLIOGRAPHY_PATH = REPOSITORY_ROOT / "paper" / "references.bib"


class PaperArtifactTests(unittest.TestCase):
    def test_paper_has_no_result_placeholders(self) -> None:
        paper = PAPER_PATH.read_text(encoding="utf-8")
        forbidden_literals = (
            "[TBD",
            "All result cells are placeholders",
            "Result figure placeholder",
        )
        for token in forbidden_literals:
            self.assertNotIn(token, paper)
        self.assertIsNone(re.search(r"\\(?:tbd|res)(?:\b|\{)", paper))

        for anchor in (
            "Across 696 confirmatory actor--task pairs",
            "Qwen3-8B & 211/232 & .137",
            "Gemma-3-12B & 223/232 & .000",
            "Ministral-3-14B & 219/232 & .178",
            "cross-model synthesis is retrospective",
        ):
            self.assertIn(anchor, paper)

    def test_citations_resolve_to_bibliography(self) -> None:
        paper = PAPER_PATH.read_text(encoding="utf-8")
        bibliography = BIBLIOGRAPHY_PATH.read_text(encoding="utf-8")
        cited = {
            key.strip()
            for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", paper)
            for key in group.split(",")
        }
        defined = set(re.findall(r"^@[A-Za-z]+\{([^,]+),", bibliography, flags=re.MULTILINE))
        self.assertTrue(cited)
        self.assertEqual(cited - defined, set())

    def test_environments_and_cross_references_are_balanced(self) -> None:
        paper = PAPER_PATH.read_text(encoding="utf-8")
        uncommented = "\n".join(re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in paper.splitlines())
        brace_depth = 0
        for match in re.finditer(r"(?<!\\)[{}]", uncommented):
            brace_depth += 1 if match.group() == "{" else -1
            self.assertGreaterEqual(brace_depth, 0, "closing brace precedes its opening brace")
        self.assertEqual(brace_depth, 0)

        environment_stack: list[str] = []
        for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", paper):
            if action == "begin":
                environment_stack.append(environment)
            else:
                self.assertTrue(environment_stack, f"unmatched end of {environment}")
                self.assertEqual(environment_stack.pop(), environment)
        self.assertEqual(environment_stack, [])

        labels = re.findall(r"\\label\{([^}]+)\}", paper)
        references = {
            key.strip()
            for group in re.findall(r"\\(?:c|C)?ref\{([^}]+)\}", paper)
            for key in group.split(",")
        }
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(references - set(labels), set())

    def test_estimator_paper_is_result_gated_and_method_complete(self) -> None:
        paper = ESTIMATOR_PAPER_PATH.read_text(encoding="utf-8")
        placeholder = PHASE2_PLACEHOLDER_PATH.read_text(encoding="utf-8")

        self.assertIn(r"\IfFileExists{generated/phase2_results.tex}", paper)
        self.assertIn(r"\ifphasetworesults", paper)
        self.assertNotIn(r"\newcommand{\PhaseTwoResultsAvailable}", placeholder)
        for anchor in (
            "Anytime finite-cohort target-risk certificate",
            "Capability-separated implementation",
            "Prospectively frozen experiments",
            "Phase~2 result disclosure is pending by design",
            "selected suffix audits",
        ):
            self.assertIn(anchor, paper)

    def test_estimator_paper_citations_resolve_and_structure_is_balanced(self) -> None:
        paper = ESTIMATOR_PAPER_PATH.read_text(encoding="utf-8")
        bibliography = BIBLIOGRAPHY_PATH.read_text(encoding="utf-8")
        cited = {
            key.strip()
            for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", paper)
            for key in group.split(",")
        }
        defined = set(
            re.findall(r"^@[A-Za-z]+\{([^,]+),", bibliography, flags=re.MULTILINE)
        )
        self.assertEqual(cited - defined, set())
        self.assertTrue(
            {"howard2021confidence", "jiang2016doubly", "manski2003partial"}
            <= cited
        )

        uncommented = "\n".join(
            re.split(r"(?<!\\)%", line, maxsplit=1)[0] for line in paper.splitlines()
        )
        brace_depth = 0
        for match in re.finditer(r"(?<!\\)[{}]", uncommented):
            brace_depth += 1 if match.group() == "{" else -1
            self.assertGreaterEqual(brace_depth, 0)
        self.assertEqual(brace_depth, 0)

        environment_stack: list[str] = []
        for action, environment in re.findall(r"\\(begin|end)\{([^}]+)\}", paper):
            if action == "begin":
                environment_stack.append(environment)
            else:
                self.assertTrue(environment_stack)
                self.assertEqual(environment_stack.pop(), environment)
        self.assertEqual(environment_stack, [])

        labels = re.findall(r"\\label\{([^}]+)\}", paper)
        references = {
            key.strip()
            for group in re.findall(r"\\(?:c|C)?ref\{([^}]+)\}", paper)
            for key in group.split(",")
        }
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(references - set(labels), set())


if __name__ == "__main__":
    unittest.main()
