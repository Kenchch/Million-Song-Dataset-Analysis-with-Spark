"""Regression check for the genre classifier's feature boundary."""

import ast
from pathlib import Path


def test_genre_label_is_excluded_from_vector_assembler_inputs():
    source = Path("src/msd_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    exclusions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "column"
        and any(isinstance(value, ast.Constant) and value.value == "label" for value in node.comparators)
    ]

    assert exclusions, "train_genre_model must exclude the numeric target column from features"
