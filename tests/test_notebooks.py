import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))
# The classes of secret the original cluster notebooks carried: Azure SAS query
# parameters, storage account keys, and academic account addresses.
#
# The course account name that was scrubbed is deliberately not written here.
# It used to be a literal in this pattern, which meant the check that existed to
# keep the identifier out of the repository was the one place still publishing
# it -- on every clone, in the file whose job was to prevent exactly that.
SENSITIVE_REMNANTS = re.compile(
    r"sp=racwdl"  # SAS permission string
    r"|[?&]sig="  # SAS signature
    r"|[?&]se=\d{4}-\d{2}-\d{2}"  # SAS expiry
    r"|AccountKey="  # storage account key
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.ac\.[a-z]{2}",  # academic address
    re.IGNORECASE,
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_is_sanitized(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert not SENSITIVE_REMNANTS.search(path.read_text(encoding="utf-8"))

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs", []) == [] for cell in code_cells)
