"""Build a flat category graph from one column of a label file."""

from __future__ import annotations

import csv
import io

from hitl_features.graph import CategoryGraph

_BLANK = {"", "NA", "NA_Value", "nan", "None"}


def parse_label_csv(text: str, index_column: str, label_column: str) -> list[tuple[int, str]]:
    """Read catalog index and label pairs. Blank labels are skipped.

    A repeated index keeps the last label in the file.
    """
    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError("label file has no header")
    if index_column not in fields or label_column not in fields:
        raise ValueError("choose an index column and a label column from the file header")
    merged: dict[int, str] = {}
    for row in reader:
        label = str(row.get(label_column) or "").strip()
        if label in _BLANK:
            continue
        raw_index = str(row.get(index_column) or "").strip()
        if raw_index in _BLANK:
            continue
        merged[int(float(raw_index))] = label
    return sorted(merged.items())


def hierarchy_fields(fieldnames: list[str] | None) -> list[str]:
    """Return ``label_1``, ``label_2``, ... in numeric order."""
    names = list(fieldnames or [])
    found: list[tuple[int, str]] = []
    for name in names:
        if not name.startswith("label_"):
            continue
        suffix = name[len("label_") :]
        if suffix.isdigit() and int(suffix) >= 1:
            found.append((int(suffix), name))
    found.sort()
    expected = 1
    ordered: list[str] = []
    for number, name in found:
        if number != expected:
            break
        ordered.append(name)
        expected += 1
    return ordered


def parse_hierarchy_csv(
    text: str,
    index_column: str | None = None,
    label_columns: list[str] | None = None,
) -> list[tuple[int, list[str]]]:
    """Read a catalog index and an ordered chain of label columns.

    With no column choice, the file must use ``index`` and ``label_1``,
    ``label_2``, ... A chosen list uses those headers in level order. A row
    is the chain until the first blank. A repeated index keeps the last chain.
    """
    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError("label file has no header")
    if label_columns is None:
        chosen_index = "index" if index_column is None else str(index_column)
        if chosen_index not in fields:
            raise ValueError("label file needs an index column")
        label_fields = hierarchy_fields(fields)
        if not label_fields:
            raise ValueError("label file needs label_1")
    else:
        chosen_index = str(index_column or "").strip()
        label_fields = [str(name).strip() for name in label_columns if str(name).strip()]
        if not chosen_index:
            raise ValueError("choose an index column")
        if chosen_index not in fields:
            raise ValueError("choose an index column from the file header")
        if not label_fields:
            raise ValueError("choose at least one label column")
        if chosen_index in label_fields or len(label_fields) != len(set(label_fields)):
            raise ValueError("label columns must be distinct and different from the index")
        missing = [name for name in label_fields if name not in fields]
        if missing:
            raise ValueError("choose label columns from the file header")
    merged: dict[int, list[str]] = {}
    for row in reader:
        raw_index = str(row.get(chosen_index) or "").strip()
        if raw_index in _BLANK:
            continue
        chain: list[str] = []
        for field in label_fields:
            label = str(row.get(field) or "").strip()
            if label in _BLANK:
                break
            chain.append(label)
        if not chain:
            continue
        merged[int(float(raw_index))] = chain
    if not merged:
        raise ValueError("label file assigned no images")
    return sorted(merged.items())


def _child_named(graph: CategoryGraph, parent_id: str | None, name: str) -> str | None:
    for category in graph.children(parent_id):
        if category.name == name and not graph.is_recycled_category(category.id):
            return category.id
    return None


def merge_hierarchy(graph: CategoryGraph, rows: list[tuple[int, list[str]]]) -> None:
    """Add each chain to ``graph``. A name already under that parent is reused."""
    for index, chain in rows:
        parent_id = None
        for name in chain:
            found = _child_named(graph, parent_id, name)
            if found is None:
                found = graph.create_category(name, parent_id=parent_id)
            parent_id = found
        graph.assign(int(index), parent_id)


def merge_labels(graph: CategoryGraph, pairs: list[tuple[int, str]]) -> None:
    """Add flat labels at the root. An existing root name receives the sample."""
    for index, label in pairs:
        found = _child_named(graph, None, label)
        if found is None:
            found = graph.create_category(label)
        graph.assign(int(index), found)


def graph_from_hierarchy(rows: list[tuple[int, list[str]]]) -> CategoryGraph:
    """One category per label path. Deeper columns become children."""
    if not rows:
        raise ValueError("label file assigned no images")
    graph = CategoryGraph()
    merge_hierarchy(graph, rows)
    return graph


def export_hierarchy_csv(graph: CategoryGraph) -> str:
    """Write each placed sample with ancestors filled out to its own label."""
    paths: list[tuple[int, list[str]]] = []
    depth = 1
    for index in sorted(graph.assignments):
        if graph.is_recycled_sample(index):
            continue
        category_id = graph.category_of(index)
        if category_id is None:
            continue
        names = [graph.categories[item].name for item in graph.path(category_id)]
        paths.append((int(index), names))
        depth = max(depth, len(names))
    header = ["index"] + [f"label_{number}" for number in range(1, depth + 1)]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for index, names in paths:
        writer.writerow([index, *names, *([""] * (depth - len(names)))])
    return buffer.getvalue()


def graph_from_labels(pairs: list[tuple[int, str]]) -> CategoryGraph:
    """One root category per distinct label, in label order."""
    if not pairs:
        raise ValueError("label file assigned no images")
    graph = CategoryGraph()
    category_ids: dict[str, str] = {}
    for label in sorted({label for _, label in pairs}):
        category_ids[label] = graph.create_category(label)
    for index, label in pairs:
        graph.assign(index, category_ids[label])
    return graph
