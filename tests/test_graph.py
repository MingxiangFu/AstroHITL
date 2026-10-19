import pytest

from hitl_features.graph import CategoryGraph


def test_position_and_unassign_round_trip():
    graph = CategoryGraph()
    spiral = graph.create_category("spiral")
    graph.assign(4, spiral)
    graph.set_position(spiral, 0.25, 0.5)
    graph.unassign(4)
    assert graph.category_of(4) is None
    assert graph.samples(spiral) == []
    loaded = CategoryGraph.from_dict(graph.to_dict())
    assert loaded.positions[spiral] == (0.25, 0.5)
    assert loaded.assignments == {}


def test_exclusive_assign_moves_sample_off_previous_category():
    graph = CategoryGraph()
    spiral = graph.create_category("spiral")
    smooth = graph.create_category("smooth")
    graph.assign(7, spiral)
    graph.assign(7, smooth)
    assert graph.category_of(7) == smooth
    assert graph.samples(spiral) == []
    assert graph.samples(smooth) == [7]


def test_waiting_samples_are_direct_assignments_only():
    graph = CategoryGraph()
    spiral = graph.create_category("spiral")
    barred = graph.create_category("barred", parent_id=spiral)
    graph.assign(1, spiral)
    graph.assign(2, barred)
    assert graph.waiting(spiral) == [1]
    assert graph.samples(barred) == [2]


def test_subtree_count_includes_descendants_and_records_assignment_order():
    graph = CategoryGraph()
    spiral = graph.create_category("spiral")
    barred = graph.create_category("barred", parent_id=spiral)
    graph.assign(1, barred)
    graph.assign(2, barred)
    assert graph.samples(spiral) == []
    assert graph.subtree_samples(spiral) == [1, 2]
    graph.assign(1, spiral)
    assert graph.assignment_order[1] > graph.assignment_order[2]
    loaded = CategoryGraph.from_dict(graph.to_dict())
    assert loaded.subtree_samples(spiral) == [1, 2]
    assert loaded.assignment_order[1] > loaded.assignment_order[2]


def test_sibling_strength_defaults_to_one_and_stores_override():
    graph = CategoryGraph()
    a = graph.create_category("a")
    b = graph.create_category("b")
    assert graph.relation_strength(a, b) == 1.0
    graph.set_strength(a, b, 1.5)
    assert graph.relation_strength(b, a) == 1.5


def test_strength_rejects_out_of_range_and_cross_level_edges():
    graph = CategoryGraph()
    parent = graph.create_category("parent")
    child = graph.create_category("child", parent_id=parent)
    other = graph.create_category("other")
    with pytest.raises(ValueError):
        graph.set_strength(parent, other, 2.5)
    with pytest.raises(ValueError):
        graph.set_strength(parent, other, -0.1)
    with pytest.raises(ValueError):
        graph.set_strength(child, other, 1.0)


def test_divergence_weight_uses_sibling_edge_at_the_split():
    graph = CategoryGraph()
    spiral = graph.create_category("spiral")
    smooth = graph.create_category("smooth")
    barred = graph.create_category("barred", parent_id=spiral)
    graph.set_strength(spiral, smooth, 0.25)
    assert graph.divergence_weight(spiral, smooth) == 0.25
    assert graph.divergence_weight(barred, smooth) == 0.25
    assert graph.divergence_weight(spiral, barred) is None
    assert graph.divergence_weight(barred, barred) is None


def test_zero_strength_is_stored():
    graph = CategoryGraph()
    a = graph.create_category("a")
    b = graph.create_category("b")
    graph.set_strength(a, b, 0.0)
    assert graph.relation_strength(a, b) == 0.0


def test_blank_name_is_rejected_and_rename_updates():
    graph = CategoryGraph()
    with pytest.raises(ValueError):
        graph.create_category("  ")
    cat = graph.create_category("old")
    graph.rename(cat, "new")
    assert graph.categories[cat].name == "new"


def test_sibling_names_are_unique_and_paths_use_bars():
    graph = CategoryGraph()
    parent = graph.create_category("barred")
    child = graph.create_category("barred", parent_id=parent)
    assert graph.name_path(child) == "barred|barred"
    with pytest.raises(ValueError):
        graph.create_category("barred", parent_id=parent)
    plain = graph.create_category("plain", parent_id=parent)
    with pytest.raises(ValueError):
        graph.rename(child, "plain")
    other = graph.create_category("other")
    twin = graph.create_category("barred", parent_id=other)
    assert graph.name_path(twin) == "other|barred"
    with pytest.raises(ValueError):
        graph.create_category("a|b")
    with pytest.raises(ValueError):
        graph.rename(other, "left|right")
    loaded = CategoryGraph.from_dict(graph.to_dict())
    assert loaded.name_path(child) == "barred|barred"
