"""Cumulative exclusive category graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass

RECYCLE_ID = "recycle"
RECYCLE_NAME = "Recycle"


@dataclass
class Category:
    id: str
    name: str
    parent_id: str | None


class CategoryGraph:
    """One image has one direct category. Sibling edges carry a repulsion strength."""

    def __init__(self) -> None:
        self.categories: dict[str, Category] = {}
        self.assignments: dict[int, str] = {}
        self.assignment_rounds: dict[int, int] = {}
        self.assignment_order: dict[int, int] = {}
        self.strengths: dict[tuple[str, str], float] = {}
        self.positions: dict[str, tuple[float, float]] = {}
        self._next = 1
        self._order = 0
        self._ensure_recycle()

    def _ensure_recycle(self) -> None:
        if RECYCLE_ID not in self.categories:
            self.categories[RECYCLE_ID] = Category(RECYCLE_ID, RECYCLE_NAME, None)

    def is_recycled_category(self, category_id: str) -> bool:
        if category_id not in self.categories:
            return False
        return RECYCLE_ID in self.path(category_id)

    def is_recycled_sample(self, sample_index: int) -> bool:
        owner = self.category_of(sample_index)
        return owner is not None and self.is_recycled_category(owner)

    def training_samples(self) -> list[int]:
        """Assigned samples that are not in the recycle bin."""
        return sorted(
            index
            for index, owner in self.assignments.items()
            if not self.is_recycled_category(owner)
        )

    def class_count(self) -> int:
        return sum(1 for category_id in self.categories if not self.is_recycled_category(category_id))

    def create_category(self, name: str, parent_id: str | None = None) -> str:
        cleaned = _clean_category_name(name)
        if parent_id is not None and parent_id not in self.categories:
            raise KeyError(f"unknown parent category: {parent_id}")
        if parent_id is not None and self.is_recycled_category(parent_id):
            raise ValueError("the recycle bin cannot contain a class")
        if self._sibling_name_taken(parent_id, cleaned):
            raise ValueError("a sibling category already has this name")
        category_id = f"c{self._next}"
        self._next += 1
        self.categories[category_id] = Category(category_id, cleaned, parent_id)
        return category_id

    def rename(self, category_id: str, name: str) -> None:
        if category_id == RECYCLE_ID:
            raise ValueError("the recycle bin cannot be renamed")
        cleaned = _clean_category_name(name)
        parent_id = self.categories[category_id].parent_id
        if self._sibling_name_taken(parent_id, cleaned, except_id=category_id):
            raise ValueError("a sibling category already has this name")
        self.categories[category_id].name = cleaned

    def assign(self, sample_index: int, category_id: str, round_index: int = 0) -> None:
        if category_id not in self.categories:
            raise KeyError(f"unknown category: {category_id}")
        sample_index = int(sample_index)
        self.assignments[sample_index] = category_id
        self.assignment_rounds[sample_index] = int(round_index)
        self._order += 1
        self.assignment_order[sample_index] = self._order

    def unassign(self, sample_index: int) -> None:
        sample_index = int(sample_index)
        if sample_index not in self.assignments:
            raise KeyError(f"sample {sample_index} is not in a category")
        del self.assignments[sample_index]
        self.assignment_rounds.pop(sample_index, None)
        self.assignment_order.pop(sample_index, None)

    def category_of(self, sample_index: int) -> str | None:
        return self.assignments.get(int(sample_index))

    def set_position(self, category_id: str, x: float, y: float) -> None:
        if category_id not in self.categories:
            raise KeyError(f"unknown category: {category_id}")
        x_value = float(x)
        y_value = float(y)
        if (
            x_value < 0.0
            or x_value > 1.0
            or y_value < 0.0
            or y_value > 1.0
        ):
            raise ValueError("position must be a fraction inside the graph")
        self.positions[category_id] = (x_value, y_value)

    def samples(self, category_id: str) -> list[int]:
        if category_id not in self.categories:
            raise KeyError(f"unknown category: {category_id}")
        return sorted(index for index, owner in self.assignments.items() if owner == category_id)

    def waiting(self, category_id: str) -> list[int]:
        """Images assigned directly to this category, with no subclass yet."""
        return self.samples(category_id)

    def descendant_ids(self, category_id: str) -> list[str]:
        if category_id not in self.categories:
            raise KeyError(f"unknown category: {category_id}")
        ordered = [category_id]
        for child in self.children(category_id):
            ordered.extend(self.descendant_ids(child.id))
        return ordered

    def subtree_samples(self, category_id: str) -> list[int]:
        """Samples assigned to this category or any descendant."""
        allowed = set(self.descendant_ids(category_id))
        return sorted(
            index for index, owner in self.assignments.items() if owner in allowed
        )

    def children(self, parent_id: str | None) -> list[Category]:
        kids = [cat for cat in self.categories.values() if cat.parent_id == parent_id]
        return sorted(kids, key=lambda cat: cat.id)

    def name_path(self, category_id: str) -> str:
        """Ancestor names joined by ``|``, including this category.

        ``|`` is reserved so a child can share its parent's name, as in
        ``barred|barred``. The same name under another parent is a different path.
        """
        if category_id not in self.categories:
            raise KeyError(f"unknown category: {category_id}")
        return "|".join(self.categories[item].name for item in self.path(category_id))

    def path(self, category_id: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        current: str | None = category_id
        while current is not None:
            if current in seen:
                raise ValueError("category parent cycle")
            seen.add(current)
            out.append(current)
            current = self.categories[current].parent_id
        out.reverse()
        return out

    def set_strength(self, category_a: str, category_b: str, strength: float) -> None:
        if not self._are_siblings(category_a, category_b):
            raise ValueError("relation strengths are only defined between sibling categories")
        value = float(strength)
        if value < 0.0 or value > 2.0:
            raise ValueError("strength must be in [0, 2]")
        self.strengths[_edge_key(category_a, category_b)] = value

    def relation_strength(self, category_a: str, category_b: str) -> float:
        if not self._are_siblings(category_a, category_b):
            raise ValueError("relation strengths are only defined between sibling categories")
        return self.strengths.get(_edge_key(category_a, category_b), 1.0)

    def divergence_weight(self, category_a: str, category_b: str) -> float | None:
        """Repulsion weight for two categories.

        None means the pair is not a repulsion pair: the categories are the same,
        one sits inside the other, or either one is the recycle bin.
        """
        if category_a == category_b:
            return None
        if self.is_recycled_category(category_a) or self.is_recycled_category(category_b):
            return None
        path_a = self.path(category_a)
        path_b = self.path(category_b)
        shared = 0
        while (
            shared < len(path_a)
            and shared < len(path_b)
            and path_a[shared] == path_b[shared]
        ):
            shared += 1
        if shared == len(path_a) or shared == len(path_b):
            return None
        return self.relation_strength(path_a[shared], path_b[shared])

    def to_dict(self) -> dict:
        categories = []
        for cat in self.children_depth_first():
            row = asdict(cat)
            row["path"] = self.name_path(cat.id)
            categories.append(row)
        return {
            "next_id": self._next,
            "categories": categories,
            "assignments": [
                {
                    "sample_index": index,
                    "category_id": category_id,
                    "round": self.assignment_rounds.get(index, 0),
                    "order": self.assignment_order.get(index, 0),
                }
                for index, category_id in sorted(self.assignments.items())
            ],
            "strengths": [
                {"a": left, "b": right, "strength": strength}
                for (left, right), strength in sorted(self.strengths.items())
            ],
            "positions": [
                {"id": category_id, "x": xy[0], "y": xy[1]}
                for category_id, xy in sorted(self.positions.items())
            ],
        }

    def children_depth_first(self) -> list[Category]:
        ordered: list[Category] = []

        def walk(parent_id: str | None) -> None:
            for cat in self.children(parent_id):
                ordered.append(cat)
                walk(cat.id)

        walk(None)
        return ordered

    @classmethod
    def from_dict(cls, payload: dict) -> "CategoryGraph":
        graph = cls()
        graph._next = int(payload.get("next_id", 1))
        for raw in payload.get("categories", []):
            graph.categories[raw["id"]] = Category(
                id=raw["id"],
                name=raw["name"],
                parent_id=raw.get("parent_id"),
            )
        for raw in payload.get("assignments", []):
            index = int(raw["sample_index"])
            graph.assignments[index] = raw["category_id"]
            graph.assignment_rounds[index] = int(raw.get("round", 0))
            order = int(raw.get("order", 0))
            graph.assignment_order[index] = order
            graph._order = max(graph._order, order)
        for raw in payload.get("strengths", []):
            graph.strengths[_edge_key(raw["a"], raw["b"])] = float(raw["strength"])
        for raw in payload.get("positions", []):
            category_id = raw["id"]
            if category_id in graph.categories:
                graph.positions[category_id] = (float(raw["x"]), float(raw["y"]))
        graph._ensure_recycle()
        return graph

    def _sibling_name_taken(
        self, parent_id: str | None, name: str, except_id: str | None = None
    ) -> bool:
        for cat in self.children(parent_id):
            if cat.id == except_id or self.is_recycled_category(cat.id):
                continue
            if cat.name == name:
                return True
        return False

    def _are_siblings(self, category_a: str, category_b: str) -> bool:
        if category_a == category_b:
            return False
        if category_a not in self.categories or category_b not in self.categories:
            raise KeyError("unknown category")
        if self.is_recycled_category(category_a) or self.is_recycled_category(category_b):
            return False
        return self.categories[category_a].parent_id == self.categories[category_b].parent_id


def _clean_category_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        raise ValueError("category name is required")
    if "|" in cleaned:
        raise ValueError("category name cannot contain |")
    return cleaned


def _edge_key(category_a: str, category_b: str) -> tuple[str, str]:
    return tuple(sorted((category_a, category_b)))
