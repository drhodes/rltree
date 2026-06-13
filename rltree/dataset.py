import random
import xml.etree.ElementTree as ET

VOCAB = ["svg", "g", "rect", "circle", "path"]
TAG_TO_ID = {tag: i + 1 for i, tag in enumerate(VOCAB)}
TAG_TO_ID["NONE"] = 0
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}

MAX_DEPTH = 3
MAX_NODES = 6


def generate_random_svg(seed: int | None = None) -> str:
    """
    Generates a single random, valid SVG string matching:
    - Maximum depth of 3
    - Maximum node count of 6
    - Restricted vocabulary: ['svg', 'g', 'rect', 'circle', 'path']
    """
    if seed is not None:
        random.seed(seed)

    root = ET.Element("svg")
    node_count = 1

    # Helper to recursively add child elements
    def add_children(parent: ET.Element, current_depth: int) -> None:
        nonlocal node_count
        if current_depth >= MAX_DEPTH or node_count >= MAX_NODES:
            return

        # Maximum children we can add depends on the remaining node budget
        max_children = min(3, MAX_NODES - node_count)
        if max_children <= 0:
            return

        num_children = random.randint(0, max_children)
        for _ in range(num_children):
            if node_count >= MAX_NODES:
                break
            # Children can only be 'g', 'rect', 'circle', 'path' (no nested svg)
            child_tag = random.choice(["g", "rect", "circle", "path"])
            child = ET.SubElement(parent, child_tag)
            node_count += 1

            # Recurse for nested group tags
            if child_tag == "g":
                add_children(child, current_depth + 1)

    add_children(root, 1)
    return str(ET.tostring(root, encoding="utf-8").decode("utf-8"))


def generate_datasets(
    train_size: int = 100, val_size: int = 20, seed: int = 42
) -> tuple[list[str], list[str]]:
    """
    Generates training and validation datasets.
    """
    random.seed(seed)

    # Generate unique SVGs to ensure diversity
    svgs: set[str] = set()

    attempts = 0
    max_attempts = (train_size + val_size) * 10

    while len(svgs) < (train_size + val_size) and attempts < max_attempts:
        svgs.add(generate_random_svg())
        attempts += 1

    # If we failed to get enough unique ones, generate non-unique ones to fill
    svg_list = list(svgs)
    while len(svg_list) < (train_size + val_size):
        svg_list.append(generate_random_svg())

    train_svgs = svg_list[:train_size]
    val_svgs = svg_list[train_size : train_size + val_size]

    return train_svgs, val_svgs
