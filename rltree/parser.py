import xml.etree.ElementTree as ET

from rltree.dataset import TAG_TO_ID

# Action space size: 5 add_child actions (for each vocab tag) + 1 close_node action
ACTION_ADD_CHILD_BASE = 0  # 0 to 4
ACTION_CLOSE_NODE = 5


def parse_svg_string(svg_str: str) -> ET.Element:
    """
    Parses an SVG string using xml.etree.ElementTree.
    """
    return ET.fromstring(svg_str)


def get_trajectory_for_node(
    node: ET.Element,
    parent_tag_id: int = 0,
    depth: int = 1,
    sib_idx: int = 0,
    tot_sibs: int = 1,
) -> list[tuple[tuple[int, int, int, int, int], int]]:
    """
    Recursively extracts the DFS state-action trajectory for an Element tree.
    Each element returns a list of tuples: (state, action) where:
    - State is (depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs)
    - Action is tag_id - 1 for adding a child, or ACTION_CLOSE_NODE (5) for closing.
    """
    trajectory = []
    current_tag = node.tag
    if current_tag not in TAG_TO_ID:
        # Gracefully handle unexpected tag by mapping to NONE or raising
        raise ValueError(f"Unexpected tag found: {current_tag}")

    current_tag_id = TAG_TO_ID[current_tag]
    state = (depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs)

    children = list(node)
    num_children = len(children)

    for i, child in enumerate(children):
        child_tag = child.tag
        if child_tag not in TAG_TO_ID:
            raise ValueError(f"Unexpected tag found in child: {child_tag}")
        child_tag_id = TAG_TO_ID[child_tag]

        # Action to add this child: tag_id - 1
        action = child_tag_id - 1
        trajectory.append((state, action))

        # Recurse
        child_traj = get_trajectory_for_node(
            child,
            parent_tag_id=current_tag_id,
            depth=depth + 1,
            sib_idx=i,
            tot_sibs=num_children,
        )
        trajectory.extend(child_traj)

    # Close the current node after all children are handled
    trajectory.append((state, ACTION_CLOSE_NODE))
    return trajectory


def extract_trajectory_from_svg(
    svg_str: str,
) -> list[tuple[tuple[int, int, int, int, int], int]]:
    """
    Parses SVG string and extracts the state-action trajectory.
    """
    root = parse_svg_string(svg_str)
    return get_trajectory_for_node(root)
