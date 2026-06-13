import xml.etree.ElementTree as ET

from rltree.dataset import ID_TO_TAG, MAX_DEPTH, MAX_NODES, TAG_TO_ID


class TreeNode:
    def __init__(
        self, tag: str, parent: "TreeNode | None" = None, depth: int = 1
    ) -> None:
        self.tag = tag
        self.parent = parent
        self.depth = depth
        self.children: list[TreeNode] = []


class SvgBuildingEnv:
    """
    Stateful step-by-step tree building environment.
    Exposes reset() and step(action).
    """

    def __init__(
        self, training_svgs: list[str] | None = None, max_steps: int = 15
    ) -> None:

        self.max_steps = max_steps
        self.schema_pairs: set[tuple[str, str]] = set()
        if training_svgs:
            self.schema_pairs = self._extract_schema_pairs(training_svgs)

        self.root: TreeNode | None = None
        self.active_node: TreeNode | None = None
        self.node_count = 0
        self.step_count = 0

    def _extract_schema_pairs(self, svgs: list[str]) -> set[tuple[str, str]]:
        pairs = set()
        for s in svgs:
            try:
                root = ET.fromstring(s)

                def traverse(node: ET.Element) -> None:
                    for child in node:
                        pairs.add((node.tag, child.tag))
                        traverse(child)

                traverse(root)
            except Exception:
                pass
        return pairs

    def get_observation(self) -> tuple[int, int, int, int, int]:
        """
        Returns state context: (depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs)
        """
        if self.active_node is None:
            return (0, 0, 0, 0, 0)

        depth = self.active_node.depth
        parent_tag_id = 0
        sib_idx = 0
        tot_sibs = 1

        if self.active_node.parent:
            parent_tag_id = TAG_TO_ID[self.active_node.parent.tag]
            parent_children = self.active_node.parent.children
            sib_idx = parent_children.index(self.active_node)
            tot_sibs = len(parent_children)

        current_tag_id = TAG_TO_ID[self.active_node.tag]

        return (depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs)

    def reset(self) -> tuple[int, int, int, int, int]:
        """
        Resets environment and starts with a single root <svg> element.
        """
        self.root = TreeNode("svg", depth=1)
        self.active_node = self.root
        self.node_count = 1
        self.step_count = 0
        return self.get_observation()

    def step(
        self, action: int
    ) -> tuple[tuple[int, int, int, int, int], float, bool, dict[str, bool]]:
        """
        Executes action:
        - 0..4: Add child with corresponding tag ID (action + 1)
        - 5: Close active node and return to parent (terminates if root)

        Returns: (observation, reward, done, info)
        """
        self.step_count += 1
        done = self.step_count >= self.max_steps
        info = {"valid": False}

        if self.active_node is None:
            return (0, 0, 0, 0, 0), 0.0, True, info

        # Action: CLOSE
        if action == 5:
            if self.active_node == self.root:
                # Penalise trivially empty trees - the model must add at least one child
                if not self.root.children:
                    reward = -3.0
                else:
                    # Scale completion bonus by number of nodes built
                    reward = 5.0 + 0.5 * (self.node_count - 1)
                done = True
                info["valid"] = bool(self.root.children)
                self.active_node = None  # Tree closed
                return (0, 0, 0, 0, 0), reward, done, info
            else:
                # Return to parent node
                self.active_node = self.active_node.parent
                reward = 0.1
                info["valid"] = True
                return self.get_observation(), reward, done, info

        # Action: ADD CHILD (0..4)
        child_tag = ID_TO_TAG[action + 1]

        # Validation checks
        is_invalid = (
            child_tag == "svg"  # Only one root svg
            or self.node_count >= MAX_NODES  # Exceeds max nodes limit
            or self.active_node.depth >= MAX_DEPTH  # Exceeds max depth limit
        )

        if is_invalid:
            reward = -1.0
            return self.get_observation(), reward, done, info

        # Add child node
        child = TreeNode(
            child_tag, parent=self.active_node, depth=self.active_node.depth + 1
        )
        self.active_node.children.append(child)
        self.node_count += 1
        self.active_node = child

        # Calculate rewards
        reward = 0.1  # Base valid action reward
        parent_tag = child.parent.tag if child.parent else "NONE"
        if (parent_tag, child_tag) in self.schema_pairs:
            reward += 1.0  # Schema bonus

        info["valid"] = True
        return self.get_observation(), reward, done, info

    def to_xml_string(self) -> str:
        """
        Serializes current tree to XML string.
        """
        if self.root is None:
            return ""

        def build_element(node: TreeNode) -> ET.Element:
            el = ET.Element(node.tag)
            for child in node.children:
                el.append(build_element(child))
            return el

        et_root = build_element(self.root)
        return str(ET.tostring(et_root, encoding="utf-8").decode("utf-8"))
