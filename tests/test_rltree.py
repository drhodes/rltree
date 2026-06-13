import xml.etree.ElementTree as ET

import torch

from rltree.dataset import generate_datasets, generate_random_svg
from rltree.env import SvgBuildingEnv
from rltree.model import TreePolicyMlp
from rltree.parser import (
    extract_trajectory_from_svg,
    parse_svg_string,
)
from rltree.train import run_pipeline


def test_dataset_generation() -> None:
    # Test random SVG generator
    svg_str = generate_random_svg(seed=123)
    assert svg_str.startswith("<svg")
    root = ET.fromstring(svg_str)
    assert root.tag == "svg"

    # Test dataset limits
    train, val = generate_datasets(train_size=5, val_size=2, seed=42)
    assert len(train) == 5
    assert len(val) == 2


def test_parser() -> None:
    # Test simple parser logic
    svg_str = "<svg><circle/><g><path/></g></svg>"
    root = parse_svg_string(svg_str)
    assert root.tag == "svg"

    # Test trajectory extraction
    trajectory = extract_trajectory_from_svg(svg_str)
    assert len(trajectory) > 0

    # Test unexpected tag error
    bad_svg = "<svg><unsupported_tag/></svg>"
    try:
        extract_trajectory_from_svg(bad_svg)
    except ValueError:
        pass


def test_model() -> None:
    model = TreePolicyMlp()

    # Create dummy input batch
    depth = torch.tensor([[1]])
    parent_tag_id = torch.tensor([0])
    current_tag_id = torch.tensor([1])
    sib_idx = torch.tensor([[0]])
    tot_sibs = torch.tensor([[1]])

    logits = model(depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs)
    probs = model.get_action_probabilities(
        depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs
    )

    assert logits.shape == (1, 6)
    assert probs.shape == (1, 6)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]))


def test_environment() -> None:
    # Test environment initialization and logic
    env = SvgBuildingEnv()
    obs = env.reset()
    assert obs == (
        1,
        0,
        1,
        0,
        1,
    )  # (depth 1, parent 0, current svg=1, sib_idx 0, tot_sibs 1)

    # Try adding a valid circle (circle=4 -> action 3)
    obs_next, reward, done, info = env.step(3)
    assert info["valid"] is True
    assert reward >= 0.1
    assert env.node_count == 2
    assert env.active_node is not None and env.active_node.tag == "circle"

    # Try adding circle again (circle tag depth is 2, active node is circle. Max depth is 3.
    # From circle (depth 2), adding path (path=5 -> action 4) is valid:
    obs_next2, reward2, done2, info2 = env.step(4)
    assert info2["valid"] is True

    # Try adding from depth 3 node (should be invalid)
    obs_next3, reward3, done3, info3 = env.step(4)
    assert info3["valid"] is False
    assert reward3 == -1.0

    # Test closing
    # Close from path (depth 3) back to circle (depth 2)
    obs_c1, r_c1, d_c1, i_c1 = env.step(5)
    assert i_c1["valid"] is True
    assert env.active_node is not None and env.active_node.tag == "circle"

    # Close from circle (depth 2) back to svg (depth 1)
    obs_c2, r_c2, d_c2, i_c2 = env.step(5)
    assert i_c2["valid"] is True
    assert env.active_node == env.root

    # Close root to terminate
    obs_root, r_root, d_root, i_root = env.step(5)
    assert d_root is True
    assert r_root == 5.0

    # Step on terminated env
    obs_term, r_term, d_term, i_term = env.step(0)
    assert d_term is True
    assert r_term == 0.0

    # Test to_xml_string
    env_empty = SvgBuildingEnv()
    assert env_empty.to_xml_string() == ""


def test_pipeline() -> None:
    # Run the full pipeline to verify execution and integration coverage
    generated_svg = run_pipeline()
    assert generated_svg.startswith("<svg")
