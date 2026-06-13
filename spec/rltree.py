"""
Specifications for the rltree rapid proof-of-concept.
"""

from .err import Feat, Req


class RapidPoC(Feat):
    """
    Rapid Proof-of-Concept for rltree.

    The goal is to demonstrate a functional RL-driven tree generation pipeline
    that completes training and evaluation in under 3 minutes on local hardware.
    """


class SyntheticSvgDataset(Req):
    """
    The system must generate a tiny synthetic dataset of valid SVGs.

    To ensure rapid training, the dataset should consist of 50-100 SVGs with
    a maximum of 6 nodes and a maximum depth of 3 (e.g., <svg><g><circle/></g></svg>).
    """


class SvgTagVocabulary(SyntheticSvgDataset):
    """
    The SVG generation vocabulary must be strictly restricted to a small set of tags:
    ['svg', 'g', 'rect', 'circle', 'path'].
    """


class SvgNodeAndDepthLimits(SyntheticSvgDataset):
    """
    Generated SVG trees must have at most 6 nodes and a maximum depth of 3 to
    prevent scaling issues during rapid proof-of-concept development.
    """


class SvgDatasetSize(SyntheticSvgDataset):
    """
    The dataset generator must produce 50 to 100 training SVGs and a separate,
    held-out validation set of 10 to 20 SVGs.
    """


class DfsTupleExtraction(Req):
    """
    The system must translate parsed trees into localized state-action pairs.

    For every node visit and closure during a DFS traversal, the pipeline must
    extract the tuple: (Depth, Parent_Tag, Current_Tag, Sibling_Index, Total_Siblings)
    mapped to the corresponding tree operation (add_child or close_node).
    """


class SvgXmlParser(DfsTupleExtraction):
    """
    The pipeline must parse SVGs using standard Python `xml.etree.ElementTree`
    to guarantee well-formed input structures.
    """


class DfsTraversal(DfsTupleExtraction):
    """
    The parser must perform a depth-first search (DFS) traversal of the element tree,
    recording state contexts and target actions at each node entry and exit.
    """


class ObservationStateEncoder(DfsTupleExtraction):
    """
    The state representation must encode the current tree node context as a
    numeric tuple: (Depth, Parent_Tag_Id, Current_Tag_Id, Sibling_Index, Total_Siblings).
    """


class TargetActionEncoder(DfsTupleExtraction):
    """
    The action labels must be encoded as discrete integers corresponding to:
    - `add_child(tag_id)` for each tag in the vocabulary
    - `close_node`
    """


class MinimalPytorchMlp(Req):
    """
    The policy model must be a lightweight Multi-Layer Perceptron (MLP).

    To meet the time constraints, the model should utilize small embedding
    layers for categorical inputs and no more than two hidden layers of 64
    units each.
    """


class MlpEmbeddingLayers(MinimalPytorchMlp):
    """
    The model must use discrete embedding layers for categorical variables
    such as `Parent_Tag_Id` and `Current_Tag_Id`.
    """


class MlpHiddenLayers(MinimalPytorchMlp):
    """
    The policy network must have exactly two hidden fully connected layers
    of 64 hidden units each, utilizing ReLU activations.
    """


class MlpOutputDistribution(MinimalPytorchMlp):
    """
    The policy network output layer must produce a probability distribution
    over the action space using a Softmax activation.
    """


class BehavioralCloningBootstrap(Req):
    """
    The training process must begin with a behavioral cloning phase.

    The model must first be trained using supervised learning on the synthetic
    ground-truth pairs to quickly establish basic syntactic and structural
    competence before transitioning to reinforcement learning.
    """


class BcLossAndOptimizer(BehavioralCloningBootstrap):
    """
    Supervised learning must optimize cross-entropy loss using the Adam optimizer
    with a standard learning rate (e.g., 1e-3).

    The loss must use inverse-frequency class weighting so that the dominant
    `close_node` action — which appears once per node and skews the label
    distribution — does not overwhelm the gradient signal for rarer add-child actions.
    """


class BcTrainingLoop(BehavioralCloningBootstrap):
    """
    Supervised training must run over the extracted state-action pairs, tracking
    both training loss and validation accuracy.
    """


class BcExecutionLimit(BehavioralCloningBootstrap):
    """
    The supervised training phase must complete within 50 epochs or when
    validation accuracy exceeds 95% to allow adequate convergence.

    The REINFORCE fine-tuning phase must run for at least 300 episodes to
    provide sufficient exploration for the reward signal to shape behaviour
    beyond what the behavioral cloning phase achieves.
    """


class DenseReinforceFineTuning(Req):
    """
    The model must be fine-tuned using REINFORCE with dense rewards.

    The agent must receive immediate feedback after every action:
    - Positive reward (+1) for structurally valid moves and schema adherence.
    - Negative reward (-1) for invalid operations.
    - Termination bonus (+5) for completing a valid tree.
    """


class RlEnvironment(DenseReinforceFineTuning):
    """
    The agent must interact with a step-by-step tree-building environment that
    maintains a pointer to the active node and exposes `reset()` and `step(action)` methods.
    """


class RlRewardFunction(DenseReinforceFineTuning):
    """
    The environment's reward function must prevent mode collapse by ensuring
    the agent cannot exploit a trivial shortcut of immediately closing the root.

    Concretely, the reward function must:
    - +0.1 for a structurally valid child-add move.
    - +1.0 schema bonus if the parent-child tag pair is present in the training set.
    - -1.0 penalty for invalid child-add actions (e.g., exceeding max depth/nodes,
      or attempting to add a duplicate root).
    - -3.0 penalty for closing the root when it has no children (empty tree shortcut).
    - +5.0 + 0.5 * (node_count - 1) bonus for successfully closing the root with
      at least one child, scaling incentive with tree richness.
    """


class RlPolicyGradient(DenseReinforceFineTuning):
    """
    The policy network must be fine-tuned using the REINFORCE algorithm to optimize
    the cumulative reward over trajectories of maximum length 15.
    """


class ExecutionTimeConstraint(Req):
    """
    The entire pipeline must execute in under 3 minutes.

    This includes data generation, behavioral cloning, RL fine-tuning, and
    a sample inference run that outputs a valid SVG string.
    """


class ThreeMinuteBudget(ExecutionTimeConstraint):
    """
    A single execution run of the PoC from start to finish must complete in
    less than 180 seconds on local CPU hardware.
    """


class ValidInferenceOutput(ExecutionTimeConstraint):
    """
    At inference time, the model must output a valid SVG string that parses
    back to an XML element tree without raising exceptions.
    """
