# System Design Document: rltree Prototype
**Architecture for Reinforcement Learning-Driven Generation of Hierarchical Data**

---

## 1. Executive Summary

This document outlines the design and architectural philosophy for `rltree`, a prototype system aimed at generating strictly hierarchical structured data (initially focusing on Scalable Vector Graphics, or SVGs) using Reinforcement Learning (RL). 

Unlike standard Large Language Models (LLMs) that treat data generation as a one-dimensional sequence prediction problem, `rltree` embraces the inherent topology of the target data. By explicitly modeling generation as a tree-building process, we eliminate entire classes of syntax errors and reframe the machine learning task around structural localized decisions, informed by principles of spectral graph theory and depth-aware variable scopes.

This document serves as the foundational blueprint for the initial prototype, detailing the mathematical intuition, the state representations, and the ingestion pipeline required to train a baseline model on a modest dataset of SVGs.

---

## 2. Problem Statement: The Limits of Sequential Generation

Modern text generation is dominated by autoregressive sequence models. When tasked with generating an SVG or an Abstract Syntax Tree (AST), an LLM flattens the hierarchy into a 1D string of tokens: `[<, s, v, g, >, \n, <, r, e, c, t, ...]`.

This approach suffers from fundamental architectural weaknesses when applied to strictly structured data:

1. **Implicit Structural Burden:** The model must use its finite context window to "remember" its current depth, open tags, and parent relationships purely through statistical pattern matching of the preceding text.
2. **Syntactic Fragility:** Because the action space is character or sub-word token emission, the model is capable of making illegal syntactic moves (e.g., closing a tag that was never opened, or emitting malformed XML).
3. **Inefficient Reward Attribution:** In an RL setup, if a generated SVG fails to render, attributing that failure back to a specific token generation step in a long sequence is notoriously difficult (the credit assignment problem).

`rltree` abandons the sequence. If the target is a tree, the generation process should explicitly build a tree.

---

## 3. Core Concept: Trees, Topology, and Localized Decisions

The foundational premise of `rltree` is that the agent operates on a graph $G = (V, E)$, strictly constrained to be a rooted tree. 

### 3.1 The Action Space
The agent does not emit strings. Its action space consists of explicit tree operations:
*   `add_child(node_type, attributes)`: Creates a new child node under the currently active node and moves the active pointer to this new child.
*   `close_node()`: Finalizes the current node and moves the active pointer back up to the parent.

This constraint provides absolute syntactic safety. It is mathematically impossible for the agent to generate malformed XML structure. 

### 3.2 The State Space: Moving Beyond Flat Context
If the agent is not looking at a flattened string of previous tokens, what is its observation state? We require a representation that captures the topology of the partially built tree. 

---

## 4. Mathematical Foundations & Algebraic Intuition

To understand how to represent the state, we must look at how we measure the shape of a tree algebraically. 

### 4.1 The Adjacency Matrix and Spectral Radius
For any unweighted, undirected tree of $N$ nodes, we can construct an $N \times N$ adjacency matrix $A$. By calculating the eigenvalues of $A$, we extract its spectrum. The largest absolute eigenvalue is the **spectral radius** ($\rho$).

As discussed during the conceptual design phase, $\rho$ acts as a continuous, real-valued measure of the tree's **"splay"** or **"fanout"**. 
*   A maximum-depth "Line" tree has $\rho < 2$.
*   A maximum-breadth "Star" tree has $\rho = \sqrt{N-1}$.

The spectral radius captures the tendency of the graph to form dense hubs versus long paths.

### 4.2 The Limitation of Pure Matrices
While the adjacency matrix beautifully captures "fanout," it represents an unrooted graph. It is invariant to depth. A tree that is a long stem ending in a massive star has the exact same eigenvalues as a massive star ending in a long stem. 

For generation tasks, *where* the fanout occurs is critical. The structural rules at the root of an SVG (setting up `<defs>`) are entirely different from the structural rules at depth 5 (generating hundreds of `<path>` data points).

### 4.3 Why Not Tensors?
To encode depth and directionality algebraically, one could upgrade the 2D Adjacency Matrix to a 3D Adjacency Tensor `(Nodes, Nodes, Depth)`. However, tensor operations introduce massive computational overhead. For a small-scale prototype on limited hardware, maintaining and decomposing a growing 3D tensor at every generation step is prohibitively expensive and creates an overly sparse, massive state space that slows early learning.

---

## 5. Architectural Paradigm: Depth-Scoped Variable Scopes

To capture the insights of the algebraic analysis (fanout) without the computational penalty of tensors, `rltree` adopts a localized, scope-based architecture reminiscent of Hierarchical Reinforcement Learning (HRL).

Instead of forcing a single monolithic agent to understand the entire global topology of the tree at once, we provide the agent with highly specific, depth-aware local context.

### 5.1 The Observation State Tuple
At any given step, the agent's observation state is not the full tree, but a focused tuple describing its immediate structural environment:

`State = (Depth, Parent_Tag, Current_Tag, Sibling_Index, Total_Siblings)`

*   **Depth:** An integer representing the distance from the root. This acts as the "variable scope," allowing the model to learn different policies for shallow setup vs deep detail generation.
*   **Parent_Tag / Current_Tag:** The immediate structural lineage.
*   **Sibling_Index & Total_Siblings:** These two integers serve as a lightweight, localized proxy for the "fanout" metric discussed earlier. It tells the agent how wide the current branch is growing without needing to calculate the spectral radius of the entire graph.

### 5.2 Scoped Policies and Batching
By explicitly including `Depth` in the state, the model naturally partitions its learned behavior. During training, experiences can be batched by depth. The network effectively learns an implicit `RootPolicy` (Depth 0-1) characterized by low fanout, and a `DetailPolicy` (Depth 3+) characterized by high fanout, all within a unified architecture.

---

## 6. Prototype Implementation Plan

The immediate goal is to prove feasibility by ingesting a small dataset of SVGs and training a baseline policy.

### 6.1 Data Ingestion Pipeline (The "Parser")
Before we can train, we must translate raw SVGs into our localized tuple format.

1.  **Parse XML:** Utilize Python's `xml.etree.ElementTree` to parse the SVG into a standard memory tree.
2.  **Traverse & Extract:** Perform a Depth-First Search (DFS) traversal of the tree.
3.  **Generate Ground Truth Pairs:** For every visit and departure from a node, record the exact action taken and the state context at that moment.

**Example Extraction:**
Given `<svg><g><circle/></g></svg>`, the extracted sequence of `(State) -> Action` pairs is:
1. `(Depth: 0, Parent: None, Tag: None, Sib_Idx: 0, Tot_Sib: 0)` -> `add_child("svg")`
2. `(Depth: 1, Parent: svg, Tag: None, Sib_Idx: 0, Tot_Sib: 1)` -> `add_child("g")`
3. `(Depth: 2, Parent: g, Tag: None, Sib_Idx: 0, Tot_Sib: 1)` -> `add_child("circle")`
4. `(Depth: 3, Parent: circle, Tag: None, Sib_Idx: 0, Tot_Sib: 0)` -> `close_node()`
5. `(Depth: 2, Parent: g, Tag: circle, Sib_Idx: 0, Tot_Sib: 1)` -> `close_node()`
6. `(Depth: 1, Parent: svg, Tag: g, Sib_Idx: 0, Tot_Sib: 1)` -> `close_node()`

This ingestion pipeline creates the exact behavioral cloning dataset needed to bootstrap the baseline model.

### 6.2 Baseline Model Architecture
The initial model will be a lightweight Multi-Layer Perceptron (MLP) or a small Recurrent Neural Network (RNN/GRU) acting on the localized state.

*   **Inputs:** The state tuple `(Depth, Parent, Tag, Sibling_Index, Total_Siblings)`. Text tags will be mapped to a static embedding dictionary.
*   **Outputs:** A probability distribution over the discrete action space (`add_child(tag_1)`, `add_child(tag_2)`, ... `close_node()`).

### 6.3 The Reward System (Dense vs Sparse)
During the RL phase (after initial behavioral cloning), the agent must be guided by rewards. As established in the Socratic discussion, pure episodic (sparse) rewards for structural tasks often fail because the agent wanders blindly.

We will employ a **Dense (Per-Step) Reward**:
*   $+0.1$ for a structurally valid move (e.g., not closing the root prematurely).
*   $+1.0$ (Schema Bonus) if the parent-child pair exists in the training set (e.g., `<circle>` inside `<g>`).
*   $-1.0$ (Operation Cost) penalty to discourage infinite looping or endless horizontal fanout without closure.

A sparse reward (e.g., comparing the final rendered SVG to a target using Image IoU) will be added as a final modifier once the agent reliably generates valid trees.

---

## 7. Future Expansions & Theoretical Horizons

While the prototype relies on explicit localized tuples for efficiency, the architecture is designed to accommodate advanced techniques once feasibility is proven.

### 7.1 Retrieval-Augmented Generation (RAG) with Spectral Embeddings
Once the state space includes richer node attributes, we can revisit the algebraic properties. We can calculate the Graph Laplacian for all SVGs in a large dataset, extract their principal eigenvectors, and store these "structural signatures" in a Vector Database. During generation, the agent could query the database using the spectral signature of its partial tree to retrieve known-good sub-tree completions.

### 7.2 Generalization to Source Code (ASTs)
The `rltree` architecture is entirely domain-agnostic. By replacing the SVG XML parser with a source code parser (like Python's `ast` module), the exact same tuple-extraction and depth-scoped policies can be applied to Abstract Syntax Trees. This opens the door to structurally-aware code generation, automated refactoring, and bug detection based on topological anomalies rather than text matching.

---
*End of Document*