import time
import xml.etree.ElementTree as ET

import torch
from torch import nn, optim

from rltree.dataset import generate_datasets
from rltree.env import SvgBuildingEnv
from rltree.model import TreePolicyMlp
from rltree.parser import extract_trajectory_from_svg


def prepare_bc_data(
    svg_list: list[str],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Extracts and packs DFS state-action transitions from SVGs into PyTorch tensors.
    """
    states_depth = []
    states_parent = []
    states_current = []
    states_sib_idx = []
    states_tot_sibs = []
    actions = []

    for svg in svg_list:
        try:
            trajectory = extract_trajectory_from_svg(svg)
            for state, action in trajectory:
                depth, parent, current, sib_idx, tot_sibs = state
                states_depth.append([depth])
                states_parent.append(parent)
                states_current.append(current)
                states_sib_idx.append([sib_idx])
                states_tot_sibs.append([tot_sibs])
                actions.append(action)
        except Exception:
            pass

    return (
        torch.tensor(states_depth, dtype=torch.long),
        torch.tensor(states_parent, dtype=torch.long),
        torch.tensor(states_current, dtype=torch.long),
        torch.tensor(states_sib_idx, dtype=torch.long),
        torch.tensor(states_tot_sibs, dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
    )


def train_behavioral_cloning(
    model: TreePolicyMlp,
    train_svgs: list[str],
    val_svgs: list[str],
    epochs: int = 20,
    lr: float = 1e-3,
) -> TreePolicyMlp:
    """
    Trains the policy network using supervised learning on ground truth DFS trajectories.
    """
    print("Preparing Behavioral Cloning (BC) dataset...")
    d_tr, p_tr, c_tr, s_tr, t_tr, a_tr = prepare_bc_data(train_svgs)
    d_val, p_val, c_val, s_val, t_val, a_val = prepare_bc_data(val_svgs)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Compute inverse-frequency class weights so the dominant "close_node"
    # action (which appears once per node) does not overwhelm the gradient.
    n_classes = 6
    counts = torch.bincount(a_tr, minlength=n_classes).float()
    counts = counts.clamp(min=1)  # avoid div-by-zero for unseen classes
    class_weights = counts.sum() / (n_classes * counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(
        f"Starting BC training. Train samples: {len(a_tr)}, Val samples: {len(a_val)}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(d_tr, p_tr, c_tr, s_tr, t_tr)
        loss = criterion(logits, a_tr)
        loss.backward()
        optimizer.step()

        # Validation accuracy check
        model.eval()
        with torch.no_grad():
            val_logits = model(d_val, p_val, c_val, s_val, t_val)
            val_preds = torch.argmax(val_logits, dim=-1)
            val_acc = (val_preds == a_val).float().mean().item()

        print(
            f"BC Epoch {epoch:02d}/{epochs} | Loss: {loss.item():.4f} | Val Accuracy: {val_acc * 100:.2f}%"
        )

        # Stop early if validation accuracy is > 95%
        if val_acc >= 0.95:
            print("Validation accuracy exceeded 95%. Stopping early.")
            break

    return model


def fine_tune_reinforce(
    model: TreePolicyMlp,
    env: SvgBuildingEnv,
    episodes: int = 100,
    lr: float = 1e-4,
    gamma: float = 0.99,
) -> TreePolicyMlp:
    """
    Fine-tunes the policy using the REINFORCE policy gradient algorithm.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"Starting REINFORCE fine-tuning over {episodes} episodes...")

    for episode in range(1, episodes + 1):
        # Roll out one trajectory
        s = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            d_t = torch.tensor([[s[0]]], dtype=torch.long)
            p_t = torch.tensor([s[1]], dtype=torch.long)
            c_t = torch.tensor([s[2]], dtype=torch.long)
            s_t = torch.tensor([[s[3]]], dtype=torch.long)
            t_t = torch.tensor([[s[4]]], dtype=torch.long)

            logits = model(d_t, p_t, c_t, s_t, t_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            action_tensor = dist.sample()  # type: ignore[no-untyped-call]
            action = action_tensor.item()

            log_prob = dist.log_prob(action_tensor)  # type: ignore[no-untyped-call]
            s_next, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            s = s_next

        # Compute discounted returns
        discounted_returns: list[float] = []
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            discounted_returns.insert(0, g)

        discounted_returns_tensor = torch.tensor(discounted_returns, dtype=torch.float)
        # Standardize returns to stabilize policy gradient updates
        if len(discounted_returns_tensor) > 1 and discounted_returns_tensor.std() > 0:
            discounted_returns_tensor = (
                discounted_returns_tensor - discounted_returns_tensor.mean()
            ) / (discounted_returns_tensor.std() + 1e-8)

        # Compute policy gradient loss
        policy_loss = []
        for lp, return_val in zip(log_probs, discounted_returns_tensor):
            policy_loss.append(-lp * return_val)

        optimizer.zero_grad()
        if policy_loss:
            loss = torch.cat(policy_loss).sum()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        if episode % 10 == 0:
            print(
                f"REINFORCE Episode {episode:03d} | Trajectory Length: {len(rewards)} | Total Reward: {sum(rewards):.2f}"
            )

    return model


def generate_inference_svg(model: TreePolicyMlp, env: SvgBuildingEnv) -> str:
    """
    Samples action greedily from the policy network to produce a valid SVG string.
    """
    model.eval()
    s = env.reset()
    done = False
    steps = 0

    while not done and steps < 15:
        d_t = torch.tensor([[s[0]]], dtype=torch.long)
        p_t = torch.tensor([s[1]], dtype=torch.long)
        c_t = torch.tensor([s[2]], dtype=torch.long)
        s_t = torch.tensor([[s[3]]], dtype=torch.long)
        t_t = torch.tensor([[s[4]]], dtype=torch.long)

        with torch.no_grad():
            probs = model.get_action_probabilities(d_t, p_t, c_t, s_t, t_t)
            # Greedy action selection
            action = int(torch.argmax(probs, dim=-1).item())

        s_next, _, done, _ = env.step(action)
        s = s_next
        steps += 1

    return env.to_xml_string()


def run_pipeline() -> str:
    """
    Runs the entire pipeline end-to-end and returns the generated SVG string.
    """
    start_time = time.time()

    # 1. Dataset Generation
    train_svgs, val_svgs = generate_datasets(train_size=100, val_size=20, seed=42)
    print(
        f"Dataset generated. Train size: {len(train_svgs)}, Val size: {len(val_svgs)}"
    )

    # 2. Model Initialization
    model = TreePolicyMlp()

    # 3. Behavioral Cloning (50 epochs per spec: BcExecutionLimit)
    model = train_behavioral_cloning(model, train_svgs, val_svgs, epochs=50, lr=1e-3)

    # 4. REINFORCE Fine-Tuning (300 episodes per spec: BcExecutionLimit)
    env = SvgBuildingEnv(training_svgs=train_svgs)
    model = fine_tune_reinforce(model, env, episodes=300, lr=1e-4)

    # 5. Inference / Output Validation
    generated_svg = generate_inference_svg(model, env)

    # Try parsing the output to ensure it is valid
    try:
        ET.fromstring(generated_svg)
        print("Inference SVG parsed successfully without errors!")
    except Exception as e:
        print(f"Error parsing inference SVG: {e}")

    print("\n--- Generated SVG ---")
    print(generated_svg)
    print("---------------------\n")

    elapsed = time.time() - start_time
    print(f"Pipeline executed in {elapsed:.2f} seconds.")

    return generated_svg


if __name__ == "__main__":
    run_pipeline()
