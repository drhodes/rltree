from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class TreePolicyMlp(nn.Module):
    """
    Lightweight MLP policy network for SVG tree generation.
    Inputs:
    - depth: tensor of shape (B, 1)
    - parent_tag_id: tensor of shape (B,)
    - current_tag_id: tensor of shape (B,)
    - sib_idx: tensor of shape (B, 1)
    - tot_sibs: tensor of shape (B, 1)

    Output:
    - action_probabilities: shape (B, 6) representing probability distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        # Tag vocab: pad/none=0, svg=1, g=2, rect=3, circle=4, path=5 (6 unique categories)
        self.parent_emb = nn.Embedding(num_embeddings=6, embedding_dim=8)
        self.current_emb = nn.Embedding(num_embeddings=6, embedding_dim=8)

        # 8 parent embedding + 8 current embedding + 3 numerical values = 19
        self.fc1 = nn.Linear(8 + 8 + 3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 6)  # 6 discrete actions (0..4 add child, 5 close)

    def forward(
        self,
        depth: torch.Tensor,
        parent_tag_id: torch.Tensor,
        current_tag_id: torch.Tensor,
        sib_idx: torch.Tensor,
        tot_sibs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw logits to allow numerically stable operations (e.g. CrossEntropyLoss or Categorical(logits=...)).
        """
        p_embed = self.parent_emb(parent_tag_id)  # (B, 8)
        c_embed = self.current_emb(current_tag_id)  # (B, 8)

        # Ensure numerical tensors are float and shaped (B, 1)
        d_val = depth.view(-1, 1).float()
        s_idx_val = sib_idx.view(-1, 1).float()
        t_sibs_val = tot_sibs.view(-1, 1).float()

        x = torch.cat([p_embed, c_embed, d_val, s_idx_val, t_sibs_val], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return cast(torch.Tensor, self.fc_out(x))

    def get_action_probabilities(
        self,
        depth: torch.Tensor,
        parent_tag_id: torch.Tensor,
        current_tag_id: torch.Tensor,
        sib_idx: torch.Tensor,
        tot_sibs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns action probability distribution using Softmax.
        """
        logits = self.forward(depth, parent_tag_id, current_tag_id, sib_idx, tot_sibs)
        return F.softmax(logits, dim=-1)
