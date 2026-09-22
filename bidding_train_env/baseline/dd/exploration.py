"""Local action proposals scored by an offline Monte Carlo Q ensemble.

Scores are estimates of behavior-policy return-to-go, not simulator outcomes.
No perturbed transition is used as a Bellman target or as a real state trajectory.
"""
import torch
from torch import nn
import torch.nn.functional as F


class QEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, num_categories, hidden=128, members=3):
        super().__init__()
        self.num_categories = num_categories
        self.embedding = nn.Embedding(num_categories + 1, 16)
        self.models = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim + action_dim + 17, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            for _ in range(members)
        ])

    def forward(self, states, actions, categories, cpa):
        context = self.embedding(categories)
        x = torch.cat([states, actions, context, cpa[:, None]], dim=-1)
        return torch.cat([model(x) for model in self.models], dim=-1)

    def loss(self, batch):
        mask = batch['masks']
        categories = batch['category_ids'][:, None].expand_as(mask)[mask]
        cpa = batch['returns'][:, 1, None].expand_as(mask)[mask]
        values = self(batch['states'][mask], batch['actions'][mask], categories, cpa)
        target = batch['rtg'][mask, None].expand_as(values)
        # Independent bootstrap masks give members different observed training examples.
        bootstrap = (torch.rand_like(values) < 0.8).float()
        return (F.smooth_l1_loss(values, target, reduction='none') * bootstrap).sum() / bootstrap.sum().clamp_min(1)


@torch.no_grad()
def propose_actions(q, states, actions, categories, prompts, std=0.1,
                    probability=0.5, action_min=0., action_max=10.,
                    uncertainty_max=0.1, improvement_min=0.01, max_return_delta=0.1):
    """Return local candidate action targets and bounded *estimated* total-return labels.

    std is in raw bid-multiplier units. CPA constraint is retained; compliance is marked unknown for accepted proposals.
    Only the return component receives a Q-based adjustment. This is an inverse-model surrogate.
    """
    candidate = (actions + std * torch.randn_like(actions)).clamp(action_min, action_max)
    original_q = q(states, actions, categories, prompts[:, 1])
    candidate_q = q(states, candidate, categories, prompts[:, 1])
    # Require conservative improvement and small ensemble disagreement on both actions.
    delta = candidate_q.min(-1).values - original_q.max(-1).values
    disagreement = torch.maximum(candidate_q.std(-1, unbiased=False), original_q.std(-1, unbiased=False))
    accepted = ((torch.rand_like(delta) < probability) & (delta > improvement_min) &
                (disagreement <= uncertainty_max) & torch.isfinite(delta))
    targets = torch.where(accepted[:, None], candidate, actions)
    labels = prompts.clone()
    labels[:, 0] += torch.where(accepted, delta.clamp(max=max_return_delta), 0.)
    labels[accepted, 2] = -1.  # Q-return estimates cannot certify counterfactual CPA compliance.
    return targets, labels, accepted
