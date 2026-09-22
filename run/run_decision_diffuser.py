"""Train a category/CPA-conditioned state diffuser, optionally with local Q proposals."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bidding_train_env.baseline.dd.DFUSER import DFUSER
from bidding_train_env.baseline.dd.dataset import aigb_dataset
from bidding_train_env.baseline.dd.exploration import QEnsemble


def run_decision_diffuser(save_path="saved_model/DDtest", train_epoch=1, batch_size=64,
                         train_data_path="data/trajectory/trajectory_data.csv", seed=200,
                         category_dropout=0.15, condition_guidance_w=1.2, category_guidance_w=1.,
                         exploration=False, q_epochs=10, perturb_std=0.1, perturb_probability=0.5,
                         uncertainty_max=0.1, improvement_min=0.01, max_return_delta=0.1,
                         action_max=None, device=None, model_dim=128, n_timesteps=10):
    if train_epoch < 1 or batch_size < 1 or (exploration and q_epochs < 1):
        raise ValueError("Training epochs and batch size must be positive")
    if not 0 <= perturb_probability <= 1 or min(perturb_std, uncertainty_max, improvement_min, max_return_delta) < 0:
        raise ValueError("Invalid exploration configuration")
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    dataset = aigb_dataset(train_data_path=train_data_path)
    observed_max = max(float(e['actions'].max()) for e in dataset.episodes)
    action_max = max(1., observed_max * 1.1) if action_max is None else action_max
    if action_max < observed_max:
        raise ValueError("action_max is below an expert action; fix units or raise the bound")
    algorithm = DFUSER(dim_obs=dataset.num_of_states, metadata=dataset.metadata,
                       num_categories=len(dataset.metadata['categories']), category_dropout=category_dropout,
                       condition_guidance_w=condition_guidance_w, category_guidance_w=category_guidance_w,
                       ACTION_MAX=action_max, network_random_seed=seed, model_dim=model_dim,
                       n_timesteps=n_timesteps).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    q = None
    q_losses = []
    if exploration:
        q = QEnsemble(dataset.num_of_states, 1, len(dataset.metadata['categories'])).to(device)
        optimizer = torch.optim.Adam(q.parameters(), lr=1e-3)
        for epoch in range(q_epochs):
            losses = []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                loss = q.loss(batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            q_losses.append(sum(losses) / len(losses))
            print(f"Q epoch {epoch + 1}: loss={q_losses[-1]:.6f}")
        q.eval().requires_grad_(False)
    proposal_config = dict(std=perturb_std, probability=perturb_probability,
                           action_min=0., action_max=action_max, uncertainty_max=uncertainty_max,
                           improvement_min=improvement_min, max_return_delta=max_return_delta)
    step = 0
    accepted_total = 0
    action_targets_total = 0
    for epoch in range(train_epoch):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, (diffusion, inverse) = algorithm.trainStep(
                batch['states'], batch['actions'], batch['returns'], batch['masks'],
                batch['category_ids'], q=q, exploration=proposal_config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            step += 1
            action_targets_total += int(batch['masks'].sum())
            accepted_total += int(algorithm.last_metrics.get('accepted_proposals', 0))
            print(f"epoch={epoch + 1} step={step} loss={loss.item():.6f} "
                  f"diffusion={diffusion.item():.6f} inverse={inverse.item():.6f} "
                  f"accepted={algorithm.last_metrics.get('accepted_proposals', 0)}")
    algorithm.save_net(save_path, step)
    if q is not None:
        torch.save(dict(state_dict=q.state_dict(), metadata=dataset.metadata,
                        state_dim=dataset.num_of_states, action_dim=1,
                        num_categories=len(dataset.metadata['categories'])), Path(save_path) / 'q_ensemble.pt')
    report = dict(seed=seed, train_data_path=str(train_data_path), episodes=len(dataset),
                  train_epoch=train_epoch, batch_size=batch_size, device=str(device),
                  torch_version=torch.__version__, exploration=exploration, proposal_config=proposal_config,
                  q_epochs=q_epochs if exploration else 0, q_losses=q_losses, accepted_total=accepted_total,
                  action_targets_total=action_targets_total,
                  acceptance_rate=accepted_total / max(action_targets_total, 1),
                  model_config=algorithm.config, normalization=dataset.metadata)
    (Path(save_path) / 'training_config.json').write_text(json.dumps(report, indent=2))
    return algorithm
