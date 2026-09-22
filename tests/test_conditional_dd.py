import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from bidding_train_env.baseline.dd.dataset import aigb_dataset
from bidding_train_env.baseline.dd.DFUSER import DFUSER, TemporalUnet, WeightedStateL2
from bidding_train_env.baseline.dd.exploration import QEnsemble, propose_actions
from run.run_decision_diffuser import run_decision_diffuser


torch.set_num_threads(1)


def write_data(path, lengths=(4, 2), state_dim=4):
    rows = []
    for episode, length in enumerate(lengths):
        for t in range(length):
            rows.append(dict(deliveryPeriodIndex=episode, advertiserNumber=0,
                             advertiserCategoryIndex=10 + episode * 20, CPAConstraint=2,
                             realAllCost=length, realAllConversion=length,
                             timeStepIndex=t, state=str([float(t + episode)] * state_dim),
                             action=1., reward=float(episode + 1), done=int(t == length - 1)))
    pd.DataFrame(rows[::-1]).to_csv(path, index=False)
    return path


def test_episode_normalization_and_unknown_category(tmp_path):
    path = write_data(tmp_path / 'data.csv')
    dataset = aigb_dataset(4, train_data_path=path)
    assert len(dataset) == 2
    assert dataset[1]['masks'].tolist() == [True, True, False, False]
    assert torch.equal(dataset[1]['states'][2:], torch.zeros(2, 4))
    assert dataset[0]['rtg'].tolist() == [1., .75, .5, .25]
    assert dataset[0]['returns'].tolist() == [1., 1., 1.]
    metadata = dict(dataset.metadata, categories=[10])
    assert aigb_dataset(4, train_data_path=path, metadata=metadata)[1]['category_ids'] == 1


@pytest.mark.parametrize('column,value', [('timeStepIndex', 0), ('done', 1), ('realAllCost', 999)])
def test_reject_invalid_episodes(tmp_path, column, value):
    path = write_data(tmp_path / 'data.csv')
    frame = pd.read_csv(path).sort_values(['deliveryPeriodIndex', 'timeStepIndex'])
    frame.loc[frame.index[1], column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        aigb_dataset(4, train_data_path=path)


def test_category_null_cfg_and_gradients():
    net = TemporalUnet(4, 4, 1, dim=8, num_categories=2, category_dropout=1.)
    x, t, prompts = torch.randn(2, 4, 4), torch.ones(2), torch.ones(2, 3)
    categories = torch.tensor([0, 1])
    net.eval()
    known = net(x, None, t, prompts, use_dropout=False, category_ids=categories)
    null = net(x, None, t, prompts, use_dropout=False, force_category_dropout=True)
    unknown = net(x, None, t, prompts, use_dropout=False, category_ids=torch.tensor([99, -1]))
    assert not torch.allclose(known, null)
    assert torch.allclose(null, unknown)
    net.train()
    dropped = net(x, None, t, prompts, use_dropout=True, category_ids=categories, force_dropout=True)
    forced = net(x, None, t, prompts, use_dropout=False, force_category_dropout=True, force_dropout=True)
    assert torch.allclose(dropped, forced)
    forced.square().sum().backward()
    assert net.category_embedding.weight.grad[2].abs().sum() > 0


def test_masked_loss_does_not_dilute():
    loss = WeightedStateL2(torch.ones(4, 2))
    pred = torch.ones(1, 4, 2)
    pred[:, 2:] = 100
    value, _ = loss(pred, torch.zeros_like(pred), torch.tensor([[True, True, False, False]]))
    assert value.item() == 1.


class IncreasingQ:
    def __call__(self, states, actions, categories, cpa):
        return actions.expand(-1, 3)


def test_local_proposals_relabel_only_accepted_and_bound_actions():
    torch.manual_seed(5)
    states, actions = torch.zeros(100, 4), torch.ones(100, 1)
    labels = torch.tensor([[1., .5, 1.]]).repeat(100, 1)
    result, pseudo, accepted = propose_actions(IncreasingQ(), states, actions,
        torch.zeros(100, dtype=torch.long), labels, std=.5, probability=1.,
        action_max=1.2, improvement_min=0., max_return_delta=.05)
    assert accepted.any() and (~accepted).any()
    assert result.max() <= 1.2
    assert torch.equal(result[~accepted], actions[~accepted])
    assert torch.equal(pseudo[~accepted], labels[~accepted])
    assert torch.all(pseudo[accepted, 2] == -1.)
    assert torch.all(pseudo[:, 0] <= 1.05)
    assert torch.equal(pseudo[:, 1], labels[:, 1])
    _, unchanged, rejected = propose_actions(IncreasingQ(), states, actions,
        torch.zeros(100, dtype=torch.long), labels, probability=0.)
    assert not rejected.any() and torch.equal(unchanged, labels)


def test_q_and_diffusion_train_roundtrip(tmp_path):
    dataset = aigb_dataset(4, train_data_path=write_data(tmp_path / 'data.csv', lengths=(4, 1)))
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    q = QEnsemble(4, 1, 2)
    qloss = q.loss(batch)
    qloss.backward()
    assert torch.isfinite(qloss)
    assert q.embedding.weight.grad.abs().sum() > 0
    q.eval().requires_grad_(False)
    model = DFUSER(dim_obs=4, step_len=4, n_timesteps=2, num_categories=2,
                   model_dim=8, metadata=dataset.metadata)
    loss, components = model.trainStep(batch['states'], batch['actions'], batch['returns'],
                                      batch['masks'], batch['category_ids'], q=q, exploration={})
    assert torch.isfinite(loss)
    assert all(torch.isfinite(c) for c in components)
    model.save_net(tmp_path)
    restored = DFUSER.from_checkpoint(tmp_path / 'diffuser.pt')
    for step in [0, 3]:
        x = torch.zeros(4, 5)
        x[:, -1] = step
        torch.manual_seed(1)
        original = model(x, category_id=10, cpa=2.)
        torch.manual_seed(1)
        loaded = restored(x, category_id=10, cpa=2.)
        assert torch.equal(original, loaded)
        assert (loaded >= 0).all() and (loaded <= 10).all()
    assert torch.isfinite(restored(x, category_id=999, cpa=2.)).all()


def test_runner_with_exploration(tmp_path):
    path = write_data(tmp_path / 'data.csv')
    model = run_decision_diffuser(train_data_path=path, save_path=tmp_path / 'model',
        train_epoch=1, batch_size=2, exploration=True, q_epochs=1, model_dim=8, n_timesteps=2, device='cpu')
    assert (tmp_path / 'model' / 'q_ensemble.pt').exists()
    assert (tmp_path / 'model' / 'training_config.json').exists()
    assert model.metadata['categories'] == [10, 30]


class UncertainQ:
    def __call__(self, states, actions, categories, cpa):
        return actions + actions.new_tensor([-10., 0., 10.])


def test_disagreement_rejects_candidates():
    actions = torch.ones(20, 1)
    _, _, accepted = propose_actions(UncertainQ(), torch.zeros(20, 4), actions,
        torch.zeros(20, dtype=torch.long), torch.ones(20, 3), probability=1., uncertainty_max=.1)
    assert not accepted.any()


def test_accepted_augmentation_backward(tmp_path):
    dataset = aigb_dataset(4, train_data_path=write_data(tmp_path / 'data.csv'))
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = DFUSER(dim_obs=4, step_len=4, n_timesteps=2, num_categories=2, model_dim=8)
    loss, _ = model.trainStep(batch['states'], batch['actions'], batch['returns'],
        batch['masks'], batch['category_ids'], q=IncreasingQ(),
        exploration=dict(probability=1., std=.5, improvement_min=0.))
    assert model.last_metrics['accepted_proposals'] > 0
    assert torch.isfinite(loss)


def test_strategy_bids_and_empty_history(tmp_path):
    from bidding_train_env.strategy.dd_bidding_strategy import DdBiddingStrategy
    metadata = dict(state_mean=[0.] * 16, state_std=[1.] * 16,
                    categories=[30], return_scale=10., cpa_scale=2.)
    model = DFUSER(dim_obs=16, num_categories=1, model_dim=8, n_timesteps=2, metadata=metadata)
    model.save_net(tmp_path)
    strategy = DdBiddingStrategy(category=30, cpa=2., model_path=tmp_path / 'diffuser.pt')
    values = np.array([.1, .2, .3])
    bids = strategy.bidding(0, values, np.zeros(3), [], [], [], [], [])
    assert bids.shape == values.shape
    assert np.isfinite(bids).all() and (bids >= 0).all()
    assert np.allclose(bids / values, (bids / values)[0])
    assert strategy.bidding(0, np.array([]), np.array([]), [], [], [], [], []).size == 0
    strategy.reset()
    assert not strategy.input.any()


def test_cfg_weights_select_the_expected_branches():
    model = DFUSER(dim_obs=4, step_len=4, n_timesteps=2, num_categories=2, model_dim=8)
    diffusion = model.diffuser
    model.eval()
    x, t, prompts = torch.randn(1, 4, 4), torch.ones(1, dtype=torch.long), torch.ones(1, 3)
    categories = torch.tensor([1])
    for prompt_w, category_w in [(1., 1.), (0., 0.), (1., 0.)]:
        diffusion.condition_guidance_w, diffusion.category_guidance_w = prompt_w, category_w
        direct = diffusion.model(x, None, t, prompts, use_dropout=False, category_ids=categories,
                                 force_dropout=(prompt_w == 0), force_category_dropout=(category_w == 0))
        start = diffusion.predict_start_from_noise(x, t, direct)
        expected = diffusion.q_posterior(start, x, t)[0]
        actual = diffusion.p_mean_variance(x, None, t, prompts, categories)[0]
        assert torch.allclose(actual, expected, atol=1e-5)
