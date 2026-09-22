"""Validated episode data and train-only normalization for conditional DD."""
import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class aigb_dataset(Dataset):
    def __init__(self, step_len=48, train_data_path="data/trajectory/trajectory_data.csv",
                 metadata=None):
        frame = pd.read_csv(train_data_path)
        keys = ["deliveryPeriodIndex", "advertiserNumber"]
        required = keys + ["timeStepIndex", "state", "action", "reward",
                           "advertiserCategoryIndex", "CPAConstraint",
                           "realAllCost", "realAllConversion"]
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}; regenerate RL data first")
        if frame.empty or frame[required].isna().any().any():
            raise ValueError("Training data is empty or contains missing required values")
        frame = frame.sort_values(keys + ["timeStepIndex"])
        self.episodes = []
        for _, group in frame.groupby(keys, sort=False):
            times = group.timeStepIndex.to_numpy()
            if not np.array_equal(times, np.arange(len(times))) or len(times) > step_len:
                raise ValueError("Episodes must have unique contiguous steps starting at 0")
            if "done" in group:
                done = group.done.to_numpy()
                if not np.isin(done, [0, 1]).all() or done[:-1].any() or done[-1] != 1:
                    raise ValueError("Each episode must end at its first done=1")
            elif len(times) != step_len:
                raise ValueError("Short episodes require an explicit terminal done=1")
            for column in ["advertiserCategoryIndex", "CPAConstraint", "realAllCost", "realAllConversion"]:
                if group[column].nunique() != 1:
                    raise ValueError(f"Inconsistent episode field: {column}")
            states = np.asarray([ast.literal_eval(s) for s in group.state], dtype=np.float32)
            actions = group.action.to_numpy(dtype=np.float32)[:, None]
            rewards = group.reward.to_numpy(dtype=np.float32)
            cost, conversions, cpa = (float(group[c].iloc[0]) for c in
                                      ["realAllCost", "realAllConversion", "CPAConstraint"])
            category = float(group.advertiserCategoryIndex.iloc[0])
            if not category.is_integer() or category < 0:
                raise ValueError("Category must be a nonnegative integer")
            if states.ndim != 2 or not all(np.isfinite(a).all() for a in
                                          [states, actions, rewards, [cost, conversions, cpa]]):
                raise ValueError("Non-finite or malformed trajectory")
            if min(cost, conversions, cpa) < 0 or (actions < 0).any() or (rewards < 0).any():
                raise ValueError("Costs, constraints, actions and conversions must be nonnegative")
            # Zero spend / zero conversion is compliant; positive spend / zero conversion is not.
            compliance = float(cost <= cpa * conversions)
            rtg = np.cumsum(rewards[::-1])[::-1].copy()
            self.episodes.append(dict(states=states, actions=actions, rtg=rtg,
                                      total=float(rewards.sum()), category=int(category),
                                      cpa=cpa, compliance=compliance))
        all_states = np.concatenate([e['states'] for e in self.episodes])
        if metadata is None:
            categories = sorted({e['category'] for e in self.episodes})
            metadata = dict(state_mean=all_states.mean(0).tolist(),
                            state_std=np.maximum(all_states.std(0), 1e-3).tolist(),
                            return_scale=max(max(e['total'] for e in self.episodes), 1.),
                            cpa_scale=max(max(e['cpa'] for e in self.episodes), 1.),
                            categories=categories)
        self.metadata = metadata
        self.step_len = step_len
        self.num_of_states = all_states.shape[1]
        self.num_of_actions = 1
        self.category_map = {c: i for i, c in enumerate(metadata['categories'])}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        e = self.episodes[index]
        n = len(e['states'])
        states = torch.zeros(self.step_len, self.num_of_states)
        states[:n] = torch.tensor((e['states'] - np.asarray(self.metadata['state_mean'])) /
                                 np.asarray(self.metadata['state_std']), dtype=torch.float32)
        actions = torch.zeros(self.step_len, 1)
        actions[:n] = torch.from_numpy(e['actions'])
        rtg = torch.zeros(self.step_len)
        rtg[:n] = torch.from_numpy(e['rtg']) / self.metadata['return_scale']
        return dict(states=states, actions=actions,
                    returns=torch.tensor([e['total'] / self.metadata['return_scale'],
                                          e['cpa'] / self.metadata['cpa_scale'], e['compliance']]),
                    masks=torch.arange(self.step_len) < n, rtg=rtg,
                    category_ids=torch.tensor(self.category_map.get(e['category'], len(self.category_map))))
