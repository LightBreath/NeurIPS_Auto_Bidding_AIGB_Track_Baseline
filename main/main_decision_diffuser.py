import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run.run_decision_diffuser import run_decision_diffuser


def main():
    parser = argparse.ArgumentParser(description="Category/CPA-conditioned Decision Diffusion")
    parser.add_argument('--train-data-path', default='data/trajectory/trajectory_data.csv')
    parser.add_argument('--save-path', default='saved_model/DDtest')
    parser.add_argument('--train-epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--category-dropout', type=float, default=0.15)
    parser.add_argument('--condition-guidance-w', type=float, default=1.2)
    parser.add_argument('--category-guidance-w', type=float, default=1.)
    parser.add_argument('--exploration', action='store_true')
    parser.add_argument('--q-epochs', type=int, default=10)
    parser.add_argument('--perturb-std', type=float, default=0.1)
    parser.add_argument('--perturb-probability', type=float, default=0.5)
    parser.add_argument('--uncertainty-max', type=float, default=0.1)
    parser.add_argument('--improvement-min', type=float, default=0.01)
    parser.add_argument('--max-return-delta', type=float, default=0.1)
    parser.add_argument('--action-max', type=float)
    parser.add_argument('--device')
    run_decision_diffuser(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
