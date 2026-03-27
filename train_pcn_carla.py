import argparse
import os
import sys

import numpy as np

from carla_mo_gym_env import CarlaMOGymEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCN on CARLA MO wrapper.")
    parser.add_argument(
        "--morl_repo",
        type=str,
        default=r"C:\Users\12482\Documents\GitHub\morl-baselines",
        help="Absolute path to cloned morl-baselines repository.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town04")
    parser.add_argument("--episode_length", type=int, default=200)
    parser.add_argument("--delta_seconds", type=float, default=0.05)
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--total_timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.morl_repo):
        raise FileNotFoundError(f"morl-baselines path does not exist: {args.morl_repo}")
    if args.morl_repo not in sys.path:
        sys.path.insert(0, args.morl_repo)

    try:
        from morl_baselines.multi_policy.pcn.pcn import PCN
    except Exception as exc:
        raise ImportError(
            "Could not import PCN from morl-baselines. "
            "Install dependencies in your Python env first (e.g., pip install -e <morl_repo>)."
        ) from exc

    def make_env():
        return CarlaMOGymEnv(
            host=args.host,
            port=args.port,
            town=args.town,
            episode_length=args.episode_length,
            delta_seconds=args.delta_seconds,
            action_repeat=args.action_repeat,
        )

    env = make_env()
    eval_env = make_env()

    # Reward dim is 3, PCN command includes horizon => length 4.
    scaling_factor = np.array([1.0, 1.0, 1.0, 0.1], dtype=np.float32)
    ref_point = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    max_return = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    agent = PCN(
        env,
        scaling_factor=scaling_factor,
        learning_rate=1e-3,
        batch_size=256,
        log=False,
        seed=args.seed,
        project_name="CARLA-MORL",
        experiment_name="PCN-CARLA",
    )

    try:
        agent.train(
            eval_env=eval_env,
            total_timesteps=int(args.total_timesteps),
            ref_point=ref_point,
            num_er_episodes=10,
            max_buffer_size=50,
            num_model_updates=20,
            num_step_episodes=5,
            max_return=max_return,
        )
    finally:
        eval_env.close()
        env.close()


if __name__ == "__main__":
    main()

