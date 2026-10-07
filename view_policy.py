"""Replay a trained 50 cm / five-second PPO mission in Crazyflow's MuJoCo viewer."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import serialization
import mujoco
import numpy as np

from crazyflow.rl_takeoff_full_mission.env import (
    EPISODE_STEPS,
    HOVER_STEPS,
    MASS_KG,
    TARGET_HEIGHT,
    TakeoffEnv,
)
from crazyflow.rl_takeoff_full_mission.train_ppo import ActorCritic, _deterministic_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "saves"
        / "cf2x_L250_ppo_full_mission_50cm_5s"
        / "best.msgpack",
    )
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--hold", type=float, default=10.0, help="keep the final viewer frame open")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    # One extra internal step prevents the environment timeout from auto-resetting the viewer at
    # the final frame; replay itself still executes exactly the mission's 400 policy actions.
    env = TakeoffEnv(
        n_worlds=1,
        episode_steps=EPISODE_STEPS + 1,
        randomize=True,
        mass_kg=MASS_KG,
        seed=args.seed,
    )
    model = ActorCritic()
    template = model.init(jax.random.key(0), jnp.zeros((1, env.obs_dim), dtype=jnp.float32))
    params = serialization.from_bytes(template, args.checkpoint.read_bytes())

    @jax.jit
    def policy(obs: jax.Array) -> jax.Array:
        return _deterministic_action(model, params, obs)

    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    vbuf, _, prev_action = env.reset_buffer(reset_key)
    carry = (
        vbuf,
        prev_action,
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.float32),
        env.sim.data.core.rng_key,
    )
    camera = {"distance": 1.3, "elevation": -25.0, "azimuth": 135.0, "lookat": [0.0, 0.0, 0.4]}
    history: list[float] = []

    print(
        f"MuJoCo replay: cf2x_L250, target={TARGET_HEIGHT:.2f} m, "
        f"episode={EPISODE_STEPS / 50.0:.1f} s, final hover={HOVER_STEPS / 50.0:.1f} s"
    )
    env.sim.data = carry[0].to_sim_data(env.template)
    env.sim.render(mode="human", cam_config=camera)
    for _ in range(EPISODE_STEPS):
        start = time.perf_counter()
        vbuf, prev_action, t, ep_return, sim_key = carry
        obs = env.observation(vbuf.to_sim_data(env.template), prev_action, vbuf.anchor_yaw)
        action = policy(obs)
        key, step_key = jax.random.split(key)
        carry, (_, info) = env.step(carry, action, step_key)
        env.sim.data = carry[0].to_sim_data(env.template)
        env.sim.viewer.viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.02, 0.02, 0.02]),
            pos=np.array([0.0, 0.0, TARGET_HEIGHT]),
            rgba=np.array([1.0, 0.0, 0.0, 0.6]),
        )
        env.sim.render(mode="human")
        history.append(float(np.asarray(info["z"])[0]))
        remaining = 1.0 / 50.0 - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    tail = np.asarray(history[-HOVER_STEPS:])
    print(
        f"Final 5 s height: {tail.mean():.3f} ± {tail.std():.3f} m; "
        f"viewer remains open for {args.hold:.1f} s. Close the MuJoCo window to finish sooner."
    )
    time.sleep(max(0.0, args.hold))
    env.sim.close()


if __name__ == "__main__":
    main()
