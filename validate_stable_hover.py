"""Validate a checkpoint using the firmware's stable-before-hold timing in Crazyflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from crazyflow.rl_takeoff_full_mission.env import MASS_KG, TARGET_HEIGHT, TakeoffEnv
from crazyflow.rl_takeoff_full_mission.train_ppo import ActorCritic, _deterministic_action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "saves/cf2x_L250_ppo_takeoff_hover_yaw_robust_v3/best.msgpack",
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    steps = round(args.max_seconds * 50)
    if steps <= 300:
        parser.error("max-seconds must allow room for 0.5 s settling plus a 5 s hold")

    env = TakeoffEnv(
        n_worlds=args.episodes,
        episode_steps=steps + 1,
        randomize=True,
        randomize_yaw=True,
        mass_kg=MASS_KG,
        seed=args.seed,
    )
    model = ActorCritic()
    template = model.init(jax.random.key(0), jnp.zeros((1, env.obs_dim), dtype=jnp.float32))
    params = serialization.from_bytes(template, args.checkpoint.read_bytes())
    vbuf, _, prev_action = env.reset_buffer(jax.random.key(args.seed))
    carry = (
        vbuf,
        prev_action,
        jnp.zeros((args.episodes,), dtype=jnp.int32),
        jnp.zeros((args.episodes,), dtype=jnp.float32),
        env.sim.data.core.rng_key,
    )

    def one_step(carry_, inputs):
        vbuf_, prev_action_, t_, ep_return_, sim_key_ = carry_
        key = inputs
        obs = env.observation(vbuf_.to_sim_data(env.template), prev_action_, vbuf_.anchor_yaw)
        action = _deterministic_action(model, params, obs)
        result, (next_obs, info) = env.step(carry_, action, key)
        del next_obs
        return result, (info["z"], info["pos_xy"], info["vel"], info["tilt_deg"], info["failed"])

    keys = jax.random.split(jax.random.key(args.seed + 1), steps)
    final, history = jax.jit(
        lambda c, k: jax.lax.scan(one_step, c, k)
    )(carry, keys)
    del final
    z, xy, speed, tilt, failed = (np.asarray(x) for x in history)
    xy_radius = np.linalg.norm(xy, axis=-1)

    confirm_count = np.zeros(args.episodes, dtype=np.int32)
    hold_count = np.zeros(args.episodes, dtype=np.int32)
    holding = np.zeros(args.episodes, dtype=bool)
    success = np.zeros(args.episodes, dtype=bool)
    geofence = np.zeros(args.episodes, dtype=bool)
    for i in range(steps):
        geofence |= (np.abs(xy[i]).max(axis=-1) > 0.25) | (z[i] < -0.03) | (z[i] > 0.65)
        settle_ok = (
            (np.abs(z[i] - TARGET_HEIGHT) < 0.04)
            & (xy_radius[i] < 0.10)
            & (speed[i] < 0.15)
            & (tilt[i] < 10.0)
            & ~failed[i]
            & ~geofence
        )
        hold_ok = (
            (np.abs(z[i] - TARGET_HEIGHT) < 0.04)
            & (xy_radius[i] < 0.15)
            & (speed[i] < 0.20)
            & (tilt[i] < 12.0)
            & ~failed[i]
            & ~geofence
        )
        confirm_count = np.where(settle_ok, confirm_count + 1, 0)
        start_hold = (~holding) & (confirm_count >= 25)
        holding |= start_hold
        hold_count = np.where(holding & hold_ok, hold_count + 1, 0)
        completed = hold_count >= 250
        success |= completed
        holding &= ~completed
        confirm_count = np.where(completed, 0, confirm_count)
        hold_count = np.where(completed, 0, hold_count)

    result = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "max_mission_s": steps / 50.0,
        "target_confirmation_s": 0.5,
        "required_continuous_hover_s": 5.0,
        "stable_hover_success_rate": float(success.mean()),
        "failure_rate": float(np.asarray(failed).any(axis=0).mean()),
        "xy_geofence_rate": float(geofence.mean()),
        "final_5s_height_mean_m": float(z[-250:].mean()),
        "final_5s_height_std_m": float(z[-250:].std()),
        "final_5s_xy_radius_max_mean_m": float(xy_radius[-250:].max(axis=0).mean()),
        "max_abs_height_mean_m": float(z.max(axis=0).mean()),
        "max_xy_radius_mean_m": float(xy_radius.max(axis=0).mean()),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    env.sim.close()


if __name__ == "__main__":
    main()
