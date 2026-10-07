"""Train and evaluate PPO for the complete cf2x_L250 50 cm takeoff/hover phase.

Run from the repository root with ``python crazyflow/rl_takeoff_full_mission/train_ppo.py``.
Each episode is eight seconds: PPO takes off and must keep the vehicle stable at 50 cm for the
final five seconds. Checkpoints are written to ``crazyflow/saves/cf2x_L250_ppo_full_mission_50cm_5s``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from jax import Array

from crazyflow.rl_takeoff_full_mission.env import (
    ACT_HIGH,
    ACT_LOW,
    EPISODE_STEPS,
    HOVER_STEPS,
    HOVER_XY_RADIUS,
    MASS_KG,
    MASS_RANDOMIZATION,
    MAX_DEPLOY_TILT_DEG,
    MAX_HEIGHT,
    MELLINGER_CONTROLLER_MASS_KG,
    OVERSHOOT_MARGIN,
    ROTOR_TAU_RANDOMIZATION,
    TARGET_HEIGHT,
    THRUST_RANDOMIZATION,
    TakeoffEnv,
    VBuffer,
    W_ALT,
    W_MISSION_COMPLETE,
    W_OVERSHOOT,
    W_SMOOTH,
    W_VERTICAL_VELOCITY,
    W_XY,
)
from crazyflow.control import Control


def _initial_policy_bias(key: Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> Array:
    """Start near a takeoff command while keeping attitude commands centered."""
    del key
    return jnp.zeros(shape, dtype=dtype).at[-1].set(0.5)


class ActorCritic(nn.Module):
    """Small MLP policy and value function for the fully observed flight state."""

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array, Array]:
        actor = nn.tanh(nn.Dense(128)(obs))
        actor = nn.tanh(nn.Dense(128)(actor))
        mean = nn.Dense(
            4,
            kernel_init=nn.initializers.orthogonal(0.01),
            # The identified rotor time constant is about 0.14 s. A positive collective-thrust
            # prior gives the rotors enough spin-up command to leave the floor during exploration.
            bias_init=_initial_policy_bias,
            name="policy_mean",
        )(actor)
        critic = nn.tanh(nn.Dense(128)(obs))
        critic = nn.tanh(nn.Dense(128)(critic))
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
            name="value",
        )(critic)[..., 0]
        log_std = self.param("log_std", nn.initializers.constant(-1.5), (4,))
        return mean, jnp.broadcast_to(log_std, mean.shape), value


def _log_prob(mean: Array, log_std: Array, latent: Array) -> Array:
    """Log probability after the tanh squashing transform (affine action scaling cancels)."""
    inv_std = jnp.exp(-log_std)
    gaussian = -0.5 * jnp.square((latent - mean) * inv_std) - log_std - 0.5 * jnp.log(2 * jnp.pi)
    squash_jacobian = jnp.log(1.0 - jnp.square(jnp.tanh(latent)) + 1e-6)
    return jnp.sum(gaussian - squash_jacobian, axis=-1)


def _action_from_latent(latent: Array) -> Array:
    unit_action = jnp.tanh(latent)
    attitude = ACT_LOW[:3] + (unit_action[..., :3] + 1.0) * 0.5 * (ACT_HIGH[:3] - ACT_LOW[:3])
    # Center collective thrust near hover and reserve a compact interval for the policy to shape.
    # This prevents an early negative velocity response from cutting thrust below takeoff level.
    thrust = (0.32 + 0.16 * unit_action[..., 3:4]).clip(ACT_LOW[3], ACT_HIGH[3])
    return jnp.concatenate((attitude, thrust), axis=-1)


def _sample_action(
    model: ActorCritic, params: Any, obs: Array, key: Array
) -> tuple[Array, Array, Array, Array]:
    mean, log_std, value = model.apply(params, obs)
    latent = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    return _action_from_latent(latent), latent, _log_prob(mean, log_std, latent), value


def _deterministic_action(model: ActorCritic, params: Any, obs: Array) -> Array:
    mean, _, _ = model.apply(params, obs)
    return _action_from_latent(mean)


def _collect(
    env: TakeoffEnv,
    model: ActorCritic,
    params: Any,
    carry: tuple[VBuffer, Array, Array, Array, Array],
    key: Array,
) -> tuple[tuple[VBuffer, Array, Array, Array, Array], dict[str, Array]]:
    """Collect a full on-policy batch with the simulator's JAX scan."""
    def one_step(carry_, _):
        vbuf, prev_action, t, ep_return, sim_key, rng_key = carry_
        obs = env.observation(vbuf.to_sim_data(env.template), prev_action, vbuf.anchor_yaw)
        rng_key, action_key, reset_key = jax.random.split(rng_key, 3)
        action, latent, old_logprob, value = _sample_action(
            model, params, obs, action_key
        )
        (vbuf, prev_action, t, ep_return, sim_key), (next_obs, info) = env.step(
            (vbuf, prev_action, t, ep_return, sim_key), action, reset_key
        )
        _, _, next_value = model.apply(params, next_obs)
        transition = {
            "obs": obs,
            "latent": latent,
            "logprob": old_logprob,
            "value": value,
            "next_value": next_value,
            "reward": info["reward"],
            "done": info["done"],
            "z": info["z"],
            "alt_err": info["alt_err"],
            "tilt_deg": info["tilt_deg"],
            "failed": info["failed"],
            "timeout": info["timeout"],
            "ep_return": info["ep_return"],
        }
        return (vbuf, prev_action, t, ep_return, sim_key, rng_key), transition

    initial = (*carry, key)
    final, batch = jax.lax.scan(one_step, initial, None, length=env.episode_steps)
    return final[:5], batch


def _advantages(batch: dict[str, Array], gamma: float, gae_lambda: float) -> tuple[Array, Array]:
    rewards = batch["reward"]
    values = batch["value"]
    next_values = batch["next_value"]
    dones = batch["done"]

    def backward(carry, transition):
        next_adv = carry
        reward, value, next_value, done = transition
        delta = reward + gamma * next_value * (1.0 - done) - value
        advantage = delta + gamma * gae_lambda * (1.0 - done) * next_adv
        return advantage, advantage

    _, advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(values[0]),
        (rewards, values, next_values, dones),
        reverse=True,
    )
    return advantages, advantages + values


def _ppo_update(
    model: ActorCritic,
    params: Any,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    batch: dict[str, Array],
    advantages: Array,
    returns: Array,
    key: Array,
    epochs: int,
    minibatch_size: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[Any, Any, Array]:
    n = batch["obs"].shape[0]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    data = {k: v for k, v in batch.items() if k in ("obs", "latent", "logprob", "value")}
    data["advantage"] = advantages
    data["returns"] = returns
    num_minibatches = n // minibatch_size

    def loss_fn(p, mb):
        mean, log_std, value = model.apply(p, mb["obs"])
        logprob = _log_prob(mean, log_std, mb["latent"])
        ratio = jnp.exp(logprob - mb["logprob"])
        unclipped = ratio * mb["advantage"]
        clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb["advantage"]
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        value_loss = 0.5 * jnp.mean(jnp.square(value - mb["returns"]))
        entropy_estimate = -jnp.mean(logprob)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_estimate
        return loss, jnp.array([policy_loss, value_loss, entropy_estimate])

    def one_minibatch(carry, indices):
        p, state, totals = carry
        mb = jax.tree.map(lambda x: x[indices], data)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, mb)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return (p, state, totals + metrics), loss

    def one_epoch(carry, _):
        p, state, rng_key, metrics = carry
        rng_key, shuffle_key = jax.random.split(rng_key)
        indices = jax.random.permutation(shuffle_key, n).reshape((num_minibatches, minibatch_size))
        (p, state, metrics), losses = jax.lax.scan(
            one_minibatch, (p, state, metrics), indices
        )
        return (p, state, rng_key, metrics), losses

    initial = (params, opt_state, key, jnp.zeros((3,), dtype=jnp.float32))
    (params, opt_state, _, metric_sum), _ = jax.lax.scan(one_epoch, initial, None, length=epochs)
    return params, opt_state, metric_sum / (epochs * num_minibatches)


def _evaluate(
    env: TakeoffEnv, model: ActorCritic, params: Any, key: Array,
    episodes: int = 128, randomize_yaw: bool = True, randomize_plant: bool = True,
) -> dict[str, float]:
    eval_env = TakeoffEnv(
        n_worlds=episodes,
        episode_steps=EPISODE_STEPS,
        randomize=randomize_plant,
        randomize_yaw=randomize_yaw,
        mass_kg=MASS_KG,
        control=env.control_mode,
        seed=19,
        device=env.device,
    )
    vbuf, _, prev_action = eval_env.reset_buffer(key)
    t = jnp.zeros((episodes,), dtype=jnp.int32)
    ep_return = jnp.zeros((episodes,), dtype=jnp.float32)
    sim_key = key

    def one_step(carry, _):
        vbuf_, prev_action_, t_, ep_return_, sim_key_, rng_key = carry
        obs = eval_env.observation(
            vbuf_.to_sim_data(eval_env.template), prev_action_, vbuf_.anchor_yaw
        )
        action = _deterministic_action(model, params, obs)
        rng_key, reset_key = jax.random.split(rng_key)
        result, (next_obs, info) = eval_env.step(
            (vbuf_, prev_action_, t_, ep_return_, sim_key_), action, reset_key
        )
        pos_xy = info["pos_xy"]
        speed = info["vel"]
        return (*result, rng_key), (info["z"], info["tilt_deg"], info["failed"], pos_xy, speed)

    (_, _, _, _, _, _), (z, tilt, failed, xy, speed) = jax.jit(
        lambda c, k: jax.lax.scan(one_step, (*c, k), None, length=EPISODE_STEPS)
    )((vbuf, prev_action, t, ep_return, sim_key), key)
    z_np = np.asarray(z)
    tilt_np = np.asarray(tilt)
    failed_np = np.asarray(failed)
    xy_np = np.asarray(xy)
    speed_np = np.asarray(speed)
    tail = z_np[-HOVER_STEPS:]
    tail_xy = xy_np[-HOVER_STEPS:]
    tail_speed = speed_np[-HOVER_STEPS:]
    tail_mean = tail.mean(axis=0)
    tail_std = tail.std(axis=0)
    tail_xy_radius = np.linalg.norm(tail_xy, axis=-1).max(axis=0)
    tail_speed_mean = tail_speed.mean(axis=0)
    fail_rate = failed_np.any(axis=0).mean()
    final_tilt = tilt_np[-HOVER_STEPS:].max(axis=0)
    episode_max_height = z_np.max(axis=0)
    episode_max_xy_radius = np.linalg.norm(xy_np, axis=-1).max(axis=0)
    episode_max_speed = speed_np.max(axis=0)
    height_gate = (np.abs(tail - TARGET_HEIGHT) < 0.03).all(axis=0)
    xy_gate = (np.linalg.norm(tail_xy, axis=-1) < HOVER_XY_RADIUS).all(axis=0)
    speed_gate = (tail_speed < 0.15).all(axis=0)
    tilt_gate = (tilt_np[-HOVER_STEPS:] < 10.0).all(axis=0)
    no_failure_gate = ~failed_np.any(axis=0)
    success = height_gate & xy_gate & speed_gate & tilt_gate & no_failure_gate
    eval_env.sim.close()
    return {
        "success_rate": float(success.mean()),
        "failure_rate": float(fail_rate),
        "tail_height_mean_m": float(tail_mean.mean()),
        "tail_height_std_m": float(tail_std.mean()),
        "tail_abs_error_m": float(np.abs(tail_mean - TARGET_HEIGHT).mean()),
        "tail_tilt_max_deg": float(final_tilt.mean()),
        "tail_xy_radius_max_m": float(tail_xy_radius.mean()),
        "tail_speed_mean_mps": float(tail_speed_mean.mean()),
        "five_second_hover_success_rate": float(success.mean()),
        "height_gate_rate": float(height_gate.mean()),
        "xy_gate_rate": float(xy_gate.mean()),
        "speed_gate_rate": float(speed_gate.mean()),
        "tilt_gate_rate": float(tilt_gate.mean()),
        "no_failure_gate_rate": float(no_failure_gate.mean()),
        "episode_max_height_m": float(episode_max_height.mean()),
        "episode_max_xy_radius_m": float(episode_max_xy_radius.mean()),
        "episode_max_speed_mps": float(episode_max_speed.mean()),
        "episode_max_height_worst_m": float(episode_max_height.max()),
        "episode_max_xy_radius_worst_m": float(episode_max_xy_radius.max()),
        "episode_max_speed_worst_mps": float(episode_max_speed.max()),
    }


def _save(path: Path, params: Any, metrics: dict[str, float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))
    metadata = {
        "algorithm": "PPO",
        "drone": "cf2x_L250",
        "mass_kg": MASS_KG,
        "plant_domain_randomization": {
            "mass_fraction": MASS_RANDOMIZATION,
            "thrust_efficiency_fraction": THRUST_RANDOMIZATION,
            "rotor_response_fraction": ROTOR_TAU_RANDOMIZATION,
        },
        "target_height_m": TARGET_HEIGHT,
        "episode_duration_s": EPISODE_STEPS / 50.0,
        "required_final_hover_s": HOVER_STEPS / 50.0,
        "landing_trained": False,
        "policy_frequency_hz": 50,
        "control": "[roll_rad, pitch_rad, yaw_rad, collective_thrust_N]",
        "policy_action_mapping": "attitude affine-tanh; collective_thrust_N = clip(0.32 + 0.16*tanh(latent_thrust), 0.05127, 0.48); mapped to state acceleration through Mellinger mass",
        "sim_control_interface": str(args.control_interface),
        "mellinger_controller_mass_kg": MELLINGER_CONTROLLER_MASS_KG,
        "reward_weights": {
            "altitude": W_ALT,
            "linear_velocity": 1.5,
            "vertical_velocity_extra": W_VERTICAL_VELOCITY,
            "tilt": 1.0,
            "xy_position": W_XY,
            "angular_velocity": 0.03,
            "action_effort": 0.02,
            "action_smoothness": W_SMOOTH,
            "overshoot_start_m_above_target": OVERSHOOT_MARGIN,
            "overshoot_penalty": W_OVERSHOOT,
            "crash_or_height_limit": 20.0,
            "tilt_failure": 12.0,
            "mission_complete": W_MISSION_COMPLETE,
        },
        "safety_limits": {
            "training_max_height_m": MAX_HEIGHT,
            "onboard_height_geofence_m": 0.65,
            "hover_xy_radius_m": HOVER_XY_RADIUS,
            "max_deploy_tilt_deg": MAX_DEPLOY_TILT_DEG,
        },
        "action_low": np.asarray(ACT_LOW).tolist(),
        "action_high": np.asarray(ACT_HIGH).tolist(),
        "observation": "[anchor_x_m, anchor_y_m, z_minus_target_m, anchor_relative_quat_xyzw, anchor_vx_mps, anchor_vy_mps, vz_mps, body_wx_radps, body_wy_radps, body_wz_radps, previous_action_4, target_height_m]",
        "metrics": metrics,
        "training": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--control-interface", choices=("state", "attitude"), default="state")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument(
        "--init-checkpoint",
        type=Path,
        default=PACKAGE_ROOT / "saves" / "cf2x_L250_ppo_fullstate_remote_v2" / "best.msgpack",
        help="optional 30 cm PPO actor checkpoint used to warm-start the full-mission training",
    )
    init_group.add_argument(
        "--from-scratch",
        action="store_true",
        help="initialize new actor/critic parameters instead of loading the warm-start actor",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "saves" / "cf2x_L250_ppo_full_mission_50cm_5s",
    )
    args = parser.parse_args()
    if args.num_envs * EPISODE_STEPS % args.minibatch_size:
        raise ValueError("num-envs * episode-steps must be divisible by minibatch-size")

    key = jax.random.key(args.seed)
    key, init_key, reset_key = jax.random.split(key, 3)
    env = TakeoffEnv(
        n_worlds=args.num_envs,
        episode_steps=EPISODE_STEPS,
        randomize=True,
        mass_kg=MASS_KG,
        device=args.device,
        seed=args.seed,
        control=Control(args.control_interface),
    )
    vbuf, _, prev_action = env.reset_buffer(reset_key)
    carry = (
        vbuf,
        prev_action,
        jnp.zeros((args.num_envs,), dtype=jnp.int32),
        jnp.zeros((args.num_envs,), dtype=jnp.float32),
        env.sim.data.core.rng_key,
    )
    model = ActorCritic()
    params = model.init(init_key, jnp.zeros((1, env.obs_dim), dtype=jnp.float32))
    if not args.from_scratch:
        if not args.init_checkpoint.is_file():
            raise FileNotFoundError(f"warm-start checkpoint not found: {args.init_checkpoint}")
        params = serialization.from_bytes(params, args.init_checkpoint.read_bytes())
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.learning_rate, eps=1e-5))
    opt_state = optimizer.init(params)

    def train_iteration(params_, opt_state_, carry_, rng_key):
        rng_key, rollout_key, update_key = jax.random.split(rng_key, 3)
        carry_, rollout = _collect(env, model, params_, carry_, rollout_key)
        adv, returns = _advantages(rollout, 0.99, 0.95)
        flattened = {
            k: v.reshape((-1, *v.shape[2:])) if v.ndim >= 3 else v.reshape((-1,))
            for k, v in rollout.items()
        }
        adv = adv.reshape((-1,))
        returns = returns.reshape((-1,))
        params_, opt_state_, losses = _ppo_update(
            model,
            params_,
            optimizer,
            opt_state_,
            flattened,
            adv,
            returns,
            update_key,
            args.epochs,
            args.minibatch_size,
            0.2,
            0.5,
            0.003,
        )
        return params_, opt_state_, carry_, rng_key, rollout, losses

    compiled_iteration = jax.jit(train_iteration)
    best_score = float("inf")
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"PPO full mission: {args.num_envs} envs x {EPISODE_STEPS} steps/iteration, "
        f"{args.iterations} iterations; mass={MASS_KG:.4f} kg, "
        f"Mellinger mass={MELLINGER_CONTROLLER_MASS_KG:.4f} kg, target={TARGET_HEIGHT:.2f} m, "
        f"hover window={HOVER_STEPS / 50.0:.1f} s; "
        f"domain rand mass/thrust/rotor=±{100*MASS_RANDOMIZATION:.0f}%/"
        f"±{100*THRUST_RANDOMIZATION:.0f}%/±{100*ROTOR_TAU_RANDOMIZATION:.0f}%; "
        f"yaw randomized; hover XY gate={HOVER_XY_RADIUS:.2f} m; "
        f"interface={args.control_interface}; backend={jax.default_backend()}"
    )
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        key, iter_key = jax.random.split(key)
        params, opt_state, carry, key, rollout, losses = compiled_iteration(
            params, opt_state, carry, iter_key
        )
        jax.block_until_ready(losses)
        timeout = np.asarray(rollout["timeout"])
        returns = np.asarray(rollout["ep_return"])
        completed = np.asarray(rollout["done"])
        completed_returns = returns[completed]
        reward_mean = float(completed_returns.mean()) if completed_returns.size else float("nan")
        elapsed = time.perf_counter() - started
        print(
            f"iter {iteration:04d}/{args.iterations}  "
            f"ep_return={reward_mean:8.2f}  "
            f"z={float(np.asarray(rollout['z']).mean()):.3f}m  "
            f"tilt={float(np.asarray(rollout['tilt_deg']).mean()):.1f}deg  "
            f"fail={float(np.asarray(rollout['failed']).mean()):.3f}  "
            f"pi={float(losses[0]):.4f} vf={float(losses[1]):.4f}  "
            f"elapsed={elapsed / 60:.1f}min",
            flush=True,
        )

        if iteration % args.eval_interval == 0 or iteration == args.iterations:
            key, eval_key = jax.random.split(key)
            metrics = _evaluate(env, model, params, eval_key)
            print("  eval:", json.dumps(metrics), flush=True)
            score = (
                10.0 * (1.0 - metrics["five_second_hover_success_rate"])
                + 4.0 * metrics["failure_rate"]
                + metrics["tail_abs_error_m"]
                + 2.0 * metrics["tail_height_std_m"]
                + 0.75 * metrics["tail_xy_radius_max_m"]
                + 0.5 * metrics["tail_speed_mean_mps"]
            )
            if score < best_score:
                best_score = score
                _save(args.output / "best.msgpack", params, metrics, args)
            _save(args.output / "latest.msgpack", params, metrics, args)

    env.sim.close()
    best_path = args.output / "best.msgpack"
    best_params = serialization.from_bytes(params, best_path.read_bytes())
    validation = _evaluate(env, model, best_params, jax.random.key(args.seed + 1009), episodes=512)
    validation_path = args.output / "validation_512.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"512-world randomized validation: {json.dumps(validation)}", flush=True)
    print(f"Finished. Best checkpoint: {args.output / 'best.msgpack'}")


if __name__ == "__main__":
    main()

