"""Reinforcement learning environment: vertical takeoff to a 50 cm, five-second hover.

Task
----
The drone starts on the ground, takes off to 50 cm, and holds there for the final five seconds of
the episode. The policy commands
[roll, pitch, yaw, collective_thrust] at ``POLICY_FREQ`` Hz through the Crazyflie's own Mellinger
attitude controller, which runs at ``CTRL_FREQ`` Hz inside the simulation. Only the outer
translational loop is learned.

Why this shape
--------------
* ``cf2x_L250`` with ``first_principles`` dynamics is the platform under test (measured flight mass
  32.4 g).
* The attitude abstraction is deliberate: roll/pitch/yaw + collective thrust is exactly what the
  Crazyflie firmware exposes, so a policy trained here is a candidate for the real vehicle.
* Everything is a pure function of ``SimData`` and therefore ``jit``/``scan`` safe. The step pipeline
  is the simulation's own ``build_step_fn``; state is carried as a pytree through ``lax.scan``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

import crazyflow.sim.functional as F
from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.data import SimData

# ---------------------------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------------------------

DRONE = "cf2x_L250"
DYNAMICS = Dynamics.first_principles
CONTROL = Control.state

SIM_FREQ = 500
CTRL_FREQ = 500  # Mellinger inner loop, matches the firmware
POLICY_FREQ = 50  # learned outer loop

N_SUBSTEPS = SIM_FREQ // POLICY_FREQ
POLICY_DT = 1.0 / POLICY_FREQ

TARGET_HEIGHT = 0.50
MASS_KG = 0.0324  # measured mass supplied for this vehicle, including the installed battery
# Match the onboard OOT Mellinger instance, configured to the measured 32.4 g airframe mass.
MELLINGER_CONTROLLER_MASS_KG = 0.0324
# The attitude action space for this drone: [roll, pitch, yaw, collective thrust]
MAX_DEPLOY_TILT_DEG = 12.0
MAX_DEPLOY_TILT = jnp.deg2rad(MAX_DEPLOY_TILT_DEG)
ACT_LOW = jnp.array([-MAX_DEPLOY_TILT, -MAX_DEPLOY_TILT, 0.0, 0.05127031], dtype=jnp.float32)
ACT_HIGH = jnp.array([MAX_DEPLOY_TILT, MAX_DEPLOY_TILT, 0.0, 0.48], dtype=jnp.float32)

EPISODE_STEPS = 400  # 8 s at 50 Hz: ascent plus a final five-second hover window
HOVER_STEPS = 250  # 5 s at 50 Hz
CRASH_Z = 0.005  # near the floor, below the reset clearance
TAKEOFF_GRACE_STEPS = 40  # allow the simulated rotors to spool up before judging floor contact
MAX_HEIGHT = 0.60  # keep training rollouts below the onboard 65 cm height geofence
MAX_TILT = 0.85  # rad; beyond this the episode fails, which blocks tumbling exploits
GROUND_CLEARANCE = 0.02  # start this far above the floor so the contact clip is never engaged
HOVER_XY_RADIUS = 0.05
MASS_RANDOMIZATION = 0.10
THRUST_RANDOMIZATION = 0.10
ROTOR_TAU_RANDOMIZATION = 0.15

# Observation: [pos_xy(2), height_err(1), quat(4), vel(3), ang_vel(3), prev_action(4), target(1)]
OBS_DIM = 18
ACT_DIM = 4

# Reward weights
W_ALT = 16.0
W_VEL = 1.5
W_VERTICAL_VELOCITY = 6.0
W_TILT = 1.0
W_XY = 60.0
W_SPIN = 0.03
W_ACT = 0.02
W_SMOOTH = 0.10
W_CRASH = 20.0
W_TILT_FAIL = 12.0
W_MISSION_COMPLETE = 30.0
W_OVERSHOOT = 500.0
OVERSHOOT_MARGIN = 0.05

# Observation scaling, so every network input is O(1)
SCALE_POS_XY = 1.0
SCALE_POS_ERR = 2.0
SCALE_VEL = 1.5
SCALE_ANG_VEL = 5.0


class VBuffer(struct.PyTreeNode):
    """The parts of ``SimData`` that vary over time.

    ``params`` and the controller parameter dicts are dropped: they are constant, and carrying them
    through every scan step would be pure overhead.
    """

    states: Any
    controls: Any
    core: Any
    anchor_yaw: Array

    @classmethod
    def from_sim_data(cls, data: SimData, anchor_yaw: Array) -> VBuffer:
        return cls(states=data.states, controls=data.controls, core=data.core, anchor_yaw=anchor_yaw)

    def to_sim_data(self, template: SimData) -> SimData:
        return template.replace(states=self.states, controls=self.controls, core=self.core)


def action_to_full_state(data: SimData, action: Array, anchor_yaw: Array) -> Array:
    """Convert PPO [roll, pitch, yaw, thrust_N] to Crazyflie Mellinger full-state commands.

    Position and velocity references equal the current estimate, so the positional feedback terms
    are zero at the controller update. The feed-forward acceleration then requests the action's
    thrust vector using the measured Mellinger controller mass (32.4 g), matching the OOT firmware.
    """
    roll, pitch, yaw, thrust = (action[:, i] for i in range(4))
    world_yaw = anchor_yaw + yaw
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(world_yaw), jnp.sin(world_yaw)
    body_z = jnp.stack(
        (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr), axis=-1
    )
    acceleration = thrust[:, None] * body_z / MELLINGER_CONTROLLER_MASS_KG
    acceleration = acceleration.at[:, 2].add(-9.81)
    states = data.states
    pos = states.pos[:, 0, :]
    vel = states.vel[:, 0, :]
    yaw_quat = jnp.stack(
        (jnp.zeros_like(world_yaw), jnp.zeros_like(world_yaw),
         jnp.sin(world_yaw / 2), jnp.cos(world_yaw / 2)),
        axis=-1,
    )
    rates = jnp.zeros_like(pos)
    return jnp.concatenate((pos, vel, acceleration, yaw_quat, rates), axis=-1)


def _select(new: Any, old: Any, mask: Array) -> Any:
    """Elementwise ``where(mask, new, old)`` over a pytree, broadcasting along the world axis.

    Leaves whose leading dimension is not the world axis (static fields such as ``freq``, the scalar
    ``rng_key``/``mjx_synced``, and the control parameter dicts) are returned unchanged.
    """
    if isinstance(old, dict):
        return {k: _select(new[k], old[k], mask) for k in old}
    if hasattr(old, "replace") and hasattr(old, "__dataclass_fields__"):
        return old.replace(
            **{f: _select(getattr(new, f), getattr(old, f), mask) for f in old.__dataclass_fields__}
        )
    if not hasattr(old, "ndim") or old.ndim == 0:
        return old
    leading = old.shape[0]
    if leading != mask.shape[0]:
        return old  # constant leaf with a batch-like shape (e.g. drone_mocap_ids)
    return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - 1)), new, old)


class TakeoffEnv:
    """Vectorised takeoff-to-hover environment, written as pure JAX.

    ``obs_dim``/``act_dim`` are exposed so the trainer can size the network without Gymnasium.
    """

    obs_dim = OBS_DIM
    act_dim = ACT_DIM

    def __init__(
        self,
        n_worlds: int = 4096,
        target_height: float = TARGET_HEIGHT,
        episode_steps: int = EPISODE_STEPS,
        init_height_std: float = 0.005,
        init_vel_std: float = 0.04,
        init_ang_vel_std: float = 0.08,
        randomize: bool = True,
        randomize_yaw: bool = True,
        device: str = "cpu",
        seed: int = 0,
        mass_kg: float = MASS_KG,
        mass_randomization: float = MASS_RANDOMIZATION,
        thrust_randomization: float = THRUST_RANDOMIZATION,
        rotor_tau_randomization: float = ROTOR_TAU_RANDOMIZATION,
        control: Control = CONTROL,
    ):
        self.n_worlds = n_worlds
        self.target_height = target_height
        self.episode_steps = episode_steps
        self.init_height_std = init_height_std
        self.init_vel_std = init_vel_std
        self.init_ang_vel_std = init_ang_vel_std
        self.randomize = randomize
        self.randomize_yaw = randomize_yaw
        self.device = device
        self.mass_kg = mass_kg
        self.mass_randomization = mass_randomization
        self.thrust_randomization = thrust_randomization
        self.rotor_tau_randomization = rotor_tau_randomization
        self.control_mode = Control(control)

        self.sim = Sim(
            n_worlds=n_worlds,
            n_drones=1,
            drone=DRONE,
            dynamics=DYNAMICS,
            control=self.control_mode,
            freq=SIM_FREQ,
            state_freq=POLICY_FREQ,
            attitude_freq=CTRL_FREQ,
            force_torque_freq=CTRL_FREQ,
            device=device,
        )
        # Randomize plant parameters independently per world while keeping the Mellinger command
        # conversion fixed at the measured 32.4 g firmware value. This exposes the policy to mass,
        # thrust efficiency, and rotor response variation during training and evaluation.
        k_mass, k_thrust, k_tau = jax.random.split(jax.random.key(seed), 3)
        if randomize:
            mass_scale = jax.random.uniform(
                k_mass,
                (n_worlds, 1, 1),
                minval=1.0 - mass_randomization,
                maxval=1.0 + mass_randomization,
            )
            thrust_scale = jax.random.uniform(
                k_thrust,
                (n_worlds, 1, 1, 1),
                minval=1.0 - thrust_randomization,
                maxval=1.0 + thrust_randomization,
            )
            tau_scale = jax.random.uniform(
                k_tau,
                (n_worlds, 1, 1, 1),
                minval=1.0 - rotor_tau_randomization,
                maxval=1.0 + rotor_tau_randomization,
            )
        else:
            mass_scale = jnp.ones((n_worlds, 1, 1))
            thrust_scale = jnp.ones((n_worlds, 1, 1, 1))
            tau_scale = jnp.ones((n_worlds, 1, 1, 1))
        base_params = self.sim.data.params
        params = base_params.replace(
            mass=mass_kg * mass_scale,
            rpm2thrust=jnp.reshape(base_params.rpm2thrust, (1, 1, 1, 3)) * thrust_scale,
            rotor_dyn_coef=jnp.reshape(base_params.rotor_dyn_coef, (1, 1, 1, 4)) * tau_scale,
        )
        self.sim.data = self.sim.data.replace(params=params)
        self.sim.default_data = self.sim.default_data.replace(params=params)
        self.sim.reset()
        self._step_fn = self.sim.build_step_fn()
        self._template = self.sim.data

        # The simulation's default data already holds a zeroed state with zeroed controller
        # integrals and a zeroed rotor state. It only needs lifting off the floor.
        self._default_data = self.sim.default_data.replace(
            states=self.sim.default_data.states.replace(
                pos=self.sim.default_data.states.pos.at[..., 2].set(GROUND_CLEARANCE)
            )
        )
        self._seed_key = jax.random.key(seed)

    # -- accessors -----------------------------------------------------------------------------

    @property
    def step_fn(self) -> Callable:
        return self._step_fn

    @property
    def template(self) -> SimData:
        return self._template

    @property
    def default_data(self) -> SimData:
        return self._default_data

    def reset_buffer(self, key: Array) -> tuple[VBuffer, Array, Array]:
        """Sample a fresh start state for every world.

        Returns:
            The buffer, the matching observation, and ``(prev_action, t, episode_return)``.
        """
        if self.randomize:
            k1, k2, k3, k4 = jax.random.split(key, 4)
            shape = (self.n_worlds, 1)
            dz = jax.random.normal(k1, shape) * self.init_height_std
            vel = jax.random.normal(k2, (*shape, 3)) * self.init_vel_std
            ang_vel = jax.random.normal(k3, (*shape, 3)) * self.init_ang_vel_std
            if self.randomize_yaw:
                anchor_yaw = jax.random.uniform(
                    k4, (self.n_worlds,), minval=-jnp.pi, maxval=jnp.pi
                )
            else:
                anchor_yaw = jnp.zeros((self.n_worlds,))
        else:
            dz = jnp.zeros((self.n_worlds, 1))
            vel = jnp.zeros((self.n_worlds, 1, 3))
            ang_vel = jnp.zeros((self.n_worlds, 1, 3))
            anchor_yaw = jnp.zeros((self.n_worlds,))

        # Randomize takeoff heading and express the policy state/action in the anchor-local frame.
        quat = self._default_data.states.quat
        quat = quat.at[:, 0, 0].set(0.0)
        quat = quat.at[:, 0, 1].set(0.0)
        quat = quat.at[:, 0, 2].set(jnp.sin(anchor_yaw / 2.0))
        quat = quat.at[:, 0, 3].set(jnp.cos(anchor_yaw / 2.0))

        states = self._default_data.states.replace(
            pos=self._default_data.states.pos.at[..., 2].add(dz),
            quat=quat,
            vel=vel,
            ang_vel=ang_vel,
        )
        data = self._default_data.replace(states=states)
        vbuf = VBuffer.from_sim_data(data, anchor_yaw)
        prev_action = jnp.zeros((self.n_worlds, ACT_DIM))
        obs = self.observation(data, prev_action, anchor_yaw)
        return vbuf, obs, prev_action

    # -- observation ---------------------------------------------------------------------------

    def observation(self, data: SimData, prev_action: Array, anchor_yaw: Array) -> Array:
        states = data.states
        pos, quat = states.pos[:, 0, :], states.quat[:, 0, :]
        c, s = jnp.cos(anchor_yaw), jnp.sin(anchor_yaw)
        x_local = c * pos[:, 0] + s * pos[:, 1]
        y_local = -s * pos[:, 0] + c * pos[:, 1]
        vel_x_local = c * states.vel[:, 0, 0] + s * states.vel[:, 0, 1]
        vel_y_local = -s * states.vel[:, 0, 0] + c * states.vel[:, 0, 1]
        half_c, half_s = jnp.cos(anchor_yaw / 2.0), jnp.sin(anchor_yaw / 2.0)
        qx, qy, qz, qw = (quat[:, i] for i in range(4))
        quat_local = jnp.stack(
            (
                half_c * qx + half_s * qy,
                half_c * qy - half_s * qx,
                half_c * qz - half_s * qw,
                half_c * qw + half_s * qz,
            ),
            axis=-1,
        )
        height_err = (pos[:, 2] - self.target_height)[:, None]
        obs = jnp.concatenate(
            [
                jnp.stack((x_local, y_local), axis=-1) * SCALE_POS_XY,
                height_err * SCALE_POS_ERR,
                quat_local,
                jnp.stack((vel_x_local, vel_y_local, states.vel[:, 0, 2]), axis=-1) * SCALE_VEL,
                states.ang_vel[:, 0, :] * SCALE_ANG_VEL,
                prev_action,
                jnp.full_like(height_err, 1.0 / SCALE_POS_ERR),  # deployment-compatible target token
            ],
            axis=-1,
        )
        return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # -- reward --------------------------------------------------------------------------------

    def reward(
        self, data: SimData, action: Array, prev_action: Array, t: Array, anchor_yaw: Array
    ) -> tuple[Array, dict[str, Array]]:
        states = data.states
        pos, quat = states.pos[:, 0, :], states.quat[:, 0, :]
        alt_err = jnp.abs(pos[:, 2] - self.target_height)

        # exp() peak at the target, plus a wide term so the gradient survives a large error
        r_alt = W_ALT * (1.35 * jnp.exp(-4.0 * alt_err) - 0.45 - 0.35 * jnp.square(alt_err))
        c, s = jnp.cos(anchor_yaw), jnp.sin(anchor_yaw)
        x_local = c * pos[:, 0] + s * pos[:, 1]
        y_local = -s * pos[:, 0] + c * pos[:, 1]
        xy_local = jnp.stack((x_local, y_local), axis=-1)
        r_xy = -W_XY * jnp.sum(jnp.square(xy_local), axis=-1)
        r_vel = -W_VEL * jnp.sum(jnp.square(states.vel[:, 0, :]), axis=-1)
        r_vertical_velocity = -W_VERTICAL_VELOCITY * jnp.square(states.vel[:, 0, 2])
        overshoot = jnp.maximum(pos[:, 2] - (self.target_height + OVERSHOOT_MARGIN), 0.0)
        r_overshoot = -W_OVERSHOOT * jnp.square(overshoot)

        # XYZW quaternion: only qx/qy tilt the body Z axis; qz is yaw and must not count as tilt.
        x, y = quat[:, 0], quat[:, 1]
        upright = jnp.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)  # body z . world z
        r_tilt = -W_TILT * (1.0 - upright)

        r_spin = -W_SPIN * jnp.sum(jnp.square(states.ang_vel[:, 0, :]), axis=-1)
        r_act = -W_ACT * jnp.mean(jnp.square(action), axis=-1)
        r_smooth = -W_SMOOTH * jnp.mean(jnp.square(action - prev_action), axis=-1)

        crashed = (pos[:, 2] < CRASH_Z) & (t >= TAKEOFF_GRACE_STEPS)
        too_high = pos[:, 2] > MAX_HEIGHT
        tilted = jnp.abs(x) + jnp.abs(y) > jnp.sin(MAX_TILT / 2)
        reward = (
            r_alt + r_xy + r_vel + r_vertical_velocity + r_overshoot
            + r_tilt + r_spin + r_act + r_smooth
        )
        reward = jnp.where(crashed, reward - W_CRASH, reward)
        reward = jnp.where(too_high, reward - W_CRASH, reward)
        reward = jnp.where(tilted, reward - W_TILT_FAIL, reward)
        xy_radius = jnp.linalg.norm(xy_local, axis=-1)
        speed = jnp.linalg.norm(states.vel[:, 0, :], axis=-1)
        hover_gate = (
            (alt_err < 0.03)
            & (speed < 0.15)
            & (xy_radius < HOVER_XY_RADIUS)
            & (jnp.abs(x) + jnp.abs(y) < jnp.sin(jnp.deg2rad(10.0) / 2.0))
        )
        mission_end = (t + 1) >= self.episode_steps
        reward = reward + jnp.where(
            mission_end & hover_gate & ~crashed & ~too_high & ~tilted,
            W_MISSION_COMPLETE,
            0.0,
        )

        info = {
            "r_alt": r_alt,
            "r_xy": r_xy,
            "r_vel": r_vel,
            "r_vertical_velocity": r_vertical_velocity,
            "r_overshoot": r_overshoot,
            "r_tilt": r_tilt,
            "pos_xy": xy_local,
            "alt_err": alt_err,
            "z": pos[:, 2],
            "vel": speed,
            "tilt_deg": jnp.degrees(jnp.arccos(upright)),
            "thrust": action[:, 3],
            "crashed": crashed,
            "too_high": too_high,
            "tilted": tilted,
        }
        return reward, info

    @staticmethod
    def _failed(pos: Array, quat: Array, t: Array) -> Array:
        crashed = (pos[:, 2] < CRASH_Z) & (t >= TAKEOFF_GRACE_STEPS)
        too_high = pos[:, 2] > MAX_HEIGHT
        tilted = jnp.abs(quat[:, 0]) + jnp.abs(quat[:, 1]) > jnp.sin(MAX_TILT / 2)
        return crashed | too_high | tilted

    # -- environment step ----------------------------------------------------------------------

    def step(
        self, carry: tuple[VBuffer, Array, Array, Array, Array], action: Array, key: Array
    ) -> tuple[tuple[VBuffer, Array, Array, Array, Array], tuple[Array, dict[str, Array]]]:
        """One policy step.

        Clips the action, integrates ``N_SUBSTEPS`` simulation steps, scores the result, then
        auto-resets the worlds that crashed, tilted past the limit, or timed out.

        Args:
            carry: ``(vbuf, prev_action, t, episode_return, sim_key)``.
            action: Raw policy output, shape ``(n_worlds, ACT_DIM)``.
            key: PRNG key used to randomise the worlds that reset this step.

        Returns:
            The new carry and ``(next_observation, info)``.
        """
        vbuf, prev_action, t, ep_return, sim_key = carry
        data = vbuf.to_sim_data(self._template)

        a = jnp.clip(
            jnp.broadcast_to(action, (self.n_worlds, self.act_dim)), ACT_LOW, ACT_HIGH
        )
        if self.control_mode == Control.state:
            data = F.state_control(
                data, action_to_full_state(data, a, vbuf.anchor_yaw)[:, None, :]
            )
        else:
            data = F.attitude_control(data, a[:, None, :])
        data = self._step_fn(data, N_SUBSTEPS)

        reward, info = self.reward(data, a, prev_action, t, vbuf.anchor_yaw)
        pos, quat = data.states.pos[:, 0, :], data.states.quat[:, 0, :]
        failed = self._failed(pos, quat, t)
        timeout = (t + 1) >= self.episode_steps
        done = failed | timeout

        ep_return = ep_return + reward
        info = {
            **info,
            "reward": reward,
            "done": done,
            "failed": failed,
            "timeout": timeout,
            "ep_return": jnp.where(done, ep_return, 0.0),
        }

        # Reset done worlds to a freshly sampled start state and blend it in.
        sim_key, subkey = jax.random.split(sim_key)
        reset_vbuf, reset_obs, _ = self.reset_buffer(subkey)
        reset_data = reset_vbuf.to_sim_data(self._template)
        data = _select(reset_data, data, done)
        anchor_yaw = _select(reset_vbuf.anchor_yaw, vbuf.anchor_yaw, done)
        vbuf = VBuffer.from_sim_data(data, anchor_yaw)

        obs = _select(reset_obs, self.observation(data, a, anchor_yaw), done)
        prev_action = jnp.where(done[:, None], 0.0, a)
        t = jnp.where(done, 0, t + 1)
        ep_return = jnp.where(done, 0.0, ep_return)
        return (vbuf, prev_action, t, ep_return, sim_key), (obs, info)

    # -- rollout -------------------------------------------------------------------------------

    def rollout(
        self,
        policy: Callable[[Any, Array], tuple[Array, Any]],
        params: Any,
        vbuf: VBuffer,
        key: Array,
    ) -> tuple[Any, VBuffer, dict[str, Array]]:
        """Collect ``episode_steps`` policy steps with ``lax.scan``.

        Args:
            policy: ``(params, obs) -> (action, new_params)``. ``params`` is threaded through so a
                stateful policy works; a stateless one returns them unchanged.
            params: Initial policy parameters.
            vbuf: Starting simulation buffer.
            key: PRNG key.

        Returns:
            Final params, final buffer, and per-step statistics with a leading time axis.
        """
        prev_action = jnp.zeros((self.n_worlds, ACT_DIM))
        t0 = jnp.zeros((self.n_worlds,), dtype=jnp.int32)
        r0 = jnp.zeros((self.n_worlds,))

        def scan_fn(carry, _):
            params_, vbuf_, prev_a, t_, ep_ret, key_ = carry
            obs_ = self.observation(
                vbuf_.to_sim_data(self._template), prev_a, vbuf_.anchor_yaw
            )
            action, params_ = policy(params_, obs_)
            key_, subkey = jax.random.split(key_)
            (vbuf_, prev_a, t_, ep_ret, _), (_, info) = self.step(
                (vbuf_, prev_a, t_, ep_ret, key_), action, subkey
            )
            return (params_, vbuf_, prev_a, t_, ep_ret, key_), (obs_, action, info)

        carry = (params, vbuf, prev_action, t0, r0, key)
        (params, vbuf, prev_action, _, _, _), (obs, actions, info) = jax.lax.scan(
            scan_fn, carry, None, length=self.episode_steps
        )
        # Observation of the state after the last action, needed to bootstrap the final value.
        obs_final = self.observation(
            vbuf.to_sim_data(self._template), prev_action, vbuf.anchor_yaw
        )
        return params, vbuf, {"obs": obs, "obs_final": obs_final, "action": actions, **info}



