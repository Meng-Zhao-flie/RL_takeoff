# PPO: complete takeoff to 50 cm and five-second hover

This folder contains a standalone training setup for the `cf2x_L250` **takeoff and hover** task.
PPO controls the aircraft from the first lift-off command through the full five-second hold. A safe
landing controller remains separate; this task does not train a touchdown maneuver. The simulator
resets the zero-rotor aircraft at 2 cm above the floor to avoid a contact-clipping artifact, so this
run does not model the first floor-contact centimeters or ground effect.

## Run

From `E:\Crazyflie\crazyflow\crazyflow` in the `crazyflie` Conda environment:

```powershell
python rl_takeoff_full_mission\train_ppo.py --iterations 300
```

The default run warm-starts from the earlier 30 cm PPO actor, then fine-tunes against the 50 cm,
eight-second objective. Use `--init-checkpoint` with another compatible PPO checkpoint, or add
`--from-scratch` to initialize new weights. The best
and latest Flax checkpoints and JSON metrics are written to
`saves\cf2x_L250_ppo_full_mission_50cm_5s`.

The batch defaults to 128 parallel Crazyflow worlds, 400 policy steps per episode at 50 Hz, four PPO
epochs, and 2,048-sample minibatches. The final 250 steps are exactly five simulated seconds.

## Selected sim-to-real checkpoint

The current candidate is
`saves\cf2x_L250_ppo_takeoff_hover_yaw_robust_v3\best.msgpack`. It was warm-started from the prior
horizontal-refinement actor, then trained for 200 iterations with randomized takeoff heading,
stronger height/vertical-speed and position rewards, and deployment-matched 12-degree roll/pitch
limits. The trainer keeps the best evaluation checkpoint instead of assuming the last iteration is
best. To reproduce it:

```powershell
python rl_takeoff_full_mission\train_ppo.py `
  --iterations 200 --eval-interval 10 --num-envs 128 --epochs 4 --minibatch-size 2048 `
  --learning-rate 0.0001 --seed 67 `
  --init-checkpoint saves\cf2x_L250_ppo_full_mission_50cm_5s_xy_refine\best.msgpack `
  --output saves\cf2x_L250_ppo_takeoff_hover_yaw_robust_v3
```

The independent 512-world randomized validation passed 455/512 missions (88.9%). The full final
five-second window stayed within ±3 cm height in 89.5% of worlds and within a 5 cm XY radius in
91.6%; all worlds passed speed and tilt gates, and none failed the simulator safety limits. Mean
tail height was 0.4996 m; the worst transient was 0.5995 m, below the 0.60 m training ceiling. The
single-seed MuJoCo replay finished at 0.504 m mean height over the final five seconds. These remain
simulation results, not a real-flight success rate.

## Design

- **Physics and interface:** Crazyflow `first_principles` dynamics with `Control.state`. PPO emits
  roll, pitch, yaw, and collective thrust; the Mellinger loop runs at 500 Hz while PPO updates at
  50 Hz. Plant mass is centered on 32.4 g and the simulated Mellinger mass is also 32.4 g to match
  the OOT firmware.
- **Domain randomization:** each world independently varies mass by ±10%, thrust efficiency by
  ±10%, and rotor response coefficients by ±15%. Initial height, linear velocity, and angular
  velocity are randomized too. This trains one actor against a range of plausible aircraft and
  launch conditions instead of a single nominal simulation.
- **Observation (18 values):** position, quaternion, and horizontal velocity are expressed in the
  takeoff-anchor frame; height error, body angular rates, previous action, and target token are also
  included. This makes the policy independent of the vehicle's starting yaw.
- **Action (4 values):** tanh-squashed roll and pitch within ±12 degrees, fixed zero relative yaw,
  and collective thrust mapped to 0.051–0.48 N. The observation includes the previous action so the
  reward can favor smooth commands.
- **Reward:** every step rewards height accuracy and penalizes horizontal drift, speed, vertical
  speed, overshoot, tilt, spin, effort, and command changes. Crashes, excessive height, and excessive
  tilt receive additional penalties. A terminal bonus requires the full hover gate.
- **PPO update:** a 128×128 actor and critic MLP, Gaussian exploration during rollout, clipped PPO
  objective (`clip=0.2`), GAE (`gamma=0.99`, `lambda=0.95`), Adam (`3e-4`), four epochs, and gradient
  clipping at 0.5. Inference uses the actor mean, so the deployed policy is deterministic.
- **Success gate:** the entire final five-second window must stay within 3 cm of 50 cm, below
  0.15 m/s total speed, inside 5 cm XY radius, below 10 degrees tilt, and without a failure event.
  Evaluation runs 128 randomized worlds at each checkpoint interval.

## Replay in MuJoCo

After training, replay the deterministic actor in Crazyflow's viewer:

```powershell
python rl_takeoff_full_mission\view_policy.py `
  --checkpoint saves\cf2x_L250_ppo_takeoff_hover_yaw_robust_v3\best.msgpack
```

Simulation success is a gate before hardware deployment, not proof of real-flight reliability.
Export the final checkpoint to the OOT C header only after reviewing its evaluation metrics and
MuJoCo replay.
