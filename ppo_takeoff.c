/* On-board Crazyflow PPO takeoff policy using the native Mellinger controller. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"
#include "commander.h"
#include "controller_mellinger.h"
#include "param.h"
#include "stabilizer_types.h"
#include "usec_time.h"

#define DEBUG_MODULE "PPO"
#include "debug.h"

#include "ppo_policy_weights.h"

#define PI_F 3.14159265358979323846f
#define DEG_TO_RAD (PI_F / 180.0f)
#define POLICY_TICKS 20u       /* stabilizer callback is 1000 Hz */
#define MAX_MISSION_TICKS 18000u /* hard timeout; policy may need time to climb and settle */
#define TARGET_CONFIRM_TICKS 500u /* require 0.5 s continuously stable at target */
#define HOVER_HOLD_TICKS 5000u    /* five full seconds after target confirmation */
#define PPO_TARGET_Z 0.50f
#define AIRFRAME_MASS_KG 0.0324f
#define LAND_Z 0.03f
#define LAND_TICKS 2500u
#define MAX_XY 0.25f
#define MIN_Z -0.03f
#define MAX_Z 0.65f

typedef enum { PHASE_IDLE, PHASE_POLICY, PHASE_LANDING, PHASE_LANDED } ppo_phase_t;

static uint8_t g_start = 0;
static uint8_t g_abort = 0;
static uint8_t g_status = 0; /* 0 idle, 1 ready, 2 PPO mission, 3 landing, 4 landed, 5 rejected */
static uint32_t g_policy_us_last = 0;
static uint32_t g_policy_us_max = 0;
static uint32_t g_hover_elapsed_ms = 0;

PARAM_GROUP_START(ppo)
PARAM_ADD(PARAM_UINT8, start, &g_start)
PARAM_ADD(PARAM_UINT8, abort, &g_abort)
PARAM_ADD(PARAM_UINT8, status, &g_status)
PARAM_ADD(PARAM_UINT32, policy_us, &g_policy_us_last)
PARAM_ADD(PARAM_UINT32, policy_us_max, &g_policy_us_max)
PARAM_ADD(PARAM_UINT32, hover_ms, &g_hover_elapsed_ms)
PARAM_GROUP_STOP(ppo)

static controllerMellinger_t g_mellinger;
static volatile ppo_phase_t g_phase = PHASE_IDLE;
static float g_anchor_x, g_anchor_y, g_anchor_z;
static float g_anchor_yaw, g_anchor_yaw_cos, g_anchor_yaw_sin;
static float g_anchor_yaw_half_cos, g_anchor_yaw_half_sin;
static float g_prev_action[4];
static float g_command_pos[3], g_command_vel[3], g_command_acc[3];
static float g_command_yaw_deg;
static uint32_t g_flight_start_tick, g_last_policy_tick, g_land_start_tick;
static uint32_t g_target_stable_since_tick, g_hover_start_tick;
static float g_land_x, g_land_y, g_land_z, g_land_yaw_deg;

static inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static bool estimatorValid(const state_t *state, const sensorData_t *sensors) {
  const float qx = state->attitudeQuaternion.x;
  const float qy = state->attitudeQuaternion.y;
  const float qz = state->attitudeQuaternion.z;
  const float qw = state->attitudeQuaternion.w;
  const float qnorm = sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
  return isfinite(state->position.x) && isfinite(state->position.y) &&
         isfinite(state->position.z) && isfinite(state->velocity.x) &&
         isfinite(state->velocity.y) && isfinite(state->velocity.z) &&
         isfinite(sensors->gyro.x) && isfinite(sensors->gyro.y) &&
         isfinite(sensors->gyro.z) && isfinite(qnorm) && qnorm >= 0.8f && qnorm <= 1.2f;
}

static void beginLanding(const state_t *state, uint32_t tick) {
  if (g_phase == PHASE_LANDING || g_phase == PHASE_LANDED) return;
  g_phase = PHASE_LANDING;
  g_status = 3;
  g_land_start_tick = tick;
  g_land_x = state->position.x;
  g_land_y = state->position.y;
  g_land_z = state->position.z;
  g_land_yaw_deg = state->attitude.yaw;
  DEBUG_PRINT("Landing from z=%.2f\n", (double)(g_land_z - g_anchor_z));
}

static void runPolicy(const state_t *state, const sensorData_t *sensors, uint32_t tick) {
  const float dx = state->position.x - g_anchor_x;
  const float dy = state->position.y - g_anchor_y;
  const float x = g_anchor_yaw_cos * dx + g_anchor_yaw_sin * dy;
  const float y = -g_anchor_yaw_sin * dx + g_anchor_yaw_cos * dy;
  const float z = state->position.z - g_anchor_z;
  if (fabsf(x) > MAX_XY || fabsf(y) > MAX_XY || z < MIN_Z || z > MAX_Z) {
    DEBUG_PRINT("Policy geofence: xy=(%.2f,%.2f) z=%.2f\n", (double)x, (double)y, (double)z);
    beginLanding(state, tick);
    return;
  }

  if ((uint32_t)(tick - g_last_policy_tick) >= POLICY_TICKS) {
    g_last_policy_tick = tick;
    float obs[PPO_OBS_DIM];
    int k = 0;
    obs[k++] = x;
    obs[k++] = y;
    obs[k++] = (z - PPO_TARGET_Z) * 2.0f;
    const float qnorm = sqrtf(state->attitudeQuaternion.x * state->attitudeQuaternion.x +
                              state->attitudeQuaternion.y * state->attitudeQuaternion.y +
                              state->attitudeQuaternion.z * state->attitudeQuaternion.z +
                              state->attitudeQuaternion.w * state->attitudeQuaternion.w);
    const float qx = state->attitudeQuaternion.x / qnorm;
    const float qy = state->attitudeQuaternion.y / qnorm;
    const float qz = state->attitudeQuaternion.z / qnorm;
    const float qw = state->attitudeQuaternion.w / qnorm;
    /* Quaternion rotation uses half-angles; x/y positions and velocities use full yaw. */
    obs[k++] = g_anchor_yaw_half_cos * qx + g_anchor_yaw_half_sin * qy;
    obs[k++] = g_anchor_yaw_half_cos * qy - g_anchor_yaw_half_sin * qx;
    obs[k++] = g_anchor_yaw_half_cos * qz - g_anchor_yaw_half_sin * qw;
    obs[k++] = g_anchor_yaw_half_cos * qw + g_anchor_yaw_half_sin * qz;
    obs[k++] = (g_anchor_yaw_cos * state->velocity.x +
                g_anchor_yaw_sin * state->velocity.y) * 1.5f;
    obs[k++] = (-g_anchor_yaw_sin * state->velocity.x +
                g_anchor_yaw_cos * state->velocity.y) * 1.5f;
    obs[k++] = state->velocity.z * 1.5f;
    obs[k++] = sensors->gyro.x * DEG_TO_RAD * 5.0f;
    obs[k++] = sensors->gyro.y * DEG_TO_RAD * 5.0f;
    obs[k++] = sensors->gyro.z * DEG_TO_RAD * 5.0f;
    for (int i = 0; i < 4; ++i) obs[k++] = g_prev_action[i];
    obs[k++] = 0.5f;

    float action[4];
    const uint64_t policy_start_us = usecTimestamp();
    crazyflowPpoPolicy(obs, action);
    const uint64_t policy_elapsed_us = usecTimestamp() - policy_start_us;
    g_policy_us_last = (uint32_t)policy_elapsed_us;
    if (g_policy_us_last > g_policy_us_max) g_policy_us_max = g_policy_us_last;
    memcpy(g_prev_action, action, sizeof(g_prev_action));

    /* Match the simulator adapter: the current position/velocity are references, while this
       acceleration encodes the action's desired force vector using Mellinger's configured mass. */
    const float roll = action[0], pitch = action[1], yaw = action[2], thrust = action[3];
    const float cr = cosf(roll), sr = sinf(roll);
    const float cp = cosf(pitch), sp = sinf(pitch);
    const float cy = cosf(yaw), sy = sinf(yaw);
    const float body_z_x_local = cy * sp * cr + sy * sr;
    const float body_z_y_local = sy * sp * cr - cy * sr;
    const float body_z_x = g_anchor_yaw_cos * body_z_x_local -
                           g_anchor_yaw_sin * body_z_y_local;
    const float body_z_y = g_anchor_yaw_sin * body_z_x_local +
                           g_anchor_yaw_cos * body_z_y_local;
    const float body_z_z = cp * cr;
    const float controller_mass = g_mellinger.mass;
    g_command_pos[0] = state->position.x;
    g_command_pos[1] = state->position.y;
    g_command_pos[2] = state->position.z;
    g_command_vel[0] = state->velocity.x;
    g_command_vel[1] = state->velocity.y;
    g_command_vel[2] = state->velocity.z;
    g_command_acc[0] = thrust * body_z_x / controller_mass;
    g_command_acc[1] = thrust * body_z_y / controller_mass;
    g_command_acc[2] = thrust * body_z_z / controller_mass - 9.81f;
    g_command_yaw_deg = (g_anchor_yaw + yaw) / DEG_TO_RAD;
  }
}

static bool targetHoldGate(const state_t *state, bool settling) {
  const float dx = state->position.x - g_anchor_x;
  const float dy = state->position.y - g_anchor_y;
  const float x = g_anchor_yaw_cos * dx + g_anchor_yaw_sin * dy;
  const float y = -g_anchor_yaw_sin * dx + g_anchor_yaw_cos * dy;
  const float z = state->position.z - g_anchor_z;
  const float speed2 = state->velocity.x * state->velocity.x +
                       state->velocity.y * state->velocity.y +
                       state->velocity.z * state->velocity.z;
  const float xy_limit = settling ? 0.10f : 0.15f;
  const float speed_limit = settling ? 0.15f : 0.20f;
  const float tilt_limit_deg = settling ? 10.0f : 12.0f;
  return fabsf(z - PPO_TARGET_Z) < 0.04f &&
         x * x + y * y < xy_limit * xy_limit &&
         speed2 < speed_limit * speed_limit &&
         fabsf(state->attitude.roll) < tilt_limit_deg &&
         fabsf(state->attitude.pitch) < tilt_limit_deg;
}

static bool updateHoverTimer(const state_t *state, uint32_t tick) {
  if (!targetHoldGate(state, g_hover_start_tick == 0)) {
    g_target_stable_since_tick = 0;
    g_hover_start_tick = 0;
    g_hover_elapsed_ms = 0;
    return false;
  }

  if (g_hover_start_tick == 0) {
    if (g_target_stable_since_tick == 0) g_target_stable_since_tick = tick;
    if ((uint32_t)(tick - g_target_stable_since_tick) >= TARGET_CONFIRM_TICKS) {
      g_hover_start_tick = tick;
      g_hover_elapsed_ms = 0;
      DEBUG_PRINT("Target stable; starting 5 s hover timer\n");
    }
    return false;
  }

  g_hover_elapsed_ms = (uint32_t)(tick - g_hover_start_tick);
  return g_hover_elapsed_ms >= HOVER_HOLD_TICKS;
}

static void makePolicySetpoint(setpoint_t *sp) {
  memset(sp, 0, sizeof(*sp));
  sp->mode.x = modeAbs;
  sp->mode.y = modeAbs;
  sp->mode.z = modeAbs;
  sp->mode.yaw = modeAbs;
  sp->position.x = g_command_pos[0];
  sp->position.y = g_command_pos[1];
  sp->position.z = g_command_pos[2];
  sp->velocity.x = g_command_vel[0];
  sp->velocity.y = g_command_vel[1];
  sp->velocity.z = g_command_vel[2];
  sp->acceleration.x = g_command_acc[0];
  sp->acceleration.y = g_command_acc[1];
  sp->acceleration.z = g_command_acc[2];
  sp->attitude.yaw = g_command_yaw_deg;
}

static void runLanding(const state_t *state, uint32_t tick, setpoint_t *sp) {
  float u = clampf((float)(tick - g_land_start_tick) / (float)LAND_TICKS, 0.0f, 1.0f);
  float u2 = u * u, u3 = u2 * u, u4 = u3 * u, u5 = u4 * u;
  float smooth = 10.0f * u3 - 15.0f * u4 + 6.0f * u5;
  float ds = 30.0f * u2 * (u - 1.0f) * (u - 1.0f);
  float dds = 60.0f * u * (2.0f * u - 1.0f) * (u - 1.0f);
  float dz = (g_anchor_z + LAND_Z) - g_land_z;
  memset(sp, 0, sizeof(*sp));
  sp->mode.x = modeAbs;
  sp->mode.y = modeAbs;
  sp->mode.z = modeAbs;
  sp->mode.yaw = modeAbs;
  sp->position.x = g_land_x;
  sp->position.y = g_land_y;
  sp->position.z = g_land_z + dz * smooth;
  sp->velocity.z = dz * ds / (LAND_TICKS / 1000.0f);
  sp->acceleration.z = dz * dds / ((LAND_TICKS / 1000.0f) * (LAND_TICKS / 1000.0f));
  sp->attitude.yaw = g_land_yaw_deg;
  if (u >= 1.0f && state->position.z <= g_anchor_z + 0.075f && fabsf(state->velocity.z) < 0.18f) {
    g_phase = PHASE_LANDED;
    g_status = 4;
    DEBUG_PRINT("Landed at z=%.2f\n", (double)(state->position.z - g_anchor_z));
  }
}

void appMain(void) {
  vTaskDelay(M2T(3000));
  /* OOT=6 is required: this callback wraps the firmware's native Mellinger controller. */
  paramSetInt(paramGetVarId("stabilizer", "controller"), 6);
  g_status = 1;
  DEBUG_PRINT("PPO ready; set ppo.start=1; full-mission target %.2f m, hold 5 s\n",
              (double)PPO_TARGET_Z);
  while (1) vTaskDelay(M2T(1000));
}

void controllerOutOfTreeInit(void) {
  controllerMellingerInit(&g_mellinger);
  /* The measured CF2x_L250 vehicle mass is 32.4 g; firmware default is 29 g. */
  g_mellinger.mass = AIRFRAME_MASS_KG;
  g_phase = PHASE_IDLE;
  g_status = 0;
  memset(g_prev_action, 0, sizeof(g_prev_action));
  memset(g_command_pos, 0, sizeof(g_command_pos));
  memset(g_command_vel, 0, sizeof(g_command_vel));
  memset(g_command_acc, 0, sizeof(g_command_acc));
}

bool controllerOutOfTreeTest(void) { return true; }

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                          const sensorData_t *sensors, const state_t *state,
                          const stabilizerStep_t tick) {
  if (g_start && (g_phase == PHASE_IDLE || g_phase == PHASE_LANDED)) {
    g_start = 0;
    g_abort = 0;
    if (!estimatorValid(state, sensors)) {
      g_status = 5;
      DEBUG_PRINT("Start rejected: invalid position/velocity/quaternion/gyro estimate\n");
      controllerMellinger(&g_mellinger, control, setpoint, sensors, state, tick);
      return;
    }
    g_phase = PHASE_POLICY;
    g_status = 2;
    g_anchor_x = state->position.x;
    g_anchor_y = state->position.y;
    g_anchor_z = state->position.z;
    const float qx = state->attitudeQuaternion.x;
    const float qy = state->attitudeQuaternion.y;
    const float qz = state->attitudeQuaternion.z;
    const float qw = state->attitudeQuaternion.w;
    g_anchor_yaw = atan2f(2.0f * (qw * qz + qx * qy),
                          1.0f - 2.0f * (qy * qy + qz * qz));
    g_anchor_yaw_cos = cosf(g_anchor_yaw);
    g_anchor_yaw_sin = sinf(g_anchor_yaw);
    g_anchor_yaw_half_cos = cosf(0.5f * g_anchor_yaw);
    g_anchor_yaw_half_sin = sinf(0.5f * g_anchor_yaw);
    g_flight_start_tick = tick;
    g_last_policy_tick = tick - POLICY_TICKS;
    g_target_stable_since_tick = 0;
    g_hover_start_tick = 0;
    g_hover_elapsed_ms = 0;
    g_policy_us_last = 0;
    g_policy_us_max = 0;
    memset(g_prev_action, 0, sizeof(g_prev_action));
    DEBUG_PRINT("PPO start anchor=(%.3f,%.3f,%.3f)\n", (double)g_anchor_x,
                (double)g_anchor_y, (double)g_anchor_z);
  }

  if (g_phase == PHASE_IDLE) {
    controllerMellinger(&g_mellinger, control, setpoint, sensors, state, tick);
    return;
  }

  if (g_phase == PHASE_POLICY) {
    if (g_abort) {
      g_abort = 0;
      beginLanding(state, tick);
    } else if ((uint32_t)(tick - g_flight_start_tick) >= MAX_MISSION_TICKS) {
      DEBUG_PRINT("PPO mission timeout before completed hover; controlled landing\n");
      beginLanding(state, tick);
    } else {
      runPolicy(state, sensors, tick);
      if (g_phase == PHASE_POLICY && updateHoverTimer(state, tick)) {
        DEBUG_PRINT("5 s stable hover complete; controlled landing\n");
        beginLanding(state, tick);
      }
    }
  }

  setpoint_t policySetpoint;
  if (g_phase == PHASE_POLICY) {
    makePolicySetpoint(&policySetpoint);
  } else if (g_phase == PHASE_LANDING) {
    runLanding(state, tick, &policySetpoint);
  } else if (g_phase == PHASE_LANDED) {
    memset(control, 0, sizeof(*control));
    control->controlMode = controlModeLegacy;
    return;
  }
  controllerMellinger(&g_mellinger, control, &policySetpoint, sensors, state, tick);
}
