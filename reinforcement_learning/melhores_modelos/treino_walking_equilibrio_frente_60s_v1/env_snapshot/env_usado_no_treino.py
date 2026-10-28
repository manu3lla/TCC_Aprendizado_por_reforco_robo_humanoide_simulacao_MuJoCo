import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer


class AtomEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        debug_reward=False,
        use_reference_gait=False,
        gait_stand_mode=False,
        gait_walk_mode=False,
    ):

        super().__init__()

        self.render_mode = render_mode
        self.viewer = None

        # Carrega o modelo MuJoCo
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "model",
            "scene.xml"
        )
        model_path = os.path.abspath(model_path)

        print("Carregando modelo:", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.debug_reward = debug_reward
        self.gait_stand_mode = gait_stand_mode
        self.gait_walk_mode = gait_walk_mode

        # Espaço de observação: posições, velocidades, fase da marcha,
        # contato/altura dos pes e direcao ate o alvo.
        self.extra_obs_dim = 9
        obs_dim = self.model.nq + self.model.nv + self.extra_obs_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Espaço de ação: uma ação para cada atuador
        act_dim = self.model.nu

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

        # Estado inicial
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

        # Controle inicial dos atuadores
        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float32)

        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.default_ctrl[i] = self.initial_qpos[qpos_adr]

        self.actuator_index = {
            self.model.actuator(i).name: i for i in range(self.model.nu)
        }
        self.joint_qposadr = {
            self.model.joint(i).name: int(self.model.jnt_qposadr[i])
            for i in range(self.model.njnt)
            if self.model.joint(i).name
        }
        self.joint_dofadr = {
            self.model.joint(i).name: int(self.model.jnt_dofadr[i])
            for i in range(self.model.njnt)
            if self.model.joint(i).name
        }

        # A fase da marcha entra na observacao. Quando use_reference_gait=True,
        # a politica aprende correcoes residuais em cima de uma marcha inicial.
        self.use_reference_gait = (
            use_reference_gait or gait_stand_mode or gait_walk_mode
        )
        self.gait_period_steps = 76
        self.gait_phase_offset = 0.0

        # Escala residual do PPO. A marcha base ja movimenta as pernas; estes
        # valores limitam o quanto a rede pode corrigir cada junta.
        self.action_scale = np.ones(self.model.nu, dtype=np.float32) * 0.12

        if self.model.nu >= 20:
            self.action_scale[0] = 0.02  # head_pan
            self.action_scale[1] = 0.02  # head_tilt

            self.action_scale[2:8] = 0.06

            self.action_scale[8] = 0.08
            self.action_scale[9] = 0.12
            self.action_scale[10] = 0.20
            self.action_scale[11] = 0.38
            self.action_scale[12] = 0.20
            self.action_scale[13] = 0.10

            self.action_scale[14] = 0.08
            self.action_scale[15] = 0.12
            self.action_scale[16] = 0.20
            self.action_scale[17] = 0.38
            self.action_scale[18] = 0.20
            self.action_scale[19] = 0.10

        if self.gait_stand_mode and self.model.nu >= 20:
            self.action_scale[2:8] *= 0.50
            self.action_scale[8:20] *= 0.35
        elif self.gait_walk_mode and self.model.nu >= 20:
            self.action_scale[2:8] *= 0.60
            self.action_scale[8:20] *= 0.55
        elif self.use_reference_gait and self.model.nu >= 20:
            self.action_scale[2:8] *= 0.50
            self.action_scale[8:20] *= 0.35

        self.gait_amplitude = {
            "arm_pitch": 0.000,
            "hip_lift": -0.550,
            "knee_crouch": 0.180,
            "knee_swing": 0.950,
            "foot_lift": -0.350,
            "ankle_crouch": -0.200,
        }
        if self.gait_walk_mode:
            self.gait_amplitude["hip_lift"] = -0.550
        self.reference_gait_strength = 0.85
        self.gait_double_support = 0.14
        self.gait_warmup_steps = 60

        self.frame_skip = 5
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.max_episode_steps = 1000
        if self.gait_walk_mode:
            self.max_episode_steps = int(np.ceil(60.0 / self.dt))

        self.prev_x = 0.0
        self.current_step = 0
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_ctrl = self.default_ctrl.copy()
        self.prev_distance = 0.0
        self.ctrl_blend = 0.72 if self.gait_stand_mode else 0.66 if self.gait_walk_mode else 0.62
        self.prev_left_contact = False
        self.prev_right_contact = False
        self.prev_left_foot_pos = np.zeros(3, dtype=np.float64)
        self.prev_right_foot_pos = np.zeros(3, dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)

        # pegar a posição x do target
        target_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target"
        )

        self.target_pos = self.data.xpos[target_id].copy()
        print("target pos:" + str(self.target_pos))
        self.initial_xy = self.initial_qpos[:2].astype(np.float32).copy()
        target_delta_xy = self.target_pos[:2] - self.initial_xy
        target_delta_norm = float(np.linalg.norm(target_delta_xy))
        if target_delta_norm > 1e-6:
            self.target_line_dir = (target_delta_xy / target_delta_norm).astype(
                np.float32
            )
        else:
            self.target_line_dir = np.array([-1.0, 0.0], dtype=np.float32)
        self.target_line_normal = np.array(
            [-self.target_line_dir[1], self.target_line_dir[0]],
            dtype=np.float32,
        )

        self.chest_body_id = self._body_id("chest")
        self.left_foot_body_id = self._body_id("left_foot")
        self.right_foot_body_id = self._body_id("right_foot")
        self.initial_torso_height = self._torso_height()
        self.initial_left_foot_z = float(self.data.xpos[self.left_foot_body_id][2])
        self.initial_right_foot_z = float(self.data.xpos[self.right_foot_body_id][2])
        self.initial_left_foot_rot = (
            self.data.xmat[self.left_foot_body_id].reshape(3, 3).copy()
        )
        self.initial_right_foot_rot = (
            self.data.xmat[self.right_foot_body_id].reshape(3, 3).copy()
        )
        self.initial_foot_spacing_y = abs(
            float(
                self.data.xpos[self.right_foot_body_id][1]
                - self.data.xpos[self.left_foot_body_id][1]
            )
        )
        initial_right_left_y = float(
            self.data.xpos[self.right_foot_body_id][1]
            - self.data.xpos[self.left_foot_body_id][1]
        )
        self.initial_foot_side_sign = float(np.sign(initial_right_left_y) or 1.0)
        initial_foot_info = self._foot_info()
        self.prev_left_contact = initial_foot_info["left_contact"]
        self.prev_right_contact = initial_foot_info["right_contact"]
        self.prev_left_foot_pos = initial_foot_info["left_pos"].copy()
        self.prev_right_foot_pos = initial_foot_info["right_pos"].copy()
        # Permite uma postura mais agachada antes de considerar queda.
        self.healthy_min_height = 0.23 if self.use_reference_gait else 0.30
        self.healthy_min_upright = 0.30 if self.use_reference_gait else 0.50
        self.desired_crouch = 0.060
        self.allowed_crouch = 0.140
        self.desired_crouch_knee = 0.130
        self.max_forward_lean = 0.44
        self.target_radius = 0.30
        self.desired_walk_speed = 0.185
        self.desired_step_length = 0.120
        if self.gait_walk_mode:
            self.desired_walk_speed = 0.145
            self.desired_step_length = 0.080

        self.reward_weights = {
            "target_velocity": 5.20,
            "distance_progress": 62.00,
            "alive": 0.005,
            "height": 0.08,
            "soft_crouch": 0.10,
            "upright": 0.46,
            "low_height": 5.00,
            "forward_lean": 24.00,
            "unstable_forward_lean": 105.00,
            "moving_forward_lean": 7.50,
            "stride_forward_lean": 5.50,
            "body_projection": 90.00,
            "moving_body_projection": 9.00,
            "lateral_drift": 1.25,
            "path_drift": 7.50,
            "lateral_velocity": 0.42,
            "crooked_velocity": 0.85,
            "target_alignment": 0.70,
            "stationary_xy": 0.00,
            "stationary_velocity": 0.00,
            "reverse_velocity": 0.00,
            "reverse_progress": 0.00,
            "reverse_drift": 0.00,
            "control": 0.014,
            "smooth_action": 0.018,
            "smooth_ctrl": 0.010,
            "upper_body_motion": 0.030,
            "arm_swing": 0.00,
            "arm_balance": 0.00,
            "arm_flail": 0.00,
            "reference_gait": 0.28,
            "gait_pose": 2.40,
            "gait_forward": 0.00,
            "gait_pose_error": 0.35,
            "stance_foot_flat": 0.75,
            "foot_support": 0.12,
            "single_leg_support": 0.24,
            "one_foot_walk": 0.18,
            "stance_stability": 0.18,
            "gait_timing": 0.00,
            "foot_clearance": 0.38,
            "swing_lift": 0.62,
            "missing_swing_lift": 0.78,
            "foot_spacing": 0.04,
            "foot_crossing": 0.25,
            "walk_speed": 1.05,
            "safe_speed": 0.75,
            "swing_foot_forward": 0.42,
            "step_length": 0.70,
            "human_stride": 0.86,
            "stride_progress": 1.65,
            "short_step": 0.95,
            "footstep_landing": 0.36,
            "step_transition": 0.54,
            "fluid_step": 0.36,
            "step_commit": 0.12,
            "foot_slip": 0.76,
            "standstill": 1.25,
            "slow_velocity": 1.55,
            "step_order": 0.00,
            "alternating_knees": 0.18,
            "swing_knee_flex": 0.62,
            "missing_swing_knee": 0.46,
            "stance_knee_straight": 0.03,
            "knee_clearance_pair": 0.46,
            "knee_velocity": 0.22,
            "ankle_knee_coordination": 0.16,
            "double_support": 0.08,
            "double_knee_flex": 0.05,
            "double_foot_lift": 1.30,
            "vertical_motion": 0.08,
            "fall": 340.0,
            "target_reached": 300.0,
        }

        if self.gait_stand_mode:
            self.reward_weights.update(
                {
                    "target_velocity": 0.00,
                    "distance_progress": 0.00,
                    "target_alignment": 0.00,
                    "crooked_velocity": 0.00,
                    "walk_speed": 0.00,
                    "safe_speed": 0.00,
                    "swing_foot_forward": 0.00,
                    "step_length": 0.00,
                    "human_stride": 0.00,
                    "stride_progress": 0.00,
                    "stride_forward_lean": 1.50,
                    "short_step": 0.00,
                    "footstep_landing": 0.00,
                    "step_commit": 0.00,
                    "standstill": 0.00,
                    "slow_velocity": 0.00,
                    "reference_gait": 0.45,
                    "gait_pose": 5.00,
                    "gait_pose_error": 0.90,
                    "stance_foot_flat": 2.00,
                    "soft_crouch": 0.22,
                    "foot_support": 0.30,
                    "single_leg_support": 0.16,
                    "one_foot_walk": 0.12,
                    "foot_clearance": 0.22,
                    "swing_lift": 0.55,
                    "missing_swing_lift": 0.25,
                    "foot_slip": 1.10,
                    "stationary_xy": 22.00,
                    "stationary_velocity": 0.55,
                    "reverse_velocity": 0.00,
                    "reverse_progress": 0.00,
                    "reverse_drift": 0.00,
                    "control": 0.010,
                    "smooth_action": 0.012,
                    "smooth_ctrl": 0.006,
                    "fall": 260.0,
                    "target_reached": 0.0,
                }
            )
        elif self.gait_walk_mode:
            self.reward_weights.update(
                {
                    "target_velocity": 6.15,
                    "distance_progress": 62.00,
                    "target_alignment": 1.45,
                    "crooked_velocity": 1.70,
                    "lateral_drift": 2.25,
                    "path_drift": 15.00,
                    "walk_speed": 0.90,
                    "safe_speed": 0.95,
                    "swing_foot_forward": 0.24,
                    "step_length": 0.30,
                    "human_stride": 0.32,
                    "stride_progress": 0.45,
                    "stride_forward_lean": 3.20,
                    "forward_lean": 32.00,
                    "unstable_forward_lean": 140.00,
                    "moving_forward_lean": 11.00,
                    "body_projection": 125.00,
                    "moving_body_projection": 14.00,
                    "short_step": 0.25,
                    "footstep_landing": 0.28,
                    "step_transition": 0.68,
                    "fluid_step": 0.52,
                    "step_commit": 0.16,
                    "standstill": 0.55,
                    "slow_velocity": 0.85,
                    "reference_gait": 0.38,
                    "gait_pose": 3.20,
                    "gait_forward": 4.40,
                    "gait_pose_error": 0.80,
                    "stance_foot_flat": 1.90,
                    "soft_crouch": 0.18,
                    "foot_support": 0.26,
                    "single_leg_support": 0.18,
                    "one_foot_walk": 0.14,
                    "stance_stability": 0.28,
                    "foot_clearance": 0.24,
                    "swing_lift": 0.55,
                    "missing_swing_lift": 0.28,
                    "foot_slip": 1.35,
                    "stationary_xy": 0.00,
                    "stationary_velocity": 0.00,
                    "reverse_velocity": 18.00,
                    "reverse_progress": 320.00,
                    "reverse_drift": 95.00,
                    "control": 0.011,
                    "smooth_action": 0.018,
                    "smooth_ctrl": 0.010,
                    "fall": 390.0,
                    "target_reached": 300.0,
                }
            )

        # self.initial_height = float(self.initial_qpos[2])

        self.prev_x = float(self.data.qpos[0])
        self.prev_distance = float(
            np.linalg.norm(self.target_pos[:2] - self.data.qpos[:2])
        )

    def _body_id(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        if body_id < 0:
            raise ValueError(f"Body nao encontrado no modelo MuJoCo: {name}")

        return body_id


    def _get_obs(self):
        phase = self._gait_phase()
        foot_info = self._foot_info()
        target_delta = self.target_pos[:2] - self.data.qpos[:2]

        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            np.array(
                [
                    np.sin(phase),
                    np.cos(phase),
                    foot_info["left_pos"][2] - self.initial_left_foot_z,
                    foot_info["right_pos"][2] - self.initial_right_foot_z,
                    float(foot_info["left_contact"]),
                    float(foot_info["right_contact"]),
                    self._torso_upright(),
                    target_delta[0],
                    target_delta[1],
                ],
                dtype=np.float64,
            ),
        ])

        return obs.astype(np.float32)

    def _torso_height(self):
        return float(self.data.xpos[self.chest_body_id][2])

    def _torso_upright(self):
        chest_rot = self.data.xmat[self.chest_body_id].reshape(3, 3)
        return float(np.clip(chest_rot[2, 2], -1.0, 1.0))

    def _torso_up_axis(self):
        chest_rot = self.data.xmat[self.chest_body_id].reshape(3, 3)
        return chest_rot[:, 2].copy()

    def _body_rotation_error(self, body_id, initial_rot):
        current_rot = self.data.xmat[body_id].reshape(3, 3)
        relative_rot = initial_rot.T @ current_rot
        cosine = (np.trace(relative_rot) - 1.0) / 2.0
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _has_fallen(self):
        return (
            self._torso_height() < self.healthy_min_height
            or self._torso_upright() < self.healthy_min_upright
        )

    def _gait_phase(self):
        phase = (
            2.0
            * np.pi
            * (self.current_step % self.gait_period_steps)
            / self.gait_period_steps
        )
        return phase + self.gait_phase_offset

    def _set_ctrl_offset(self, ctrl, actuator_name, offset):
        idx = self.actuator_index.get(actuator_name)
        if idx is not None:
            ctrl[idx] = self.default_ctrl[idx] + offset

    def _add_ctrl_offset(self, ctrl, actuator_name, offset):
        idx = self.actuator_index.get(actuator_name)
        if idx is not None:
            ctrl[idx] += offset

    @staticmethod
    def _smoothstep(value):
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _ctrl_offset(self, ctrl, actuator_name):
        idx = self.actuator_index.get(actuator_name)

        if idx is None:
            return 0.0

        return float(ctrl[idx] - self.default_ctrl[idx])

    def _joint_qpos(self, joint_name):
        qpos_adr = self.joint_qposadr.get(joint_name)

        if qpos_adr is None:
            return 0.0

        return float(self.data.qpos[qpos_adr])

    def _joint_offset(self, joint_name):
        qpos_adr = self.joint_qposadr.get(joint_name)

        if qpos_adr is None:
            return 0.0

        return float(self.data.qpos[qpos_adr] - self.initial_qpos[qpos_adr])

    def _joint_qvel(self, joint_name):
        dof_adr = self.joint_dofadr.get(joint_name)

        if dof_adr is None:
            return 0.0

        return float(self.data.qvel[dof_adr])

    def _gait_reference_ctrl(self):
        ctrl = self.default_ctrl.copy()

        cycle = (self._gait_phase() / (2.0 * np.pi)) % 1.0
        left_is_swing = cycle < 0.5
        step_phase = (cycle * 2.0) % 1.0
        double_support = self.gait_double_support
        swing_raw = (step_phase - double_support) / max(
            1.0 - 2.0 * double_support,
            1e-6,
        )
        swing_t = self._smoothstep(swing_raw)
        swing_bell = float(np.sin(np.pi * np.clip(swing_raw, 0.0, 1.0)))
        load_bell = float(np.sin(np.pi * step_phase))
        warmup = self._smoothstep(self.current_step / self.gait_warmup_steps)
        strength = self.reference_gait_strength * warmup

        hip_lift = self.gait_amplitude["hip_lift"]
        knee_crouch = self.gait_amplitude["knee_crouch"]
        knee_swing = self.gait_amplitude["knee_swing"]
        foot_lift = self.gait_amplitude["foot_lift"]
        ankle_crouch = self.gait_amplitude["ankle_crouch"]
        arm_amp = self.gait_amplitude["arm_pitch"]

        crouch_gain = 0.55 + 0.45 * load_bell
        swing_knee = knee_crouch + knee_swing * swing_bell
        stance_knee = knee_crouch * crouch_gain
        swing_hip = hip_lift * swing_bell
        swing_foot = ankle_crouch + foot_lift * swing_bell
        stance_foot = ankle_crouch * crouch_gain

        def apply_leg(side, hip_pitch, knee, foot_pitch):
            if side == "left":
                hip_sign = 1.0
                knee_sign = 1.0
                foot_sign = 1.0
            else:
                hip_sign = -1.0
                knee_sign = -1.0
                foot_sign = -1.0

            self._set_ctrl_offset(
                ctrl,
                f"{side}_leg_pitch",
                strength * hip_sign * hip_pitch,
            )
            self._set_ctrl_offset(
                ctrl,
                f"{side}_knee",
                strength * knee_sign * knee,
            )
            self._set_ctrl_offset(
                ctrl,
                f"{side}_foot_pitch",
                strength * foot_sign * foot_pitch,
            )

        if left_is_swing:
            apply_leg("left", swing_hip, swing_knee, swing_foot)
            apply_leg("right", 0.0, stance_knee, stance_foot)
            arm_cycle = 1.0
        else:
            apply_leg("right", swing_hip, swing_knee, swing_foot)
            apply_leg("left", 0.0, stance_knee, stance_foot)
            arm_cycle = -1.0

        arm_offset = strength * arm_amp * arm_cycle
        self._set_ctrl_offset(ctrl, "left_shoulder_pitch", -arm_offset)
        self._set_ctrl_offset(ctrl, "right_shoulder_pitch", arm_offset)

        return ctrl

    def _reference_ctrl(self):
        if self.use_reference_gait:
            return self._gait_reference_ctrl()

        return self.default_ctrl.copy()

    def _clip_ctrl(self, ctrl):
        clipped = ctrl.copy()

        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                low, high = self.model.actuator_ctrlrange[i]
                clipped[i] = np.clip(clipped[i], low, high)

        return clipped

    def _body_has_contact(self, body_id):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]

            if geom1_body == body_id or geom2_body == body_id:
                return True

        return False

    def _foot_info(self):
        left_pos = self.data.xpos[self.left_foot_body_id].copy()
        right_pos = self.data.xpos[self.right_foot_body_id].copy()

        return {
            "left_pos": left_pos,
            "right_pos": right_pos,
            "left_contact": self._body_has_contact(self.left_foot_body_id),
            "right_contact": self._body_has_contact(self.right_foot_body_id),
        }


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel

        if self.model.nu > 0:
            self.prev_ctrl = self._clip_ctrl(self._reference_ctrl())
            self.data.ctrl[:] = self.prev_ctrl

        mujoco.mj_forward(self.model, self.data)

        self.prev_x = float(self.data.qpos[0])
        self.prev_action[:] = 0.0
        self.prev_distance = float(
            np.linalg.norm(self.target_pos[:2] - self.data.qpos[:2])
        )
        foot_info = self._foot_info()
        self.prev_left_contact = foot_info["left_contact"]
        self.prev_right_contact = foot_info["right_contact"]
        self.prev_left_foot_pos = foot_info["left_pos"].copy()
        self.prev_right_foot_pos = foot_info["right_pos"].copy()

        info = {
            "x_position": float(self.data.qpos[0]),
            "y_position": float(self.data.qpos[1]),
            "distance_to_target": self.prev_distance,
            "torso_height": self._torso_height(),
            "torso_upright": self._torso_upright(),
        }

        return self._get_obs(), info

    def step(self, action):
        self.current_step += 1

        # Limita as ações
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Converte a ação da política em correções sobre a marcha base.
        reference_ctrl = self._reference_ctrl()
        desired_ctrl = reference_ctrl + self.action_scale * action
        desired_ctrl = self._clip_ctrl(desired_ctrl)
        target_ctrl = self.prev_ctrl + self.ctrl_blend * (
            desired_ctrl - self.prev_ctrl
        )
        target_ctrl = self._clip_ctrl(target_ctrl)
        gait_reference_ctrl = self._gait_reference_ctrl()

        self.data.ctrl[:] = target_ctrl

        # Executa a simulação
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            if self.render_mode == "human":
                self.render()

        # ------------------------------
        # Estado atual
        # ------------------------------
        current_x = float(self.data.qpos[0])
        current_y = float(self.data.qpos[1])
        robot_pos = np.array([current_x, current_y], dtype=np.float32)

        target_pos = self.target_pos[:2]

        current_x_vel = float(self.data.qvel[0])
        current_y_vel = float(self.data.qvel[1])
        robot_vel = np.array([current_x_vel, current_y_vel], dtype=np.float32)

        forward_velocity = (current_x - self.prev_x) / self.dt
        
        torso_height = self._torso_height()
        torso_upright = self._torso_upright()
        torso_up_axis = self._torso_up_axis()
        foot_info = self._foot_info()
        left_foot_vel = (foot_info["left_pos"] - self.prev_left_foot_pos) / self.dt
        right_foot_vel = (foot_info["right_pos"] - self.prev_right_foot_pos) / self.dt
        gait_phase = self._gait_phase()
        leg_cycle = float(np.sin(gait_phase))
        phase_strength = min(1.0, abs(leg_cycle) / 0.35)

        fallen = self._has_fallen()

        # ------------------------------
        # Recompensa
        # ------------------------------

        reward = 0.0

        # andar em direcao ao target
        pdif = target_pos - robot_pos
        distance = np.linalg.norm(pdif)

        if distance > 1e-6:
            phat = pdif / distance
            velocity_to_target = float(np.dot(robot_vel, phat))
        else:
            phat = np.array([-1.0, 0.0], dtype=np.float32)
            velocity_to_target = 0.0

        chest_xy = self.data.xpos[self.chest_body_id][:2]
        left_foot_xy = foot_info["left_pos"][:2]
        right_foot_xy = foot_info["right_pos"][:2]
        feet_mid_xy = 0.5 * (left_foot_xy + right_foot_xy)
        chest_forward_gap = float(np.dot(chest_xy - feet_mid_xy, phat))
        body_projection_penalty = max(0.0, chest_forward_gap - 0.035) ** 2
        moving_body_projection_penalty = max(
            0.0,
            chest_forward_gap - 0.025,
        ) * max(0.0, velocity_to_target)
        support_velocity_factor = np.clip(
            (0.16 - chest_forward_gap) / 0.13,
            0.0,
            1.0,
        )

        forward_lean = float(np.dot(torso_up_axis[:2], phat))
        forward_lean_penalty = max(0.0, forward_lean - 0.04) ** 2
        unstable_forward_lean = max(0.0, forward_lean - 0.20) ** 2
        moving_forward_lean_penalty = max(0.0, forward_lean - 0.075) * max(
            0.0,
            velocity_to_target,
        )
        posture_velocity_factor = np.clip((torso_upright - 0.58) / 0.32, 0.0, 1.0)
        lean_velocity_factor = np.clip((0.30 - forward_lean) / 0.26, 0.0, 1.0)
        balance_velocity_factor = (
            posture_velocity_factor * lean_velocity_factor * support_velocity_factor
        )
        movement_reward_factor = max(0.22, balance_velocity_factor)

        if velocity_to_target > 0.0:
            rewarded_velocity_to_target = velocity_to_target * movement_reward_factor
        else:
            rewarded_velocity_to_target = velocity_to_target

        progress_to_target = self.prev_distance - distance
        if progress_to_target > 0.0:
            rewarded_progress_to_target = progress_to_target * movement_reward_factor
        else:
            rewarded_progress_to_target = progress_to_target

        if forward_lean > self.max_forward_lean:
            fallen = True
            rewarded_velocity_to_target = min(0.0, rewarded_velocity_to_target)
            rewarded_progress_to_target = min(0.0, rewarded_progress_to_target)

        desired_crouch = self.desired_crouch
        allowed_crouch = self.allowed_crouch
        desired_torso_height = self.initial_torso_height - desired_crouch
        height_error = torso_height - desired_torso_height
        height_reward = np.exp(-((height_error / 0.07) ** 2))
        low_height = max(
            0.0,
            self.initial_torso_height - torso_height - allowed_crouch,
        )
        lateral_drift = abs(current_y)
        path_lateral_error = float(
            np.dot(robot_pos - self.initial_xy, self.target_line_normal)
        )
        path_drift = abs(path_lateral_error)
        straight_corridor_factor = float(np.exp(-((path_drift / 0.12) ** 2)))
        lateral_velocity_vec = robot_vel - velocity_to_target * phat
        lateral_velocity = float(np.linalg.norm(lateral_velocity_vec))
        stationary_xy_error = float(np.linalg.norm(robot_pos - self.initial_xy))
        stationary_velocity_penalty = float(np.linalg.norm(robot_vel))
        reverse_velocity_penalty = max(0.0, -velocity_to_target)
        reverse_progress_penalty = max(0.0, -progress_to_target)
        displacement_to_target = float(
            np.dot(robot_pos - self.initial_xy, self.target_line_dir)
        )
        reverse_drift_penalty = max(0.0, -displacement_to_target)
        robot_speed = float(np.linalg.norm(robot_vel))
        if robot_speed > 1e-6:
            velocity_alignment = float(
                np.clip(velocity_to_target / robot_speed, -1.0, 1.0)
            )
        else:
            velocity_alignment = 0.0
        target_alignment_score = (
            min(max((velocity_alignment - 0.55) / 0.45, 0.0), 1.0)
            * min(max(velocity_to_target / max(self.desired_walk_speed, 1e-6), 0.0), 1.0)
            * max(0.35, straight_corridor_factor)
        )
        crooked_velocity_penalty = 0.0
        if self.current_step > 20 and velocity_to_target > 0.02:
            crooked_velocity_penalty = min(
                lateral_velocity / max(abs(velocity_to_target), 0.05),
                1.0,
            ) * min(
                max(velocity_to_target / max(self.desired_walk_speed, 1e-6), 0.0),
                1.0,
            )
        control_cost = float(np.mean(np.square(action)))
        smooth_cost = float(np.mean(np.square(action - self.prev_action)))
        ctrl_delta = (target_ctrl - self.prev_ctrl) / np.maximum(
            self.action_scale,
            1e-6,
        )
        smooth_ctrl_cost = float(np.mean(np.square(ctrl_delta)))
        upper_body_cost = float(np.mean(np.square(action[:8])))
        left_arm_pitch = self._ctrl_offset(target_ctrl, "left_shoulder_pitch")
        right_arm_pitch = self._ctrl_offset(target_ctrl, "right_shoulder_pitch")
        left_arm_roll = self._ctrl_offset(target_ctrl, "left_shoulder_roll")
        right_arm_roll = self._ctrl_offset(target_ctrl, "right_shoulder_roll")
        left_elbow = self._ctrl_offset(target_ctrl, "left_elbow_pitch")
        right_elbow = self._ctrl_offset(target_ctrl, "right_elbow_pitch")

        arm_opposition = abs(left_arm_pitch - right_arm_pitch)
        arm_swing_score = np.tanh(arm_opposition / 0.12) * phase_strength
        arm_balance_penalty = min(
            abs(left_arm_pitch + right_arm_pitch) / 0.16,
            1.0,
        )
        arm_flail_penalty = min(
            (
                abs(left_arm_roll)
                + abs(right_arm_roll)
                + 0.50 * abs(left_elbow)
                + 0.50 * abs(right_elbow)
            )
            / 0.35,
            1.0,
        )
        lower_body_indices = [
            self.actuator_index[name]
            for name in (
                "right_leg_yaw",
                "right_leg_roll",
                "right_leg_pitch",
                "right_knee",
                "right_foot_pitch",
                "right_foot_roll",
                "left_leg_yaw",
                "left_leg_roll",
                "left_leg_pitch",
                "left_knee",
                "left_foot_pitch",
                "left_foot_roll",
            )
            if name in self.actuator_index
        ]
        reference_error = float(
            np.mean(
                np.square(
                    (
                        target_ctrl[lower_body_indices]
                        - gait_reference_ctrl[lower_body_indices]
                    )
                    / self.action_scale[lower_body_indices]
                )
            )
        )

        left_contact = foot_info["left_contact"]
        right_contact = foot_info["right_contact"]
        any_foot_contact = left_contact or right_contact
        both_feet_contact = left_contact and right_contact
        left_clearance = max(
            0.0, float(foot_info["left_pos"][2]) - self.initial_left_foot_z
        )
        right_clearance = max(
            0.0, float(foot_info["right_pos"][2]) - self.initial_right_foot_z
        )

        if any_foot_contact:
            support_score = 0.0 if both_feet_contact else 1.0
        else:
            support_score = -1.0

        if leg_cycle >= 0.0:
            stance_contact = right_contact
            swing_contact = left_contact
            prev_stance_contact = self.prev_right_contact
            swing_clearance = left_clearance
            stance_clearance = right_clearance
            swing_new_contact = left_contact and not self.prev_left_contact
            swing_foot_vel = left_foot_vel
            stance_foot_vel = right_foot_vel
            expected_step_order = float(
                np.dot(foot_info["left_pos"][:2] - foot_info["right_pos"][:2], phat)
            )
            left_knee_weight = 1.0
            right_knee_weight = -1.0
        else:
            stance_contact = left_contact
            swing_contact = right_contact
            prev_stance_contact = self.prev_left_contact
            swing_clearance = right_clearance
            stance_clearance = left_clearance
            swing_new_contact = right_contact and not self.prev_right_contact
            swing_foot_vel = right_foot_vel
            stance_foot_vel = left_foot_vel
            expected_step_order = float(
                np.dot(foot_info["right_pos"][:2] - foot_info["left_pos"][:2], phat)
            )
            left_knee_weight = -1.0
            right_knee_weight = 1.0

        left_right_forward_gap = float(
            np.dot(foot_info["left_pos"][:2] - foot_info["right_pos"][:2], phat)
        )
        step_forward_distance = abs(left_right_forward_gap)
        new_foot_contact = (
            (left_contact and not self.prev_left_contact)
            or (right_contact and not self.prev_right_contact)
        )
        previous_support_contact = self.prev_left_contact or self.prev_right_contact
        forward_motion_score = min(
            max(velocity_to_target / max(self.desired_walk_speed, 1e-6), 0.0),
            1.0,
        )

        gait_timing_score = 0.0
        gait_timing_score += 0.70 if stance_contact else -0.70
        gait_timing_score += 0.30 if not swing_contact else -0.30
        gait_timing_score *= phase_strength

        single_leg_support_score = support_score * max(0.25, forward_motion_score)
        if (left_contact and not right_contact) or (right_contact and not left_contact):
            one_foot_walk_score = 1.0
        elif both_feet_contact:
            one_foot_walk_score = 0.15
        elif not any_foot_contact:
            one_foot_walk_score = -0.70
        else:
            one_foot_walk_score = 0.0

        target_swing_clearance = 0.038
        swing_lift_ratio = min(swing_clearance / target_swing_clearance, 1.0)
        swing_clearance_score = swing_lift_ratio
        swing_lift_score = swing_lift_ratio * max(0.25, forward_motion_score)
        stance_lift_penalty = min(stance_clearance / 0.016, 1.0)
        clearance_score = phase_strength * (
            swing_clearance_score - 0.80 * stance_lift_penalty
        )
        foot_spacing_y = abs(
            float(foot_info["right_pos"][1] - foot_info["left_pos"][1])
        )
        foot_spacing_error = foot_spacing_y - self.initial_foot_spacing_y
        foot_spacing_reward = np.exp(-((foot_spacing_error / 0.045) ** 2))
        foot_side_distance = self.initial_foot_side_sign * float(
            foot_info["right_pos"][1] - foot_info["left_pos"][1]
        )
        min_side_distance = 0.45 * self.initial_foot_spacing_y
        foot_crossing_penalty = min(
            max(0.0, min_side_distance - foot_side_distance)
            / max(min_side_distance, 1e-6),
            1.0,
        )
        target_speed_error = (
            (velocity_to_target - self.desired_walk_speed)
            / max(self.desired_walk_speed, 1e-6)
        )
        walk_speed_score = np.exp(-(target_speed_error ** 2))
        if velocity_to_target < 0.0:
            walk_speed_score = -1.0
        safe_speed_score = min(
            max(velocity_to_target / max(self.desired_walk_speed, 1e-6), 0.0),
            1.25,
        )
        safe_speed_score *= support_velocity_factor
        safe_speed_score *= max(0.45, straight_corridor_factor)

        step_length_error = (
            (step_forward_distance - self.desired_step_length)
            / max(self.desired_step_length, 1e-6)
        )
        step_length_score = np.exp(-(step_length_error ** 2)) * max(
            0.20,
            forward_motion_score,
        )
        upright_stride_factor = np.clip((0.28 - forward_lean) / 0.24, 0.0, 1.0)
        upright_stride_factor *= np.clip((torso_upright - 0.58) / 0.32, 0.0, 1.0)
        support_stride_factor = np.clip(
            (0.13 - chest_forward_gap) / 0.10,
            0.0,
            1.0,
        )
        upright_stride_factor *= support_stride_factor
        stride_quality_factor = max(0.12, upright_stride_factor)
        min_human_step = 0.045
        max_human_step = 0.170
        stride_reach_score = min(
            max(
                (step_forward_distance - min_human_step)
                / max(self.desired_step_length - min_human_step, 1e-6),
                0.0,
            ),
            1.0,
        )
        stride_overreach_penalty = min(
            max(
                (step_forward_distance - max_human_step)
                / max(max_human_step - self.desired_step_length, 1e-6),
                0.0,
            ),
            1.0,
        )
        human_stride_score = (
            stride_reach_score
            * (1.0 - stride_overreach_penalty)
            * max(0.25, forward_motion_score)
            * stride_quality_factor
            * max(0.45, straight_corridor_factor)
            * max(0.35, swing_lift_ratio)
        )
        stride_progress_score = (
            stride_reach_score
            * forward_motion_score
            * upright_stride_factor
            * max(0.45, straight_corridor_factor)
            * max(0.35, swing_lift_ratio)
        )
        stride_forward_lean_penalty = (
            max(0.0, forward_lean - 0.10)
            * stride_reach_score
            * max(0.25, forward_motion_score)
        )
        short_step_penalty = 0.0
        if self.current_step > 20:
            short_step_penalty = (
                max(0.0, min_human_step - step_forward_distance)
                / max(min_human_step, 1e-6)
            ) * max(0.25, forward_motion_score)
        missing_swing_lift_penalty = 0.0
        if self.current_step > 20:
            missing_swing_lift_penalty = (
                max(0.0, target_swing_clearance - swing_clearance)
                / max(target_swing_clearance, 1e-6)
            ) * stride_reach_score * max(0.25, forward_motion_score)
        step_commit_score = min(
            max(0.0, step_forward_distance - 0.03)
            / max(self.desired_step_length - 0.03, 1e-6),
            1.0,
        ) * max(0.20, forward_motion_score) * stride_quality_factor * max(
            0.35,
            swing_lift_ratio,
        )

        swing_foot_forward_speed = float(np.dot(swing_foot_vel[:2], phat))
        swing_foot_forward_score = np.tanh(
            swing_foot_forward_speed / self.desired_walk_speed
        ) * max(0.25, forward_motion_score) * stride_quality_factor * max(
            0.35,
            swing_lift_ratio,
        )

        footstep_landing_score = 0.0
        if new_foot_contact:
            footstep_landing_score = np.exp(-(step_length_error ** 2))
            if step_forward_distance < 0.02:
                footstep_landing_score -= 1.0
        step_transition_score = 0.0
        if new_foot_contact:
            landing_reach_score = min(
                max(
                    (step_forward_distance - 0.03)
                    / max(self.desired_step_length - 0.03, 1e-6),
                    0.0,
                ),
                1.0,
            )
            support_continuity = 1.0 if previous_support_contact else 0.30
            step_transition_score = support_continuity * (
                0.40 + 0.60 * landing_reach_score
            )
            if step_forward_distance < 0.02:
                step_transition_score -= 0.70
        fluid_step_score = 0.0
        if new_foot_contact:
            fluid_step_score = (
                step_commit_score
                * safe_speed_score
                * max(0.25, one_foot_walk_score)
                * max(0.35, straight_corridor_factor)
            )

        left_slip_speed = np.linalg.norm(left_foot_vel[:2]) if left_contact else 0.0
        right_slip_speed = np.linalg.norm(right_foot_vel[:2]) if right_contact else 0.0
        stance_slip_speed = np.linalg.norm(stance_foot_vel[:2]) if stance_contact else 0.0
        stance_stability_score = 0.0
        if stance_contact:
            stance_stability_score = (
                1.0
                - min(max(0.0, stance_slip_speed - 0.008) / 0.040, 1.0)
            ) * phase_strength
        foot_slip_penalty = min(
            (
                max(0.0, left_slip_speed - 0.03)
                + max(0.0, right_slip_speed - 0.03)
                + max(0.0, stance_slip_speed - 0.02)
            )
            / 0.18,
            1.0,
        )
        standstill_penalty = np.exp(-((velocity_to_target / 0.025) ** 2))
        min_target_speed = 0.11
        slow_velocity_penalty = 0.0
        if self.current_step > 20 and not fallen:
            slow_velocity_penalty = min(
                max(0.0, (min_target_speed - velocity_to_target) / min_target_speed),
                1.0,
            )
        step_order_score = np.tanh(expected_step_order / 0.035) * phase_strength

        left_knee_flex = max(0.0, self._ctrl_offset(target_ctrl, "left_knee"))
        right_knee_flex = max(0.0, -self._ctrl_offset(target_ctrl, "right_knee"))
        left_knee_actual = max(0.0, self._joint_qpos("left_knee"))
        right_knee_actual = max(0.0, -self._joint_qpos("right_knee"))
        left_knee_vel = self._joint_qvel("left_knee")
        right_knee_vel = -self._joint_qvel("right_knee")
        left_ankle_offset = self._ctrl_offset(target_ctrl, "left_foot_pitch")
        right_ankle_offset = -self._ctrl_offset(target_ctrl, "right_foot_pitch")

        if leg_cycle >= 0.0:
            swing_knee_actual = left_knee_actual
            stance_knee_actual = right_knee_actual
            swing_knee_vel = left_knee_vel
            swing_ankle_offset = left_ankle_offset
        else:
            swing_knee_actual = right_knee_actual
            stance_knee_actual = left_knee_actual
            swing_knee_vel = right_knee_vel
            swing_ankle_offset = right_ankle_offset

        alternating_knee_score = np.tanh(
            (
                left_knee_weight * left_knee_flex
                + right_knee_weight * right_knee_flex
            )
            / 0.08
        )
        alternating_knee_score *= phase_strength
        target_swing_knee = 0.28
        swing_knee_score = min(swing_knee_actual / target_swing_knee, 1.0)
        swing_knee_score *= phase_strength
        missing_swing_knee_penalty = max(
            0.0,
            (target_swing_knee - swing_knee_actual) / target_swing_knee,
        )
        missing_swing_knee_penalty *= phase_strength
        stance_knee_penalty = min(
            max(0.0, stance_knee_actual - 0.20) / 0.16,
            1.0,
        ) * phase_strength
        knee_clearance_pair = (
            min(swing_knee_actual / 0.28, 1.0)
            * min(swing_clearance / target_swing_clearance, 1.0)
            * phase_strength
        )
        knee_velocity_score = np.tanh(abs(swing_knee_vel) / 1.5) * phase_strength
        ankle_knee_coordination = np.tanh(
            max(0.0, swing_knee_actual) * max(0.0, swing_ankle_offset) / 0.05
        )
        ankle_knee_coordination *= phase_strength
        average_knee_flex = 0.5 * (left_knee_actual + right_knee_actual)
        soft_crouch_score = np.exp(
            -(((average_knee_flex - self.desired_crouch_knee) / 0.10) ** 2)
        )
        soft_crouch_score *= np.clip((torso_upright - 0.46) / 0.34, 0.0, 1.0)
        double_knee_flex_penalty = min(
            max(0.0, min(left_knee_flex, right_knee_flex) - 0.22) / 0.18,
            1.0,
        )
        double_support_penalty = float(both_feet_contact) * phase_strength
        double_foot_lift_penalty = float(not any_foot_contact)

        if left_clearance > 0.018 and right_clearance > 0.018:
            double_foot_lift_penalty += 1.0

        vertical_motion_penalty = max(0.0, abs(float(self.data.qvel[2])) - 0.15) ** 2
        gait_imitation_factor = 1.0 if self.use_reference_gait else 0.0
        gait_pose_specs = (
            ("left_leg_pitch", 0.35),
            ("left_knee", 0.55),
            ("left_foot_pitch", 0.35),
            ("right_leg_pitch", 0.35),
            ("right_knee", 0.55),
            ("right_foot_pitch", 0.35),
        )
        gait_pose_errors = []

        for joint_name, scale in gait_pose_specs:
            actual_offset = self._joint_offset(joint_name)
            reference_offset = self._ctrl_offset(gait_reference_ctrl, joint_name)
            gait_pose_errors.append(
                ((actual_offset - reference_offset) / max(scale, 1e-6)) ** 2
            )

        gait_pose_error = float(np.mean(gait_pose_errors)) if gait_pose_errors else 0.0
        gait_pose_score = float(np.exp(-gait_pose_error))
        gait_forward_score = gait_pose_score * min(
            max(velocity_to_target / max(self.desired_walk_speed, 1e-6), 0.0),
            1.0,
        )
        left_foot_angle_error = self._body_rotation_error(
            self.left_foot_body_id,
            self.initial_left_foot_rot,
        )
        right_foot_angle_error = self._body_rotation_error(
            self.right_foot_body_id,
            self.initial_right_foot_rot,
        )

        if leg_cycle >= 0.0:
            stance_foot_angle_error = right_foot_angle_error
        else:
            stance_foot_angle_error = left_foot_angle_error

        stance_foot_flat_score = float(
            np.exp(-((stance_foot_angle_error / 0.08) ** 2))
        )
        if not stance_contact:
            stance_foot_flat_score = 0.0

        reward_terms = {
            "target_velocity": self.reward_weights["target_velocity"]
            * rewarded_velocity_to_target,
            "distance_progress": self.reward_weights["distance_progress"]
            * rewarded_progress_to_target,
            "alive": self.reward_weights["alive"],
            "height": self.reward_weights["height"] * height_reward,
            "soft_crouch": self.reward_weights["soft_crouch"]
            * soft_crouch_score,
            "upright": self.reward_weights["upright"] * max(0.0, torso_upright),
            "low_height": -self.reward_weights["low_height"] * (low_height ** 2),
            "forward_lean": -self.reward_weights["forward_lean"]
            * forward_lean_penalty,
            "unstable_forward_lean": -self.reward_weights["unstable_forward_lean"]
            * unstable_forward_lean,
            "moving_forward_lean": -self.reward_weights["moving_forward_lean"]
            * moving_forward_lean_penalty,
            "body_projection": -self.reward_weights["body_projection"]
            * body_projection_penalty,
            "moving_body_projection": -self.reward_weights[
                "moving_body_projection"
            ]
            * moving_body_projection_penalty,
            "lateral_drift": -self.reward_weights["lateral_drift"] * lateral_drift,
            "path_drift": -self.reward_weights["path_drift"] * (path_drift ** 2),
            "lateral_velocity": -self.reward_weights["lateral_velocity"]
            * (lateral_velocity ** 2),
            "crooked_velocity": -self.reward_weights["crooked_velocity"]
            * crooked_velocity_penalty,
            "target_alignment": self.reward_weights["target_alignment"]
            * target_alignment_score,
            "stationary_xy": -self.reward_weights["stationary_xy"]
            * (stationary_xy_error ** 2),
            "stationary_velocity": -self.reward_weights["stationary_velocity"]
            * stationary_velocity_penalty,
            "reverse_velocity": -self.reward_weights["reverse_velocity"]
            * reverse_velocity_penalty,
            "reverse_progress": -self.reward_weights["reverse_progress"]
            * reverse_progress_penalty,
            "reverse_drift": -self.reward_weights["reverse_drift"]
            * (reverse_drift_penalty ** 2),
            "control": -self.reward_weights["control"] * control_cost,
            "smooth_action": -self.reward_weights["smooth_action"] * smooth_cost,
            "smooth_ctrl": -self.reward_weights["smooth_ctrl"] * smooth_ctrl_cost,
            "upper_body_motion": -self.reward_weights["upper_body_motion"]
            * upper_body_cost,
            "arm_swing": self.reward_weights["arm_swing"] * arm_swing_score,
            "arm_balance": -self.reward_weights["arm_balance"] * arm_balance_penalty,
            "arm_flail": -self.reward_weights["arm_flail"] * arm_flail_penalty,
            "reference_gait": -self.reward_weights["reference_gait"]
            * reference_error
            * gait_imitation_factor,
            "gait_pose": self.reward_weights["gait_pose"]
            * gait_pose_score
            * gait_imitation_factor,
            "gait_forward": self.reward_weights["gait_forward"]
            * gait_forward_score
            * gait_imitation_factor,
            "gait_pose_error": -self.reward_weights["gait_pose_error"]
            * gait_pose_error
            * gait_imitation_factor,
            "stance_foot_flat": self.reward_weights["stance_foot_flat"]
            * stance_foot_flat_score
            * gait_imitation_factor,
            "foot_support": self.reward_weights["foot_support"] * support_score,
            "single_leg_support": self.reward_weights["single_leg_support"]
            * single_leg_support_score,
            "one_foot_walk": self.reward_weights["one_foot_walk"]
            * one_foot_walk_score,
            "stance_stability": self.reward_weights["stance_stability"]
            * stance_stability_score,
            "gait_timing": self.reward_weights["gait_timing"]
            * gait_timing_score,
            "foot_clearance": self.reward_weights["foot_clearance"]
            * clearance_score,
            "swing_lift": self.reward_weights["swing_lift"] * swing_lift_score,
            "missing_swing_lift": -self.reward_weights["missing_swing_lift"]
            * missing_swing_lift_penalty,
            "foot_spacing": self.reward_weights["foot_spacing"]
            * foot_spacing_reward,
            "foot_crossing": -self.reward_weights["foot_crossing"]
            * foot_crossing_penalty,
            "walk_speed": self.reward_weights["walk_speed"] * walk_speed_score,
            "safe_speed": self.reward_weights["safe_speed"] * safe_speed_score,
            "swing_foot_forward": self.reward_weights["swing_foot_forward"]
            * swing_foot_forward_score,
            "step_length": self.reward_weights["step_length"] * step_length_score,
            "human_stride": self.reward_weights["human_stride"]
            * human_stride_score,
            "stride_progress": self.reward_weights["stride_progress"]
            * stride_progress_score,
            "stride_forward_lean": -self.reward_weights["stride_forward_lean"]
            * stride_forward_lean_penalty,
            "short_step": -self.reward_weights["short_step"]
            * short_step_penalty,
            "step_commit": self.reward_weights["step_commit"] * step_commit_score,
            "footstep_landing": self.reward_weights["footstep_landing"]
            * footstep_landing_score,
            "step_transition": self.reward_weights["step_transition"]
            * step_transition_score,
            "fluid_step": self.reward_weights["fluid_step"] * fluid_step_score,
            "foot_slip": -self.reward_weights["foot_slip"] * foot_slip_penalty,
            "standstill": -self.reward_weights["standstill"]
            * standstill_penalty,
            "slow_velocity": -self.reward_weights["slow_velocity"]
            * slow_velocity_penalty,
            "step_order": self.reward_weights["step_order"] * step_order_score,
            "alternating_knees": self.reward_weights["alternating_knees"]
            * alternating_knee_score,
            "swing_knee_flex": self.reward_weights["swing_knee_flex"]
            * swing_knee_score,
            "missing_swing_knee": -self.reward_weights["missing_swing_knee"]
            * missing_swing_knee_penalty,
            "stance_knee_straight": -self.reward_weights["stance_knee_straight"]
            * stance_knee_penalty,
            "knee_clearance_pair": self.reward_weights["knee_clearance_pair"]
            * knee_clearance_pair,
            "knee_velocity": self.reward_weights["knee_velocity"]
            * knee_velocity_score,
            "ankle_knee_coordination": self.reward_weights[
                "ankle_knee_coordination"
            ]
            * ankle_knee_coordination,
            "double_support": -self.reward_weights["double_support"]
            * double_support_penalty,
            "double_knee_flex": -self.reward_weights["double_knee_flex"]
            * double_knee_flex_penalty,
            "double_foot_lift": -self.reward_weights["double_foot_lift"]
            * double_foot_lift_penalty,
            "vertical_motion": -self.reward_weights["vertical_motion"]
            * vertical_motion_penalty,
        }

        reward = sum(reward_terms.values())

        # penalidade por queda
        if fallen:
            reward_terms["fall"] = -self.reward_weights["fall"]
            reward += reward_terms["fall"]

        # ------------------------------
        # Finalização
        # ------------------------------
        self.prev_x = current_x
        self.prev_distance = distance
        self.prev_action[:] = action
        self.prev_ctrl[:] = target_ctrl
        self.prev_left_contact = left_contact
        self.prev_right_contact = right_contact
        self.prev_left_foot_pos = foot_info["left_pos"].copy()
        self.prev_right_foot_pos = foot_info["right_pos"].copy()

        terminated = fallen

        # bônus por chegar ao alvo
        if distance < self.target_radius and not fallen:
            reward_terms["target_reached"] = self.reward_weights["target_reached"]
            reward += reward_terms["target_reached"]
            terminated = True

        truncated = self.current_step >= self.max_episode_steps

        obs = self._get_obs()

        info = {
            "forward_velocity": forward_velocity,
            "is_success": bool(distance < self.target_radius and not fallen),
            "fallen": bool(fallen),
            "elapsed_simulation_seconds": self.current_step * self.dt,
            "time_limit_reached": bool(truncated and not terminated),
            "x_position": current_x,
            "y_position": current_y,
            "torso_height": torso_height,
            "torso_upright": torso_upright,
            "distance_to_target": distance,
            "velocity_to_target": velocity_to_target,
            "lateral_velocity": lateral_velocity,
            "rewarded_velocity_to_target": rewarded_velocity_to_target,
            "progress_to_target": progress_to_target,
            "rewarded_progress_to_target": rewarded_progress_to_target,
            "balance_velocity_factor": balance_velocity_factor,
            "movement_reward_factor": movement_reward_factor,
            "slow_velocity_penalty": slow_velocity_penalty,
            "soft_crouch_score": soft_crouch_score,
            "forward_lean": forward_lean,
            "moving_forward_lean_penalty": moving_forward_lean_penalty,
            "chest_forward_gap": chest_forward_gap,
            "body_projection_penalty": body_projection_penalty,
            "moving_body_projection_penalty": moving_body_projection_penalty,
            "support_velocity_factor": support_velocity_factor,
            "path_lateral_error": path_lateral_error,
            "path_drift": path_drift,
            "straight_corridor_factor": straight_corridor_factor,
            "velocity_alignment": velocity_alignment,
            "target_alignment_score": target_alignment_score,
            "crooked_velocity_penalty": crooked_velocity_penalty,
            "stationary_xy_error": stationary_xy_error,
            "stationary_velocity_penalty": stationary_velocity_penalty,
            "reverse_velocity_penalty": reverse_velocity_penalty,
            "reverse_progress_penalty": reverse_progress_penalty,
            "reverse_drift_penalty": reverse_drift_penalty,
            "displacement_to_target": displacement_to_target,
            "walk_speed_score": walk_speed_score,
            "safe_speed_score": safe_speed_score,
            "swing_foot_forward_score": swing_foot_forward_score,
            "step_length_score": step_length_score,
            "human_stride_score": human_stride_score,
            "stride_progress_score": stride_progress_score,
            "upright_stride_factor": upright_stride_factor,
            "support_stride_factor": support_stride_factor,
            "stride_forward_lean_penalty": stride_forward_lean_penalty,
            "short_step_penalty": short_step_penalty,
            "swing_lift_score": swing_lift_score,
            "swing_lift_ratio": swing_lift_ratio,
            "missing_swing_lift_penalty": missing_swing_lift_penalty,
            "swing_clearance": swing_clearance,
            "step_commit_score": step_commit_score,
            "footstep_landing_score": footstep_landing_score,
            "step_transition_score": step_transition_score,
            "fluid_step_score": fluid_step_score,
            "one_foot_walk_score": one_foot_walk_score,
            "stance_stability_score": stance_stability_score,
            "foot_slip_penalty": foot_slip_penalty,
            "foot_crossing_penalty": foot_crossing_penalty,
            "smooth_ctrl_cost": smooth_ctrl_cost,
            "reference_error": reference_error,
            "gait_pose_score": gait_pose_score,
            "gait_forward_score": gait_forward_score,
            "gait_pose_error": gait_pose_error,
            "stance_foot_flat_score": stance_foot_flat_score,
            "left_foot_angle_error": left_foot_angle_error,
            "right_foot_angle_error": right_foot_angle_error,
            "arm_swing_score": arm_swing_score,
            "arm_balance_penalty": arm_balance_penalty,
            "arm_flail_penalty": arm_flail_penalty,
            "double_foot_lift_penalty": double_foot_lift_penalty,
            "step_forward_distance": step_forward_distance,
            "left_right_forward_gap": left_right_forward_gap,
            "expected_step_order": expected_step_order,
            "left_foot_contact": foot_info["left_contact"],
            "right_foot_contact": foot_info["right_contact"],
            "left_knee_flex": left_knee_actual,
            "right_knee_flex": right_knee_actual,
            "gait_phase": gait_phase,
            "reward_terms": reward_terms,
            "reward": reward,
        }

        if self.debug_reward:
            print(
                f"x={current_x:.4f} "
                f"y={current_y:.4f} "
                f"vel_target={velocity_to_target:.3f} "
                f"torso={torso_height:.3f} "
                f"up={torso_upright:.2f} "
                f"feet=({int(foot_info['left_contact'])},"
                f"{int(foot_info['right_contact'])}) "
                f"one_foot={one_foot_walk_score:.2f} "
                f"stride={human_stride_score:.2f} "
                f"knees=({left_knee_actual:.2f},{right_knee_actual:.2f}) "
                f"reward={reward:.3f} "
                f"dist={distance:.2f} "
            )

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
