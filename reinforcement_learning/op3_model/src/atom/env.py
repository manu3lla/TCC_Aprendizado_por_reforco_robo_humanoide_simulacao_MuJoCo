import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer


class AtomEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode=None, debug_reward=False, use_reference_gait=False):

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

        # A fase da marcha entra na observacao. A referencia guia a recompensa,
        # mas por padrao nao pilota os motores diretamente.
        self.use_reference_gait = use_reference_gait
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

        self.gait_amplitude = {
            "arm_pitch": 0.09,
            "leg_pitch": 0.16,
            "knee": 0.42,
            "foot_pitch": 0.10,
        }

        self.frame_skip = 5
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.max_episode_steps = 1000

        self.prev_x = 0.0
        self.current_step = 0
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_ctrl = self.default_ctrl.copy()
        self.prev_distance = 0.0
        self.ctrl_blend = 0.55
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

        self.chest_body_id = self._body_id("chest")
        self.left_foot_body_id = self._body_id("left_foot")
        self.right_foot_body_id = self._body_id("right_foot")
        self.initial_torso_height = self._torso_height()
        self.initial_left_foot_z = float(self.data.xpos[self.left_foot_body_id][2])
        self.initial_right_foot_z = float(self.data.xpos[self.right_foot_body_id][2])
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
        self.healthy_min_height = 0.38
        self.healthy_min_upright = 0.56
        self.max_forward_lean = 0.58
        self.target_radius = 0.30
        self.desired_walk_speed = 0.16
        self.desired_step_length = 0.12

        self.reward_weights = {
            "target_velocity": 4.30,
            "distance_progress": 56.00,
            "alive": 0.005,
            "height": 0.06,
            "upright": 0.30,
            "low_height": 14.00,
            "forward_lean": 8.00,
            "unstable_forward_lean": 35.00,
            "lateral_drift": 0.80,
            "lateral_velocity": 0.00,
            "control": 0.018,
            "smooth_action": 0.025,
            "smooth_ctrl": 0.00,
            "upper_body_motion": 0.030,
            "arm_swing": 0.00,
            "arm_balance": 0.00,
            "arm_flail": 0.00,
            "reference_gait": 0.00,
            "foot_support": 0.10,
            "single_leg_support": 0.48,
            "gait_timing": 0.37,
            "foot_clearance": 0.26,
            "foot_spacing": 0.04,
            "foot_crossing": 0.25,
            "walk_speed": 0.95,
            "swing_foot_forward": 0.55,
            "step_length": 0.85,
            "footstep_landing": 0.90,
            "step_commit": 0.55,
            "foot_slip": 0.82,
            "standstill": 0.85,
            "slow_velocity": 1.25,
            "step_order": 0.55,
            "alternating_knees": 0.45,
            "swing_knee_flex": 0.95,
            "missing_swing_knee": 0.85,
            "stance_knee_straight": 0.30,
            "knee_clearance_pair": 0.55,
            "knee_velocity": 0.16,
            "ankle_knee_coordination": 0.18,
            "double_support": 0.10,
            "double_knee_flex": 0.30,
            "double_foot_lift": 1.30,
            "vertical_motion": 0.08,
            "fall": 340.0,
            "target_reached": 300.0,
        }

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

    def _joint_qvel(self, joint_name):
        dof_adr = self.joint_dofadr.get(joint_name)

        if dof_adr is None:
            return 0.0

        return float(self.data.qvel[dof_adr])

    def _gait_reference_ctrl(self):
        ctrl = self.default_ctrl.copy()

        phase = self._gait_phase()
        leg_cycle = float(np.sin(phase))
        left_swing = max(0.0, leg_cycle)
        right_swing = max(0.0, -leg_cycle)
        walk_direction = float(np.sign(self.target_pos[0] - self.data.qpos[0]))

        if walk_direction == 0.0:
            walk_direction = -1.0

        arm_amp = self.gait_amplitude["arm_pitch"]
        hip_amp = self.gait_amplitude["leg_pitch"]
        knee_amp = self.gait_amplitude["knee"]
        foot_amp = self.gait_amplitude["foot_pitch"]

        self._set_ctrl_offset(
            ctrl,
            "left_shoulder_pitch",
            walk_direction * arm_amp * leg_cycle,
        )
        self._set_ctrl_offset(
            ctrl,
            "right_shoulder_pitch",
            -walk_direction * arm_amp * leg_cycle,
        )

        self._set_ctrl_offset(
            ctrl,
            "left_leg_pitch",
            -walk_direction * hip_amp * leg_cycle,
        )
        self._set_ctrl_offset(
            ctrl,
            "right_leg_pitch",
            walk_direction * hip_amp * leg_cycle,
        )

        self._set_ctrl_offset(ctrl, "left_knee", knee_amp * left_swing)
        self._set_ctrl_offset(ctrl, "right_knee", -knee_amp * right_swing)

        self._set_ctrl_offset(ctrl, "left_foot_pitch", foot_amp * left_swing)
        self._set_ctrl_offset(ctrl, "right_foot_pitch", -foot_amp * right_swing)

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

        forward_lean = float(np.dot(torso_up_axis[:2], phat))
        forward_lean_penalty = max(0.0, forward_lean - 0.05) ** 2
        unstable_forward_lean = max(0.0, forward_lean - 0.28) ** 2
        posture_velocity_factor = np.clip((torso_upright - 0.56) / 0.34, 0.0, 1.0)
        lean_velocity_factor = np.clip((0.42 - forward_lean) / 0.36, 0.0, 1.0)
        balance_velocity_factor = posture_velocity_factor * lean_velocity_factor
        movement_reward_factor = max(0.55, balance_velocity_factor)

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

        height_error = torso_height - self.initial_torso_height
        height_reward = np.exp(-((height_error / 0.04) ** 2))
        low_height = max(0.0, self.initial_torso_height - torso_height)
        lateral_drift = abs(current_y)
        lateral_velocity_vec = robot_vel - velocity_to_target * phat
        lateral_velocity = float(np.linalg.norm(lateral_velocity_vec))
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

        gait_timing_score = 0.0
        gait_timing_score += 0.70 if stance_contact else -0.70
        gait_timing_score += 0.30 if not swing_contact else -0.30
        gait_timing_score *= phase_strength

        single_leg_support_score = support_score * phase_strength
        swing_clearance_score = min(swing_clearance / 0.030, 1.0)
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

        step_length_error = (
            (expected_step_order - self.desired_step_length)
            / max(self.desired_step_length, 1e-6)
        )
        step_length_score = np.exp(-(step_length_error ** 2)) * phase_strength
        if expected_step_order < 0.0:
            step_length_score -= min(
                abs(expected_step_order) / self.desired_step_length,
                1.0,
            ) * phase_strength
        step_commit_score = min(
            max(0.0, expected_step_order - 0.03)
            / max(self.desired_step_length - 0.03, 1e-6),
            1.0,
        ) * phase_strength

        swing_foot_forward_speed = float(np.dot(swing_foot_vel[:2], phat))
        swing_foot_forward_score = np.tanh(
            swing_foot_forward_speed / self.desired_walk_speed
        ) * phase_strength

        footstep_landing_score = 0.0
        if swing_new_contact and phase_strength > 0.35:
            footstep_landing_score = np.exp(-(step_length_error ** 2))
            if expected_step_order < 0.02:
                footstep_landing_score -= 1.0

        left_slip_speed = np.linalg.norm(left_foot_vel[:2]) if left_contact else 0.0
        right_slip_speed = np.linalg.norm(right_foot_vel[:2]) if right_contact else 0.0
        stance_slip_speed = np.linalg.norm(stance_foot_vel[:2]) if stance_contact else 0.0
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
        min_target_speed = 0.07
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
        target_swing_knee = 0.34
        swing_knee_score = min(swing_knee_actual / target_swing_knee, 1.0)
        swing_knee_score *= phase_strength
        missing_swing_knee_penalty = max(
            0.0,
            (target_swing_knee - swing_knee_actual) / target_swing_knee,
        )
        missing_swing_knee_penalty *= phase_strength
        stance_knee_penalty = min(stance_knee_actual / 0.12, 1.0) * phase_strength
        knee_clearance_pair = (
            min(swing_knee_actual / 0.30, 1.0)
            * min(swing_clearance / 0.030, 1.0)
            * phase_strength
        )
        knee_velocity_score = np.tanh(abs(swing_knee_vel) / 1.5) * phase_strength
        ankle_knee_coordination = np.tanh(
            max(0.0, swing_knee_actual) * max(0.0, swing_ankle_offset) / 0.05
        )
        ankle_knee_coordination *= phase_strength
        double_knee_flex_penalty = min(
            min(left_knee_flex, right_knee_flex) / 0.08,
            1.0,
        )
        double_support_penalty = float(both_feet_contact) * phase_strength
        double_foot_lift_penalty = float(not any_foot_contact)

        if left_clearance > 0.018 and right_clearance > 0.018:
            double_foot_lift_penalty += 1.0

        vertical_motion_penalty = max(0.0, abs(float(self.data.qvel[2])) - 0.15) ** 2

        reward_terms = {
            "target_velocity": self.reward_weights["target_velocity"]
            * rewarded_velocity_to_target,
            "distance_progress": self.reward_weights["distance_progress"]
            * rewarded_progress_to_target,
            "alive": self.reward_weights["alive"],
            "height": self.reward_weights["height"] * height_reward,
            "upright": self.reward_weights["upright"] * max(0.0, torso_upright),
            "low_height": -self.reward_weights["low_height"] * (low_height ** 2),
            "forward_lean": -self.reward_weights["forward_lean"]
            * forward_lean_penalty,
            "unstable_forward_lean": -self.reward_weights["unstable_forward_lean"]
            * unstable_forward_lean,
            "lateral_drift": -self.reward_weights["lateral_drift"] * lateral_drift,
            "lateral_velocity": -self.reward_weights["lateral_velocity"]
            * (lateral_velocity ** 2),
            "control": -self.reward_weights["control"] * control_cost,
            "smooth_action": -self.reward_weights["smooth_action"] * smooth_cost,
            "smooth_ctrl": -self.reward_weights["smooth_ctrl"] * smooth_ctrl_cost,
            "upper_body_motion": -self.reward_weights["upper_body_motion"]
            * upper_body_cost,
            "arm_swing": self.reward_weights["arm_swing"] * arm_swing_score,
            "arm_balance": -self.reward_weights["arm_balance"] * arm_balance_penalty,
            "arm_flail": -self.reward_weights["arm_flail"] * arm_flail_penalty,
            "reference_gait": -self.reward_weights["reference_gait"]
            * reference_error,
            "foot_support": self.reward_weights["foot_support"] * support_score,
            "single_leg_support": self.reward_weights["single_leg_support"]
            * single_leg_support_score,
            "gait_timing": self.reward_weights["gait_timing"]
            * gait_timing_score,
            "foot_clearance": self.reward_weights["foot_clearance"]
            * clearance_score,
            "foot_spacing": self.reward_weights["foot_spacing"]
            * foot_spacing_reward,
            "foot_crossing": -self.reward_weights["foot_crossing"]
            * foot_crossing_penalty,
            "walk_speed": self.reward_weights["walk_speed"] * walk_speed_score,
            "swing_foot_forward": self.reward_weights["swing_foot_forward"]
            * swing_foot_forward_score,
            "step_length": self.reward_weights["step_length"] * step_length_score,
            "step_commit": self.reward_weights["step_commit"] * step_commit_score,
            "footstep_landing": self.reward_weights["footstep_landing"]
            * footstep_landing_score,
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
            "forward_lean": forward_lean,
            "walk_speed_score": walk_speed_score,
            "swing_foot_forward_score": swing_foot_forward_score,
            "step_length_score": step_length_score,
            "step_commit_score": step_commit_score,
            "footstep_landing_score": footstep_landing_score,
            "foot_slip_penalty": foot_slip_penalty,
            "foot_crossing_penalty": foot_crossing_penalty,
            "smooth_ctrl_cost": smooth_ctrl_cost,
            "arm_swing_score": arm_swing_score,
            "arm_balance_penalty": arm_balance_penalty,
            "arm_flail_penalty": arm_flail_penalty,
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
