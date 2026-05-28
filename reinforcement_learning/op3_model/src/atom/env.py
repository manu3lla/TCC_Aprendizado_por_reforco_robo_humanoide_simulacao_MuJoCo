import os
import gymnasium as gym
import numpy as np
import mujoco


class AtomEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

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

        obs_dim = self.model.nq + self.model.nv
        act_dim = self.model.nu

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32
        )

        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

        # ============================================================
        # Controle padrão para atuadores de posição
        # ============================================================
        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float32)

        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.default_ctrl[i] = self.initial_qpos[qpos_adr]

        # ============================================================
        # Escala de correção do PPO
        # Aqui deixamos menor para o PPO não destruir a marcha base.
        # ============================================================
        self.action_scale = np.ones(self.model.nu, dtype=np.float32) * 0.10

        if self.model.nu >= 20:
            # Cabeça quase fixa
            self.action_scale[0] = 0.02  # head_pan
            self.action_scale[1] = 0.02  # head_tilt

            # Braços: PPO corrige pouco, marcha base faz o balanço
            self.action_scale[2] = 0.15  # left_shoulder_pitch
            self.action_scale[3] = 0.10  # left_shoulder_roll
            self.action_scale[4] = 0.10  # left_elbow_pitch

            self.action_scale[5] = 0.15  # right_shoulder_pitch
            self.action_scale[6] = 0.10  # right_shoulder_roll
            self.action_scale[7] = 0.10  # right_elbow_pitch

            # Perna direita
            self.action_scale[8] = 0.12    # right_leg_yaw
            self.action_scale[9] = 0.18    # right_leg_roll
            self.action_scale[10] = 0.20   # right_leg_pitch
            self.action_scale[11] = 0.25   # right_knee
            self.action_scale[12] = 0.20   # right_foot_pitch
            self.action_scale[13] = 0.15   # right_foot_roll

            # Perna esquerda
            self.action_scale[14] = 0.12   # left_leg_yaw
            self.action_scale[15] = 0.18   # left_leg_roll
            self.action_scale[16] = 0.20   # left_leg_pitch
            self.action_scale[17] = 0.25   # left_knee
            self.action_scale[18] = 0.20   # left_foot_pitch
            self.action_scale[19] = 0.15   # left_foot_roll

        # ============================================================
        # Parâmetros da marcha base
        # ============================================================
        self.use_gait_reference = True
        self.gait_phase = 0.0

        # Passo mais lento e mais claro
        self.gait_frequency = 0.85

        # Inclinação/transferência de peso para frente
        self.forward_lean_amount = 0.25

        # Levantar bem os joelhos
        self.knee_lift_amount = 0.95

        # Balanço de quadril
        self.hip_swing_amount = 0.45

        # Compensação de tornozelo
        self.ankle_comp_amount = 0.28

        # Balanço dos braços
        self.arm_swing_amount = 0.55

        self.frame_skip = 5
        self.gait_dt = self.model.opt.timestep * self.frame_skip

        self.healthy_z_range = (0.270, 0.300)
        self.keep_alive_weight = 1.0
        self.control_weight = 1e-3
        self.velocity_weight = 3.0
        self.target_distance = 100.0
        self.reach_target_reward = 100.0
        self.knee_flex_weight = 1e-3
        self.feet_up_weight = 1e-3
        self.feet_misalignment_weight = 0.05
        self.max_episode_steps = 1000
        self.max_timestep = self.max_episode_steps
        self.current_step = 0
        self.prev_x = 0.0

        # Tenta encontrar os bodies dos pés
        self.right_foot_body_id = self._find_first_body_id([
            "right_foot",
            "right_ankle",
            "r_foot",
            "r_ankle",
            "right_foot_link"
        ])

        self.left_foot_body_id = self._find_first_body_id([
            "left_foot",
            "left_ankle",
            "l_foot",
            "l_ankle",
            "left_foot_link"
        ])

        print("Default ctrl:", self.default_ctrl)
        print("Action scale:", self.action_scale)
        print("right_foot_body_id:", self.right_foot_body_id)
        print("left_foot_body_id:", self.left_foot_body_id)

        print("\n===== DEBUG DOS ATUADORES =====")
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                i
            )

            joint_id = self.model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id
            )

            print(
                f"Atuador {i}: {act_name} | junta: {joint_name} | "
                f"default_ctrl: {self.default_ctrl[i]:.3f} | "
                f"action_scale: {self.action_scale[i]:.3f} | "
                f"ctrlrange: {self.model.actuator_ctrlrange[i]} | "
                f"ctrllimited: {self.model.actuator_ctrllimited[i]}"
            )
        print("================================\n")

    # ============================================================
    # Observação
    # ============================================================
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ]).astype(np.float32)

    # ============================================================
    # Métricas auxiliares
    # ============================================================
    def _torso_height(self):
        return float(self.data.qpos[2]) if self.model.nq > 2 else 0.0

    def _torso_upright(self):
        if self.model.nq < 7:
            return 1.0

        quat = self.data.qpos[3:7]
        norm = np.linalg.norm(quat)

        if norm < 1e-6:
            return 0.0

        quat = quat / norm

        return abs(float(quat[0]))

    def _is_unstable(self):
        torso_height = self._torso_height()
        upright = self._torso_upright()

        if torso_height < 0.13:
            return True

        if torso_height > 1.2:
            return True

        if upright < 0.40:
            return True

        return False

    def _joint_qpos(self, joint_name):
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if joint_id < 0:
            return 0.0

        qpos_adr = self.model.jnt_qposadr[joint_id]
        return float(self.data.qpos[qpos_adr])

    def _find_first_body_id(self, possible_names):
        for name in possible_names:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name
            )

            if body_id >= 0:
                return body_id

        return -1

    def _body_height_by_id(self, body_id):
        if body_id < 0:
            return 0.0

        return float(self.data.xpos[body_id][2])

    def _foot_geom_ids(self, body_id):
        if body_id < 0:
            return []

        start = int(self.model.body_geomadr[body_id])
        count = int(self.model.body_geomnum[body_id])
        return list(range(start, start + count))

    def _check_contact_body(self, body_id):
        if body_id < 0:
            return False

        foot_geom_ids = set(self._foot_geom_ids(body_id))
        ground_geom_id = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 in foot_geom_ids and contact.geom2 == ground_geom_id) or (
                contact.geom2 in foot_geom_ids and contact.geom1 == ground_geom_id
            ):
                return True

        return False

    def _foot_alignment(self, body_id):
        if body_id < 0:
            return 0.0

        rot = self.data.xmat[body_id].reshape(3, 3)
        return float(rot[2, 2])

    def mass_center_position(self):
        mass = np.expand_dims(self.model.body_mass, axis=1)
        xpos = self.data.xipos
        com = np.sum(mass * xpos, axis=0) / np.sum(mass)
        return com[0:3].copy()

    @staticmethod
    def mass_center_velocity(before, after, delta_time):
        return (after - before) / delta_time

    def is_healthy(self, z_pos):
        min_z, max_z = self.healthy_z_range
        return min_z < z_pos < max_z

    def control_effort(self):
        return np.sum(np.square(self.data.ctrl))

    def knees_angular_velocity(self):
        return abs(self.data.qvel[10]) + abs(self.data.qvel[16])

    def feet_height(self):
        left_height = self.data.xpos[self.left_foot_body_id][2] if self.left_foot_body_id >= 0 else 0.0
        right_height = self.data.xpos[self.right_foot_body_id][2] if self.right_foot_body_id >= 0 else 0.0
        return left_height + right_height

    def feets_ground_misalignment(self):
        left_alignment = self._foot_alignment(self.left_foot_body_id)
        right_alignment = self._foot_alignment(self.right_foot_body_id)

        is_left_foot_on_ground = self._check_contact_body(self.left_foot_body_id)
        is_right_foot_on_ground = self._check_contact_body(self.right_foot_body_id)

        left_misalignment = (1 - left_alignment) if is_left_foot_on_ground else 0
        right_misalignment = (1 - right_alignment) if is_right_foot_on_ground else 0

        return left_misalignment + right_misalignment

    def distance_from_origin(self):
        return np.linalg.norm(self.data.qpos[0:2], ord=2)

    def _get_reward(self, velocity, position_before, position_after):
        health_reward = self.keep_alive_weight * self.is_healthy(position_after[2])
        forward_reward = self.velocity_weight * velocity[0]
        info_knee_angvel = self.knees_angular_velocity()
        info_feet_height = self.feet_height()
        info_control = self.control_effort()
        info_feet_misalign = self.feets_ground_misalignment()

        knee_flex_reward = self.knee_flex_weight * info_knee_angvel
        feet_up_reward = self.feet_up_weight * info_feet_height
        control_penalty = self.control_weight * info_control
        feet_misalign_penalty = self.feet_misalignment_weight * info_feet_misalign

        reward = (
            health_reward
            + forward_reward
            + knee_flex_reward
            + feet_up_reward
            - feet_misalign_penalty
            - control_penalty
        )

        if self.data.qpos[0] >= self.target_distance:
            reward = self.reach_target_reward

        if self.current_step >= self.max_timestep:
            reward = 2.00

        reward_info = {
            "position_x": position_after[0],
            "position_y": position_after[1],
            "position_z": position_after[2],
            "velocity_x": velocity[0],
            "velocity_y": velocity[1],
            "velocity_z": velocity[2],
            "distance_from_origin": self.distance_from_origin(),
            "knee_angvel": info_knee_angvel,
            "feet_height": info_feet_height,
            "control_effort": info_control,
            "feet_misalignment": info_feet_misalign,
            "timestep": self.current_step,
            "health_reward": health_reward,
            "forward_reward": forward_reward,
            "knee_flex_reward": knee_flex_reward,
            "feet_up_reward": feet_up_reward,
            "control_penalty": control_penalty,
            "feet_misalign_penalty": feet_misalign_penalty,
        }

        return reward, reward_info

    # ============================================================
    # Referência de marcha humanoide
    # ============================================================
    def _gait_reference(self):
        ref = np.zeros(self.model.nu, dtype=np.float32)

        if self.model.nu < 20:
            return ref

        phase = self.gait_phase

        s = np.sin(phase)
        c = np.cos(phase)

        s_right = s
        s_left = -s

        swing_right = max(s_right, 0.0)
        swing_left = max(s_left, 0.0)

        stance_right = max(-s_right, 0.0)
        stance_left = max(-s_left, 0.0)

        # ========================================================
        # Braços em oposição às pernas
        # ========================================================
        ref[2] = self.arm_swing_amount * s       # left_shoulder_pitch
        ref[5] = -self.arm_swing_amount * s      # right_shoulder_pitch

        # Cotovelos levemente flexionados
        ref[4] = 0.14 + 0.08 * swing_left
        ref[7] = 0.14 + 0.08 * swing_right

        # ========================================================
        # Quadril alternado
        # ========================================================
        ref[10] = self.hip_swing_amount * s_right
        ref[16] = self.hip_swing_amount * s_left

        # ========================================================
        # Joelho: levanta a perna em swing
        # ========================================================
        ref[11] = self.knee_lift_amount * swing_right
        ref[17] = self.knee_lift_amount * swing_left

        # Pequena extensão na perna de apoio para não agachar tanto
        ref[11] += -0.10 * stance_right
        ref[17] += -0.10 * stance_left

        # ========================================================
        # Tornozelo: ajuda a tirar o pé do chão e compensar
        # ========================================================
        ref[12] = -self.ankle_comp_amount * swing_right
        ref[18] = -self.ankle_comp_amount * swing_left

        # Apoio do pé no chão
        ref[12] += 0.08 * stance_right
        ref[18] += 0.08 * stance_left

        # ========================================================
        # Transferência lateral de peso
        # ========================================================
        ref[9] = 0.10 * c
        ref[15] = -0.10 * c

        return ref

    def _forward_lean_reference(self):
        """
        Pequeno viés para jogar o peso para frente.
        Se ele piorar e cair mais para trás, inverter os sinais marcados abaixo.
        """
        ref = np.zeros(self.model.nu, dtype=np.float32)

        if self.model.nu < 20:
            return ref

        lean = self.forward_lean_amount

        # SINAL A:
        # Se ele continuar caindo para trás, troque lean por -lean aqui.
        ref[10] += lean
        ref[16] += lean

        # SINAL A:
        # Se inverter acima, inverta estes também.
        ref[12] += -0.10
        ref[18] += -0.10

        return ref

    # ============================================================
    # Reset
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.gait_phase = 0.0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()

        if self.model.nu > 0:
            self.data.ctrl[:] = self.default_ctrl.copy()

        # Ruído inicial pequeno
        if self.model.nq > 7:
            self.data.qpos[7:] += np.random.uniform(
                -0.008,
                0.008,
                size=self.model.nq - 7
            )

        if self.model.nv > 6:
            self.data.qvel[6:] += np.random.uniform(
                -0.008,
                0.008,
                size=self.model.nv - 6
            )

        mujoco.mj_forward(self.model, self.data)

        self.prev_x = float(self.data.qpos[0]) if self.model.nq > 0 else 0.0

        obs = self._get_obs()
        info = {}

        return obs, info

    # ============================================================
    # Step
    # ============================================================
    def step(self, action):
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self.model.nu > 0:
            gait_ref = (
                self._gait_reference()
                if self.use_gait_reference
                else np.zeros(self.model.nu, dtype=np.float32)
            )

            lean_ref = self._forward_lean_reference()

            target_ctrl = (
                self.default_ctrl
                + gait_ref
                + lean_ref
                + self.action_scale * action
            )

            # Só respeita ctrlrange se o XML tiver ctrllimited ativo.
            for i in range(self.model.nu):
                if self.model.actuator_ctrllimited[i]:
                    low = self.model.actuator_ctrlrange[i, 0]
                    high = self.model.actuator_ctrlrange[i, 1]
                    target_ctrl[i] = np.clip(target_ctrl[i], low, high)

            self.data.ctrl[:] = target_ctrl

        mc_before = self.mass_center_position()

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        mc_after = self.mass_center_position()

        velocity = self.mass_center_velocity(
            mc_before,
            mc_after,
            self.model.opt.timestep * self.frame_skip,
        )

        self.gait_phase += 2.0 * np.pi * self.gait_frequency * self.gait_dt

        obs = self._get_obs()

        torso_height = self._torso_height()
        upright = self._torso_upright()

        current_x = float(self.data.qpos[0]) if self.model.nq > 0 else 0.0
        delta_x = current_x - self.prev_x
        self.prev_x = current_x

        right_knee = self._joint_qpos("right_knee")
        left_knee = self._joint_qpos("left_knee")
        right_leg_pitch = self._joint_qpos("right_leg_pitch")
        left_leg_pitch = self._joint_qpos("left_leg_pitch")

        right_foot_h = self._body_height_by_id(self.right_foot_body_id)
        left_foot_h = self._body_height_by_id(self.left_foot_body_id)

        reward, reward_info = self._get_reward(
            velocity,
            mc_before,
            mc_after,
        )

        terminated = self._is_unstable()
        truncated = self.current_step >= self.max_episode_steps

        if terminated:
            reward -= 2.5

        if self.current_step % 200 == 0:
            print(
                f"[DEBUG] step={self.current_step} | "
                f"x={current_x:.4f} | dx={delta_x:.6f} | "
                f"h={torso_height:.3f} | upright={upright:.3f} | "
                f"rknee={right_knee:.3f} | lknee={left_knee:.3f} | "
                f"rfoot_h={right_foot_h:.3f} | lfoot_h={left_foot_h:.3f} | "
                f"phase={self.gait_phase:.2f} | "
                f"reward={reward:.3f}"
            )

        info = {
            "torso_height": torso_height,
            "upright": upright,
            "x_position": current_x,
            "delta_x": delta_x,
            "right_knee": right_knee,
            "left_knee": left_knee,
            "right_leg_pitch": right_leg_pitch,
            "left_leg_pitch": left_leg_pitch,
            "right_foot_h": right_foot_h,
            "left_foot_h": left_foot_h,
            "gait_phase": self.gait_phase,
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass