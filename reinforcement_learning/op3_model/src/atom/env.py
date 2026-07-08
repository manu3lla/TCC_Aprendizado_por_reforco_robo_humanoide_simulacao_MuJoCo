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
        target_velocity=0.25,
        target_distance=50.0,
        forward_direction=-1.0,
        torso_forward_axis=-1.0,
        debug_reward=False,
        reward_weights=None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.viewer = None
        self.debug_reward = debug_reward

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

        # Espaço de observação: posições + velocidades
        obs_dim = self.model.nq + self.model.nv

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

        # Escala da ação do PPO
        self.action_scale = np.ones(self.model.nu, dtype=np.float32) * 0.15

        if self.model.nu >= 20:
            self.action_scale[0] = 0.03  # head_pan
            self.action_scale[1] = 0.03  # head_tilt

            self.action_scale[2:8] = 0.10

            self.action_scale[8] = 0.18
            self.action_scale[9] = 0.25
            self.action_scale[10] = 0.30
            self.action_scale[11] = 0.35
            self.action_scale[12] = 0.30
            self.action_scale[13] = 0.22

            self.action_scale[14] = 0.18
            self.action_scale[15] = 0.25
            self.action_scale[16] = 0.30
            self.action_scale[17] = 0.35
            self.action_scale[18] = 0.30
            self.action_scale[19] = 0.22

        self.frame_skip = 5
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.max_episode_steps = 1000
        self.current_step = 0

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_torso_height = 0.0
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_left_contact = False
        self.prev_right_contact = False

        self.chest_body_id = self._body_id("chest")
        self.left_foot_body_id = self._body_id("left_foot")
        self.right_foot_body_id = self._body_id("right_foot")
        self.floor_geom_id = self._geom_id("floor")

        default_reward_weights = {
            "forward_velocity": 1.00,
            "target_progress": 1.20,
            "velocity_tracking": 0.20,
            "alive": 0.02,
            "upright": 0.08,
            "height": 0.08,
            "heading": 0.03,
            "backward": 1.00,
            "roll": 0.04,
            "pitch": 0.04,
            "yaw": 0.03,
            "knee_extension": 0.12,
            "lateral_drift": 0.25,
            "lateral_velocity": 0.04,
            "support": 0.02,
            "single_support": 0.04,
            "double_support_drag": 0.04,
            "step_length": 0.08,
            "foot_separation": 0.03,
            "foot_clearance": 0.035,
            "airborne": 0.35,
            "torso_vertical_velocity": 0.08,
            "excessive_foot_clearance": 0.12,
            "stance_foot_orientation": 0.10,
            "ankle_pitch": 0.04,
            "ankle_roll": 0.10,
            "arm_motion": 0.015,
            "control": 0.006,
            "smooth_action": 0.015,
            "joint_velocity": 0.0015,
            "fall": 10.0,
        }

        if reward_weights is None:
            reward_weights = {}

        self.reward_weights = {
            **default_reward_weights,
            **reward_weights,
        }

        self.target_velocity = float(target_velocity)
        self.target_distance = float(target_distance)
        self.velocity_tracking_sigma = 0.20
        self.target_step_length = 0.08
        self.foot_clearance_target = 0.035
        self.knee_extension_threshold = 0.80

        # No XML atual, a frente visual do Atom aponta para o -X do mundo
        # quando ele esta na pose inicial. O eixo X local do body "chest"
        # aponta para o lado oposto da frente visual, por isso os sinais
        # padrao sao ambos -1.0.
        self.forward_direction = float(forward_direction)
        self.torso_forward_axis = float(torso_forward_axis)
        self.target_x = float(
            self.initial_qpos[0] + self.forward_direction * self.target_distance
        )

        self.target_height = float(self.initial_qpos[2]) if self.model.nq > 2 else 0.50

        mujoco.mj_forward(self.model, self.data)

        self.initial_left_foot_pos = self._body_pos(self.left_foot_body_id).copy()
        self.initial_right_foot_pos = self._body_pos(self.right_foot_body_id).copy()
        self.initial_left_foot_matrix = self._body_matrix(
            self.left_foot_body_id
        ).copy()
        self.initial_right_foot_matrix = self._body_matrix(
            self.right_foot_body_id
        ).copy()

        self.initial_foot_separation = abs(
            self.initial_right_foot_pos[1] - self.initial_left_foot_pos[1]
        )

        self.initial_target_distance = self._distance_to_target(
            float(self.initial_qpos[0])
        )

    def _body_id(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        if body_id < 0:
            raise ValueError(f"Body nao encontrado no modelo MuJoCo: {name}")

        return body_id

    def _geom_id(self, name):
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        if geom_id < 0:
            raise ValueError(f"Geom nao encontrado no modelo MuJoCo: {name}")

        return geom_id

    def _body_pos(self, body_id):
        return self.data.xpos[body_id]

    def _body_matrix(self, body_id):
        return self.data.xmat[body_id].reshape(3, 3)

    def _body_orientation_error(self, body_id, initial_matrix):
        current_matrix = self._body_matrix(body_id)
        relative_rotation = current_matrix @ initial_matrix.T
        cos_angle = (np.trace(relative_rotation) - 1.0) / 2.0

        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _distance_to_target(self, x_position):
        signed_distance = self.forward_direction * (self.target_x - x_position)
        return max(float(signed_distance), 0.0)

    def _torso_euler(self):
        torso_matrix = self._body_matrix(self.chest_body_id)

        sy = np.sqrt(
            torso_matrix[0, 0] * torso_matrix[0, 0]
            + torso_matrix[1, 0] * torso_matrix[1, 0]
        )

        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(torso_matrix[2, 1], torso_matrix[2, 2])
            pitch = np.arctan2(-torso_matrix[2, 0], sy)
            yaw = np.arctan2(torso_matrix[1, 0], torso_matrix[0, 0])
        else:
            roll = np.arctan2(-torso_matrix[1, 2], torso_matrix[1, 1])
            pitch = np.arctan2(-torso_matrix[2, 0], sy)
            yaw = 0.0

        return float(roll), float(pitch), float(yaw)

    def _joint_qpos_by_names(self, names):
        for name in names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )

            if joint_id >= 0:
                qpos_adr = self.model.jnt_qposadr[joint_id]
                return float(self.data.qpos[qpos_adr])

        return None

    def _knee_extension_penalty_factor(self):
        right_knee = self._joint_qpos_by_names([
            "right_knee",
            "right_knee_pitch",
            "right_leg_knee",
            "right_knee_joint",
        ])

        left_knee = self._joint_qpos_by_names([
            "left_knee",
            "left_knee_pitch",
            "left_leg_knee",
            "left_knee_joint",
        ])

        if right_knee is None and self.model.nu > 11:
            joint_id = self.model.actuator_trnid[11, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            right_knee = float(self.data.qpos[qpos_adr])

        if left_knee is None and self.model.nu > 17:
            joint_id = self.model.actuator_trnid[17, 0]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            left_knee = float(self.data.qpos[qpos_adr])

        knees = [knee for knee in (right_knee, left_knee) if knee is not None]

        if len(knees) != 2:
            return 0.0

        extended_count = sum(
            abs(knee) > self.knee_extension_threshold
            for knee in knees
        )

        knee_factor = extended_count / 2.0

        if knee_factor < 1.0:
            return 0.0

        return float(knee_factor)

    def _ankle_angle_magnitudes(self):
        right_pitch = self._joint_qpos_by_names([
            "right_foot_pitch",
            "right_ankle_pitch",
        ])
        left_pitch = self._joint_qpos_by_names([
            "left_foot_pitch",
            "left_ankle_pitch",
        ])
        right_roll = self._joint_qpos_by_names([
            "right_foot_roll",
            "right_ankle_roll",
        ])
        left_roll = self._joint_qpos_by_names([
            "left_foot_roll",
            "left_ankle_roll",
        ])

        pitch_values = [
            abs(value)
            for value in (right_pitch, left_pitch)
            if value is not None
        ]
        roll_values = [
            abs(value)
            for value in (right_roll, left_roll)
            if value is not None
        ]

        pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0
        roll_mean = float(np.mean(roll_values)) if roll_values else 0.0

        return pitch_mean, roll_mean

    def _foot_has_floor_contact(self, foot_body_id):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 == self.floor_geom_id:
                other_body_id = self.model.geom_bodyid[geom2]
            elif geom2 == self.floor_geom_id:
                other_body_id = self.model.geom_bodyid[geom1]
            else:
                continue

            if other_body_id == foot_body_id:
                return True

        return False

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])

        return obs.astype(np.float32)

    def _torso_height(self):
        if self.chest_body_id >= 0:
            return float(self._body_pos(self.chest_body_id)[2])

        return 0.0

    def _torso_upright(self):
        torso_matrix = self._body_matrix(self.chest_body_id)
        torso_z_axis = torso_matrix[:, 2]

        return float(np.clip(torso_z_axis[2], 0.0, 1.0))

    def _torso_heading(self):
        torso_matrix = self._body_matrix(self.chest_body_id)
        torso_x_axis = torso_matrix[:, 0]
        heading = self.forward_direction * self.torso_forward_axis * torso_x_axis[0]

        return float(np.clip(heading, 0.0, 1.0))

    def _has_fallen(self):
        torso_height = self._torso_height()
        upright = self._torso_upright()

        if torso_height < 0.28:
            return True

        if torso_height > 1.20:
            return True

        if upright < 0.45:
            return True

        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.initial_qpos.copy()
        self.data.qvel[:] = self.initial_qvel.copy()

        if self.model.nu > 0:
            self.data.ctrl[:] = self.default_ctrl.copy()

        mujoco.mj_forward(self.model, self.data)

        self.prev_x = float(self.data.qpos[0])
        self.prev_y = float(self.data.qpos[1])
        self.prev_torso_height = self._torso_height()
        self.prev_action[:] = 0.0

        self.prev_left_contact = self._foot_has_floor_contact(self.left_foot_body_id)
        self.prev_right_contact = self._foot_has_floor_contact(self.right_foot_body_id)

        obs = self._get_obs()
        left_foot_pos = self._body_pos(self.left_foot_body_id)
        right_foot_pos = self._body_pos(self.right_foot_body_id)

        info = {
            "x_position": self.prev_x,
            "y_position": self.prev_y,
            "target_x": self.target_x,
            "distance_to_target": self._distance_to_target(self.prev_x),
            "torso_height": self.prev_torso_height,
            "torso_vertical_velocity": 0.0,
            "upright": self._torso_upright(),
            "heading": self._torso_heading(),
            "left_foot_height": float(left_foot_pos[2]),
            "right_foot_height": float(right_foot_pos[2]),
            "left_foot_contact": self.prev_left_contact,
            "right_foot_contact": self.prev_right_contact,
        }

        return obs, info

    def step(self, action):
        self.current_step += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        target_ctrl = self.default_ctrl + self.action_scale * action

        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                low = self.model.actuator_ctrlrange[i, 0]
                high = self.model.actuator_ctrlrange[i, 1]
                target_ctrl[i] = np.clip(target_ctrl[i], low, high)

        self.data.ctrl[:] = target_ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            if self.render_mode == "human":
                self.render()

        current_x = float(self.data.qpos[0])
        current_y = float(self.data.qpos[1])

        delta_x = current_x - self.prev_x
        delta_y = current_y - self.prev_y

        previous_target_distance = self._distance_to_target(self.prev_x)

        self.prev_x = current_x
        self.prev_y = current_y

        forward_delta = self.forward_direction * delta_x
        forward_velocity = forward_delta / self.dt
        lateral_velocity = delta_y / self.dt

        current_target_distance = self._distance_to_target(current_x)
        target_progress = previous_target_distance - current_target_distance
        target_progress_velocity = target_progress / self.dt

        torso_height = self._torso_height()
        torso_vertical_velocity = (
            torso_height - self.prev_torso_height
        ) / self.dt
        upright = self._torso_upright()
        heading = self._torso_heading()
        roll, pitch, yaw = self._torso_euler()

        left_foot_pos = self._body_pos(self.left_foot_body_id)
        right_foot_pos = self._body_pos(self.right_foot_body_id)

        left_contact = self._foot_has_floor_contact(self.left_foot_body_id)
        right_contact = self._foot_has_floor_contact(self.right_foot_body_id)

        has_support = left_contact or right_contact
        single_support = left_contact != right_contact

        weights = self.reward_weights

        clipped_forward_velocity = float(
            np.clip(
                forward_velocity,
                -0.5,
                max(self.target_velocity * 2.0, 0.1),
            )
        )

        forward_reward = weights["forward_velocity"] * clipped_forward_velocity

        target_progress_reward = weights["target_progress"] * float(
            np.clip(
                target_progress_velocity,
                -0.5,
                max(self.target_velocity * 2.0, 0.1),
            )
        )

        backward_penalty = weights["backward"] * max(-forward_velocity, 0.0)

        velocity_error = (
            (forward_velocity - self.target_velocity)
            / self.velocity_tracking_sigma
        )

        velocity_tracking_reward = weights["velocity_tracking"] * float(
            np.exp(-(velocity_error ** 2))
        )

        fallen = self._has_fallen()
        alive_reward = weights["alive"] if not fallen else 0.0

        height_error = abs(torso_height - self.target_height)

        height_reward = weights["height"] * float(
            np.exp(-((height_error / 0.08) ** 2))
        )

        upright_reward = weights["upright"] * upright
        heading_reward = weights["heading"] * heading

        roll_penalty = weights["roll"] * abs(roll)
        pitch_penalty = weights["pitch"] * abs(pitch)
        yaw_penalty = weights["yaw"] * abs(yaw)

        knee_extension_penalty = (
            weights["knee_extension"] * self._knee_extension_penalty_factor()
        )

        support_reward = weights["support"] if has_support else -weights["support"]
        airborne_penalty = weights["airborne"] if not has_support else 0.0

        torso_vertical_velocity_penalty = (
            weights["torso_vertical_velocity"] * abs(torso_vertical_velocity)
        )

        single_support_reward = 0.0
        if single_support and forward_velocity > 0.02:
            single_support_reward = weights["single_support"]

        double_support_drag_penalty = 0.0
        if left_contact and right_contact and forward_velocity > 0.03:
            double_support_drag_penalty = weights["double_support_drag"]

        foot_separation = abs(float(right_foot_pos[1] - left_foot_pos[1]))

        foot_separation_error = abs(
            foot_separation - self.initial_foot_separation
        ) / max(self.initial_foot_separation, 1e-6)

        foot_separation_penalty = weights["foot_separation"] * min(
            foot_separation_error,
            2.0,
        )
        foot_separation_penalty = float(foot_separation_penalty)

        step_length = abs(float(left_foot_pos[0] - right_foot_pos[0]))

        step_length_reward = 0.0
        if forward_velocity > 0.02:
            step_length_reward = float(
                weights["step_length"] * min(
                    step_length / self.target_step_length,
                    1.0,
                )
            )

        left_clearance = max(
            float(left_foot_pos[2] - self.initial_left_foot_pos[2]),
            0.0,
        )

        right_clearance = max(
            float(right_foot_pos[2] - self.initial_right_foot_pos[2]),
            0.0,
        )

        foot_clearance_reward = 0.0
        excessive_foot_clearance_penalty = 0.0

        if forward_velocity > 0.02:
            if left_contact and not right_contact:
                foot_clearance_reward = float(
                    weights["foot_clearance"] * min(
                        right_clearance / self.foot_clearance_target,
                        1.0,
                    )
                )
            elif right_contact and not left_contact:
                foot_clearance_reward = float(
                    weights["foot_clearance"] * min(
                        left_clearance / self.foot_clearance_target,
                        1.0,
                    )
                )

        if not has_support:
            excessive_foot_clearance_penalty = (
                weights["excessive_foot_clearance"]
                * min(
                    max(left_clearance, right_clearance)
                    / self.foot_clearance_target,
                    2.0,
                )
            )

        left_foot_orientation_error = self._body_orientation_error(
            self.left_foot_body_id,
            self.initial_left_foot_matrix,
        )
        right_foot_orientation_error = self._body_orientation_error(
            self.right_foot_body_id,
            self.initial_right_foot_matrix,
        )

        stance_orientation_errors = []
        if left_contact:
            stance_orientation_errors.append(left_foot_orientation_error)
        if right_contact:
            stance_orientation_errors.append(right_foot_orientation_error)

        if stance_orientation_errors:
            stance_foot_orientation_penalty = (
                weights["stance_foot_orientation"]
                * float(np.mean(stance_orientation_errors))
            )
        else:
            stance_foot_orientation_penalty = 0.0

        ankle_pitch_magnitude, ankle_roll_magnitude = self._ankle_angle_magnitudes()
        ankle_pitch_penalty = weights["ankle_pitch"] * ankle_pitch_magnitude
        ankle_roll_penalty = weights["ankle_roll"] * ankle_roll_magnitude

        lateral_drift_penalty = weights["lateral_drift"] * abs(current_y)
        lateral_velocity_penalty = weights["lateral_velocity"] * abs(lateral_velocity)

        control_penalty = weights["control"] * float(np.mean(np.square(action)))

        smooth_action_penalty = weights["smooth_action"] * float(
            np.mean(np.square(action - self.prev_action))
        )

        arm_action = (
            action[2:8]
            if self.model.nu >= 8
            else np.array([], dtype=np.float32)
        )

        if arm_action.size > 0:
            arm_motion_penalty = weights["arm_motion"] * float(
                np.mean(np.square(arm_action))
            )
        else:
            arm_motion_penalty = 0.0

        joint_velocity = self.data.qvel[6:] if self.model.nv > 6 else self.data.qvel

        joint_velocity_penalty = weights["joint_velocity"] * float(
            np.mean(np.square(np.clip(joint_velocity, -10.0, 10.0)))
        )

        reward = (
            forward_reward
            + target_progress_reward
            + velocity_tracking_reward
            + alive_reward
            + height_reward
            + upright_reward
            + heading_reward
            + support_reward
            + single_support_reward
            + step_length_reward
            + foot_clearance_reward
            - backward_penalty
            - roll_penalty #OLHAR
            - pitch_penalty
            - yaw_penalty
            - knee_extension_penalty
            - airborne_penalty
            - torso_vertical_velocity_penalty
            - excessive_foot_clearance_penalty
            - stance_foot_orientation_penalty
            - ankle_pitch_penalty
            - ankle_roll_penalty
            - lateral_drift_penalty
            - lateral_velocity_penalty
            - double_support_drag_penalty
            - foot_separation_penalty
            - control_penalty
            - smooth_action_penalty
            - arm_motion_penalty
            - joint_velocity_penalty
        )

        reached_target = current_target_distance <= 0.05
        terminated = fallen or reached_target

        if terminated:
            if fallen:
                reward -= weights["fall"]
            else:
                reward += weights["alive"] * self.max_episode_steps

        reward = float(reward)
        truncated = self.current_step >= self.max_episode_steps

        if self.debug_reward and self.current_step % 50 == 0:
            print(
                "\n[DEBUG REWARD]",
                f"step={self.current_step}",
                f"x={current_x:.5f}",
                f"y={current_y:.5f}",
                f"target_distance={current_target_distance:.5f}",
                f"delta_x={delta_x:.5f}",
                f"v_forward={forward_velocity:.5f}",
                f"forward_reward={forward_reward:.5f}",
                f"target_progress_reward={target_progress_reward:.5f}",
                f"torso_height={torso_height:.5f}",
                f"torso_vz={torso_vertical_velocity:.5f}",
                f"upright={upright:.5f}",
                f"heading={heading:.5f}",
                f"roll={roll:.5f}",
                f"pitch={pitch:.5f}",
                f"yaw={yaw:.5f}",
                f"left_contact={left_contact}",
                f"right_contact={right_contact}",
                f"reward={reward:.5f}",
            )

        obs = self._get_obs()

        info = {
            "x_position": current_x,
            "y_position": current_y,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "forward_delta": forward_delta,
            "forward_velocity": forward_velocity,
            "target_x": self.target_x,
            "distance_to_target": current_target_distance,
            "target_progress": target_progress,
            "target_progress_velocity": target_progress_velocity,
            "lateral_velocity": lateral_velocity,
            "forward_direction": self.forward_direction,
            "torso_height": torso_height,
            "torso_vertical_velocity": torso_vertical_velocity,
            "target_height": self.target_height,
            "upright": upright,
            "heading": heading,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "left_foot_height": float(left_foot_pos[2]),
            "right_foot_height": float(right_foot_pos[2]),
            "left_foot_orientation_error": left_foot_orientation_error,
            "right_foot_orientation_error": right_foot_orientation_error,
            "left_foot_contact": left_contact,
            "right_foot_contact": right_contact,
            "single_support": single_support,
            "foot_separation": foot_separation,
            "step_length": step_length,
            "forward_reward": forward_reward,
            "target_progress_reward": target_progress_reward,
            "velocity_tracking_reward": velocity_tracking_reward,
            "alive_reward": alive_reward,
            "height_reward": height_reward,
            "upright_reward": upright_reward,
            "heading_reward": heading_reward,
            "backward_penalty": backward_penalty,
            "roll_penalty": roll_penalty,
            "pitch_penalty": pitch_penalty,
            "yaw_penalty": yaw_penalty,
            "knee_extension_penalty": knee_extension_penalty,
            "airborne_penalty": airborne_penalty,
            "torso_vertical_velocity_penalty": torso_vertical_velocity_penalty,
            "excessive_foot_clearance_penalty": excessive_foot_clearance_penalty,
            "stance_foot_orientation_penalty": stance_foot_orientation_penalty,
            "ankle_pitch_penalty": ankle_pitch_penalty,
            "ankle_roll_penalty": ankle_roll_penalty,
            "ankle_pitch_magnitude": ankle_pitch_magnitude,
            "ankle_roll_magnitude": ankle_roll_magnitude,
            "support_reward": support_reward,
            "single_support_reward": single_support_reward,
            "step_length_reward": step_length_reward,
            "foot_clearance_reward": foot_clearance_reward,
            "lateral_drift_penalty": lateral_drift_penalty,
            "lateral_velocity_penalty": lateral_velocity_penalty,
            "double_support_drag_penalty": double_support_drag_penalty,
            "foot_separation_penalty": foot_separation_penalty,
            "control_penalty": control_penalty,
            "smooth_action_penalty": smooth_action_penalty,
            "arm_motion_penalty": arm_motion_penalty,
            "joint_velocity_penalty": joint_velocity_penalty,
            "reached_target": reached_target,
            "reward": reward,
            "step": self.current_step,
        }

        self.prev_action = action.copy()
        self.prev_torso_height = torso_height
        self.prev_left_contact = left_contact
        self.prev_right_contact = right_contact

        return obs, reward, terminated, truncated, info

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
