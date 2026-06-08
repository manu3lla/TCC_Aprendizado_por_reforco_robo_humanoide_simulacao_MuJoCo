import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer


class AtomEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode=None):
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
            # Cabeça quase fixa
            self.action_scale[0] = 0.03  # head_pan
            self.action_scale[1] = 0.03  # head_tilt

            # Braços com pouca liberdade
            self.action_scale[2:8] = 0.10

            # Perna direita
            self.action_scale[8] = 0.18    # right_leg_yaw
            self.action_scale[9] = 0.25    # right_leg_roll
            self.action_scale[10] = 0.30   # right_leg_pitch
            self.action_scale[11] = 0.35   # right_knee
            self.action_scale[12] = 0.30   # right_foot_pitch
            self.action_scale[13] = 0.22   # right_foot_roll

            # Perna esquerda
            self.action_scale[14] = 0.18   # left_leg_yaw
            self.action_scale[15] = 0.25   # left_leg_roll
            self.action_scale[16] = 0.30   # left_leg_pitch
            self.action_scale[17] = 0.35   # left_knee
            self.action_scale[18] = 0.30   # left_foot_pitch
            self.action_scale[19] = 0.22   # left_foot_roll

        # Configurações do episódio
        self.frame_skip = 5
        self.max_episode_steps = 1000
        self.current_step = 0

        # Guarda posição anterior no eixo X
        self.prev_x = 0.0

        # Pesos da recompensa
        self.forward_weight = 30.0
        self.alive_weight = 0.03
        self.stability_weight = 0.12
        self.control_weight = 0.001
        self.fall_penalty = 5.0

        # Direção considerada como "para frente"
        #
        # Se andar para frente no MuJoCo aumenta o X,  1.0.
        # Se andar para frente no MuJoCo diminui o X,  -1.0.
        self.forward_direction = 1.0

        # Altura aproximada desejada do tronco.
        # Pelo XML, o chest começa em z = 0.507.
        self.target_height = float(self.initial_qpos[2]) if self.model.nq > 2 else 0.50

        # Debug dos atuadores
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
                f"Atuador {i}: {act_name} | "
                f"Junta: {joint_name} | "
                f"default_ctrl: {self.default_ctrl[i]:.4f} | "
                f"action_scale: {self.action_scale[i]:.4f} | "
                f"ctrlrange: {self.model.actuator_ctrlrange[i]} | "
                f"ctrllimited: {self.model.actuator_ctrllimited[i]}"
            )
        print("================================\n")

    # Observação
    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])

        return obs.astype(np.float32)

    # Altura do tronco
    def _torso_height(self):
        if self.model.nq > 2:
            return float(self.data.qpos[2])

        return 0.0

    # Orientação aproximada do tronco
    def _torso_upright(self):
        if self.model.nq < 7:
            return 1.0

        quat = self.data.qpos[3:7]
        norm = np.linalg.norm(quat)

        if norm < 1e-6:
            return 0.0

        quat = quat / norm

        # Quanto mais perto de 1, mais próximo da orientação inicial.
        return abs(float(quat[0]))

    # Critério simples de queda
    def _has_fallen(self):
        torso_height = self._torso_height()
        upright = self._torso_upright()

        if torso_height < 0.20:
            return True

        if torso_height > 1.20:
            return True

        if upright < 0.35:
            return True

        return False

    # Reset do ambiente
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

        obs = self._get_obs()

        info = {
            "x_position": self.prev_x,
            "torso_height": self._torso_height(),
            "upright": self._torso_upright(),
        }

        return obs, info

    # Step do ambiente
    def step(self, action):
        self.current_step += 1

        # Garante que a ação fique entre -1 e 1
        # Normalização das ações para rede neural
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Aplica ação nos atuadores
        target_ctrl = self.default_ctrl + self.action_scale * action

        # Respeita limites do XML, se existirem
        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                low = self.model.actuator_ctrlrange[i, 0]
                high = self.model.actuator_ctrlrange[i, 1]
                target_ctrl[i] = np.clip(target_ctrl[i], low, high)

        self.data.ctrl[:] = target_ctrl

        # Roda a física
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

            if self.render_mode == "human":
                self.render()

        # Calcula deslocamento no eixo X
        current_x = float(self.data.qpos[0])
        delta_x = current_x - self.prev_x
        self.prev_x = current_x

        # Converte o deslocamento bruto em deslocamento para frente
        forward_delta = self.forward_direction * delta_x

        torso_height = self._torso_height()
        upright = self._torso_upright()
        
        # Recompensa principal: avanço no eixo X
        forward_reward = self.forward_weight * forward_delta

    
        if self.current_step % 50 == 0:
            print(
                "\n[DEBUG DIREÇÃO]",
                f"step={self.current_step}",
                f"x={current_x:.5f}",
                f"delta_x={delta_x:.5f}",
                f"forward_direction={self.forward_direction:.1f}",
                f"forward_delta={forward_delta:.5f}",
                f"forward_reward={forward_reward:.5f}",
                f"torso_height={torso_height:.5f}",
                f"upright={upright:.5f}"
            )

        # Recompensa por continuar vivo/em pé
        alive_reward = self.alive_weight

        # Recompensa de estabilidade
        #
        # Quanto mais perto da altura inicial e mais ereto,
        # maior a recompensa de estabilidade.
        height_error = abs(torso_height - self.target_height)

        height_stability = max(1.0 - height_error, 0.0)
        upright_stability = upright

        stability_reward = self.stability_weight * (
            height_stability + upright_stability
        )

        # Penalidade para evitar ações muito fortes/bruscas
        control_penalty = self.control_weight * float(
            np.sum(np.square(action))
        )

        reward = (
            forward_reward
            + alive_reward
            + stability_reward
            - control_penalty
        )

        # Penaliza queda
        terminated = self._has_fallen()

        if terminated:
            reward -= self.fall_penalty

        truncated = self.current_step >= self.max_episode_steps

        obs = self._get_obs()

        info = {
            "x_position": current_x,
            "delta_x": delta_x,
            "forward_delta": forward_delta,
            "forward_direction": self.forward_direction,
            "torso_height": torso_height,
            "target_height": self.target_height,
            "upright": upright,
            "forward_reward": forward_reward,
            "alive_reward": alive_reward,
            "stability_reward": stability_reward,
            "control_penalty": control_penalty,
            "reward": reward,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    # Renderização opcional
    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    # Fecha viewer
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None