import os
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer


class AtomEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode=None, debug_reward=False,):
        
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

        self.prev_x = 0.0
        self.current_step = 0

        mujoco.mj_forward(self.model, self.data)

        # pegar a posição x do target
        target_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target"
        )

        self.target_pos = self.data.xpos[target_id]
        print("target pos:" + str(self.target_pos))

        self.chest_body_id = self._body_id("chest")

        # self.initial_height = float(self.initial_qpos[2])

        
        self.prev_x = float(self.data.qpos[0])

    def _body_id(self, name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        if body_id < 0:
            raise ValueError(f"Body nao encontrado no modelo MuJoCo: {name}")

        return body_id


    def _get_obs(self):
        obs = np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])

        return obs.astype(np.float32)

    def _torso_height(self):
        return float(self.data.xpos[self.chest_body_id][2])

    def _has_fallen(self):
        return self._torso_height() < 0.30

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel

        if self.model.nu > 0:
            self.data.ctrl[:] = self.default_ctrl

        mujoco.mj_forward(self.model, self.data)

        self.prev_x = float(self.data.qpos[0])

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # Limita as ações
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        # Converte ação da política em comandos dos motores
        target_ctrl = self.default_ctrl + self.action_scale * action

        # Respeita os limites dos atuadores
        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                low, high = self.model.actuator_ctrlrange[i]
                target_ctrl[i] = np.clip(target_ctrl[i], low, high)

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

        fallen = self._has_fallen()

        # ------------------------------
        # Recompensa
        # ------------------------------

        reward = 0.0

        # andar em direção ao target x
        pdif = target_pos - robot_pos # vetor até o alvo
        distance = np.linalg.norm(pdif) # distancia até o alvo (normalização do vetor)

        if distance > 1e-6:
            phat = pdif / distance
            reward = np.dot(robot_vel, phat) * 2 # estava em 1.1
        else:
            reward = 0.0

        # keep alive
        reward += 0.1

        # penalidade por queda
        if fallen:
            reward -= 140.0

        # ------------------------------
        # Finalização
        # ------------------------------
        self.prev_x = current_x

        terminated = fallen

        # bônus por chegar ao alvo
        if distance < 0.30:
            reward += 200
            terminated = True

        truncated = self.current_step >= self.max_episode_steps

        obs = self._get_obs()

        info = {
            "forward_velocity": forward_velocity,
            "x_position": current_x,
            "torso_height": torso_height,
            "reward": reward,
        }

        print(
            f"x={current_x:.4f} "
            f"vel={forward_velocity:.3f}"
            f"torsol={torso_height:.3f}"
            f"reward={reward:.3f}"
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
