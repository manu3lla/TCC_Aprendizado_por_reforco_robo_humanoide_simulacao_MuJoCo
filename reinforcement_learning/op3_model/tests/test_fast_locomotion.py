import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

RL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RL_ROOT))
sys.path.insert(0, str(RL_ROOT / "op3_model" / "src"))
from atom.env import AtomEnv
from avaliar_locomocao import (
    SafeSpeedEvalCallback, evaluate_locomotion, restore_env_args, save_env_config,
)
from ajustar_politica_velocidade import apply_action_adjustment
from model_registry import command_for_model
from ver_modelo_atom import make_visualization_env, pace_simulation_step
from otimizar_hpps_optuna import effective_ppo_params, final_ppo_overrides


class FastEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            env = AtomEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_existing_checkpoint_keeps_same_dynamics_and_observations(self):
        normal = self.make_env(gait_walk_mode=True)
        fast = self.make_env(fast_walk_mode=True)
        normal_obs, _ = normal.reset(seed=42)
        fast_obs, _ = fast.reset(seed=42)
        np.testing.assert_array_equal(normal_obs, fast_obs)
        rng = np.random.default_rng(42)
        for _ in range(20):
            action = rng.uniform(-1, 1, normal.action_space.shape)
            normal_obs = normal.step(action)[0]
            fast_obs = fast.step(action)[0]
            np.testing.assert_array_equal(normal_obs, fast_obs)
        self.assertEqual(normal.desired_walk_speed, 0.145)
        self.assertEqual(fast.desired_walk_speed, 0.35)

    def test_human_render_runs_once_per_control_step(self):
        env = self.make_env(fast_walk_mode=True, render_mode="human")
        env.reset()
        with patch.object(env, "render") as render:
            env.step(np.zeros(env.action_space.shape))
        render.assert_called_once()

    def test_fall_never_receives_positive_speed_reward(self):
        env = self.make_env(fast_walk_mode=True)
        env.reset()
        env.data.qvel[0] = -0.35
        with patch.object(env, "_has_fallen", return_value=True):
            _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        self.assertTrue(terminated)
        self.assertGreater(info["velocity_to_target"], 0)
        for term in ("target_velocity", "distance_progress", "walk_speed", "safe_speed", "gait_forward"):
            self.assertLessEqual(info["reward_terms"][term], 0.0, term)

    def test_invalid_speed_and_conflicting_modes(self):
        for speed in (0, -1, float("nan"), float("inf")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                AtomEnv(fast_walk_mode=True, desired_walk_speed=speed)
        with self.assertRaises(ValueError):
            AtomEnv(fast_walk_mode=True, gait_stand_mode=True)
        for period in (0, 1, 1.5, float("nan"), float("inf")):
            with self.subTest(period=period), self.assertRaises(ValueError):
                AtomEnv(gait_period_steps=period)

    def test_fast_speed_tracking_is_a_cost_and_shaping_stops_on_fall(self):
        env = self.make_env(fast_walk_mode=True)
        env.reset()
        with patch.object(env, "_has_fallen", return_value=True):
            _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        self.assertLess(info["reward_terms"]["walk_speed"], 0.0)
        for term in ("gait_pose", "stance_foot_flat", "upright", "height"):
            self.assertEqual(info["reward_terms"][term], 0.0, term)


class SpeedSelectionTests(unittest.TestCase):
    def test_only_faster_successful_fall_free_models_replace_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            callback = SafeSpeedEvalCallback(None, temp, temp, 1, 1, 42)
            callback.model = SimpleNamespace(save=Mock(), logger=Mock())
            candidates = [
                (0.23, True, False, 10),
                (0.50, False, True, 10000),
                (0.40, False, False, 20000),
                (0.22, True, False, 30000),
                (0.30, True, False, 5),
            ]
            for speed, success, fallen, reward in candidates:
                result = {
                    "success_rate": float(success), "fall_rate": float(fallen),
                    "mean_progress_speed_m_s": speed, "mean_seconds": 20,
                    "mean_reward": reward,
                    "episodes": [{"reward": reward, "length": 2000, "success": success}],
                }
                with patch("avaliar_locomocao.evaluate_locomotion", return_value=result):
                    callback._evaluate()
            self.assertEqual(callback.model.save.call_count, 2)
            self.assertEqual(callback.best_speed, 0.30)
            self.assertEqual([ev["selected"] for ev in callback.evaluations],
                             [True, False, False, False, True])

    def test_metric_counts_net_progress_not_lateral_motion_or_reward(self):
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(max_episode_steps=10),
            reset=lambda **kwargs: (np.zeros(1), {"distance_to_target": 5.0}),
            step=lambda action: (np.zeros(1), 100000.0, True, False, {
                "distance_to_target": 4.0, "elapsed_simulation_seconds": 10.0,
                "path_drift": 3.0, "torso_upright": 1.0,
                "is_success": False, "fallen": False,
            }),
        )
        model = SimpleNamespace(predict=lambda obs, **kwargs: (np.zeros(1), None))
        result = evaluate_locomotion(model, env, n_episodes=1)
        self.assertEqual(result["mean_progress_speed_m_s"], 0.1)
        self.assertEqual(result["success_rate"], 0.0)
        self.assertEqual(result["max_path_drift_m"], 3.0)


class ConfigurationTests(unittest.TestCase):
    def test_real_time_playback_only_waits_for_remaining_step_time(self):
        with patch("ver_modelo_atom.time.perf_counter", return_value=10.006), \
                patch("ver_modelo_atom.time.sleep") as sleep:
            pace_simulation_step(10.0, 0.01, None)
        self.assertAlmostEqual(sleep.call_args.args[0], 0.004)
        with patch("ver_modelo_atom.time.perf_counter", return_value=10.02), \
                patch("ver_modelo_atom.time.sleep") as sleep:
            pace_simulation_step(10.0, 0.01, None)
        sleep.assert_not_called()

    def test_zero_sleep_runs_without_pacing(self):
        with patch("ver_modelo_atom.time.sleep") as sleep:
            pace_simulation_step(10.0, 0.01, 0)
        sleep.assert_not_called()

    def test_saved_model_restores_cadence_and_respects_explicit_overrides(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = AtomEnv(fast_walk_mode=True, gait_period_steps=68)
        self.addCleanup(env.close)
        with tempfile.TemporaryDirectory() as temp:
            save_env_config(env, temp)
            args = SimpleNamespace(
                velocidade_alvo=None, periodo_gait=None,
                modo_gait_andando=False, modo_gait_rapido=False,
            )
            restore_env_args(args, Path(temp) / "best_model.zip")
            self.assertTrue(args.modo_gait_rapido)
            self.assertTrue(args.modo_gait_andando)
            self.assertEqual(args.periodo_gait, 68)
            self.assertEqual(args.velocidade_alvo, 0.35)
            args.periodo_gait = 60
            args.velocidade_alvo = 0.4
            restore_env_args(args, Path(temp) / "best_model.zip")
            self.assertEqual(args.periodo_gait, 60)
            self.assertEqual(args.velocidade_alvo, 0.4)

    def test_policy_adjustment_is_symmetric_and_does_not_accumulate(self):
        import torch
        actor = torch.nn.Linear(2, 6)
        model = SimpleNamespace(policy=SimpleNamespace(action_net=actor))
        weight = actor.weight.detach().clone()
        bias = actor.bias.detach().clone()
        indices = {
            "left_leg_pitch": 0, "right_leg_pitch": 1,
            "left_knee": 2, "right_knee": 3,
            "left_foot_pitch": 4, "right_foot_pitch": 5,
        }
        apply_action_adjustment(model, weight, bias, indices, 0.7, 0.2, 0.1, -0.1)
        expected_bias = bias * 0.7 + torch.tensor([0.2, -0.2, 0.1, -0.1, -0.1, 0.1])
        torch.testing.assert_close(actor.bias, expected_bias)
        apply_action_adjustment(model, weight, bias, indices, 0.7, 0.2, 0.1, -0.1)
        torch.testing.assert_close(actor.bias, expected_bias)
        torch.testing.assert_close(actor.weight, weight * 0.7)

    def test_resume_applies_hyperparameters_without_changing_policy_weights(self):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            env = AtomEnv(fast_walk_mode=True)
        self.addCleanup(env.close)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, seed=42)
        args = SimpleNamespace(
            learning_rate=3e-5, gamma=0.995, gae_lambda=0.98,
            n_epochs=5, clip_range=0.08, target_kl=0.01,
        )
        overrides = final_ppo_overrides(args)
        with tempfile.TemporaryDirectory() as temp:
            model.save(Path(temp) / "parent")
            resumed = PPO.load(Path(temp) / "parent.zip", env=env, **overrides)
        for key, weight in model.policy.state_dict().items():
            self.assertTrue(torch.equal(weight, resumed.policy.state_dict()[key]), key)
        params = effective_ppo_params(resumed)
        for key, value in overrides.items():
            self.assertEqual(params[key], value, key)
        self.assertEqual(resumed.rollout_buffer.gamma, 0.995)
        self.assertEqual(resumed.rollout_buffer.gae_lambda, 0.98)

    def test_invalid_training_overrides_are_rejected(self):
        for name, value in (
            ("learning_rate", float("nan")), ("gamma", 1.1),
            ("gae_lambda", 0), ("n_epochs", 0),
            ("clip_range", 1.0), ("target_kl", float("inf")),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                final_ppo_overrides(SimpleNamespace(**{name: value}))

    def test_registered_view_command_repeats_speed_configuration(self):
        command = command_for_model(
            RL_ROOT / "test_model.zip", use_env_snapshot=True,
            fast_walk_mode=True, desired_walk_speed=0.4,
            gait_period_steps=68,
        )
        self.assertIn("--usar-env-snapshot", command)
        self.assertIn("--modo-gait-rapido", command)
        self.assertIn("--velocidade-alvo 0.4", command)
        self.assertIn("--periodo-gait 68", command)

    def test_old_snapshot_receives_only_legacy_arguments(self):
        args = SimpleNamespace(
            debug_reward=False, usar_gait_inicial=True,
            modo_gait_parado=False, modo_gait_andando=True,
            modo_gait_rapido=False, velocidade_alvo=None,
        )
        with patch("ver_modelo_atom.load_snapshot_env_class") as loader:
            make_visualization_env(args, Path("legacy_snapshot"))
        self.assertEqual(loader.return_value.call_args.kwargs, {
            "render_mode": "human", "debug_reward": False,
            "use_reference_gait": True, "gait_stand_mode": False,
            "gait_walk_mode": True,
        })


if __name__ == "__main__":
    unittest.main()
