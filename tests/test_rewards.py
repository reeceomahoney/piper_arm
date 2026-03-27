import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from distal.rewards import (
    _build_maha_context,
    build_reward_context,
    compute_nstep_advantages,
    load_reward_context,
    save_reward_context,
)


class RewardsTest(unittest.TestCase):
    def test_steps_context_applies_failure_penalty(self):
        steps_remaining = np.array([2, 1, 0, 1, 0], dtype=np.int32)
        success = np.array([False, False, False, True, True], dtype=bool)
        episode_index = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        cfg = SimpleNamespace(
            reward_type="steps",
            gamma=1.0,
            failure_penalty_scale=0.25,
        )

        ctx = build_reward_context(
            cfg,
            episode_index=episode_index,
            success=success,
            steps_remaining=steps_remaining,
            max_episode_length=3,
        )

        np.testing.assert_allclose(ctx.normalization_constant, 3.0)
        np.testing.assert_allclose(ctx.failure_penalty, 0.75)
        np.testing.assert_allclose(ctx.rewards, [-1 / 3, -1 / 3, -0.25, -1 / 3, 0.0])
        np.testing.assert_allclose(
            ctx.returns,
            [-0.91666667, -0.58333333, -0.25, -0.33333333, 0.0],
        )

    def test_maha_context_uses_checkpoint_norm_and_failure_penalty(self):
        maha = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        success = np.array([False, False, True, True], dtype=bool)
        episode_index = np.array([0, 0, 1, 1], dtype=np.int32)

        ctx = _build_maha_context(
            maha,
            success,
            episode_index,
            gamma=1.0,
            failure_penalty_scale=0.5,
            normalization_constant=10.0,
        )

        np.testing.assert_allclose(ctx.failure_penalty, 5.0)
        np.testing.assert_allclose(ctx.rewards, [-0.1, -0.7, -0.3, -0.4])
        np.testing.assert_allclose(ctx.returns, [-0.8, -0.7, -0.7, -0.4])

    def test_nstep_advantages_share_mc_fallback(self):
        values = np.array([0.2, 0.1, 0.0], dtype=np.float64)
        rewards = np.array([-0.2, -0.3, -0.4], dtype=np.float64)
        returns = np.array([-0.9, -0.7, -0.4], dtype=np.float64)
        episode_index = np.array([0, 0, 0], dtype=np.int32)

        advantages = compute_nstep_advantages(
            values,
            rewards,
            returns,
            episode_index,
            n_step=2,
            gamma=1.0,
        )

        np.testing.assert_allclose(advantages, [-0.7, -0.8, -0.4])

    def test_reward_context_round_trip(self):
        maha = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        success = np.array([False, False, True, True], dtype=bool)
        episode_index = np.array([0, 0, 1, 1], dtype=np.int32)

        ctx = _build_maha_context(
            maha,
            success,
            episode_index,
            gamma=1.0,
            failure_penalty_scale=0.5,
            normalization_constant=10.0,
        )

        with TemporaryDirectory() as tmpdir:
            save_reward_context(Path(tmpdir), ctx, num_frames=4)
            loaded_ctx = load_reward_context(Path(tmpdir))

        np.testing.assert_allclose(loaded_ctx.rewards, ctx.rewards)
        np.testing.assert_allclose(loaded_ctx.returns, ctx.returns)
        self.assertAlmostEqual(loaded_ctx.normalization_constant, 10.0)
        self.assertAlmostEqual(loaded_ctx.failure_penalty, 5.0)


if __name__ == "__main__":
    unittest.main()
