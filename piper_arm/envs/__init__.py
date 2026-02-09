import gymnasium as gym
from gym_aloha.env import AlohaEnv as GymAlohaEnv
from gymnasium.envs.registration import register


class TaskDescriptionWrapper(gym.Wrapper):
    """Wrapper that adds a task_description method to a gym env.

    SmolVLA requires language conditioning via task descriptions. The aloha sim
    envs don't expose task_description, so during eval the tokenizer receives an
    empty string -- a train/eval mismatch that kills performance.
    """

    def __init__(self, env, task_description: str = ""):
        super().__init__(env)
        self._task_description = task_description

    def task_description(self) -> str:
        return self._task_description

    def task(self) -> str:
        return self._task_description


def _make_aloha_transfer_cube(**kwargs):
    env = GymAlohaEnv(task="transfer_cube", **kwargs)
    return TaskDescriptionWrapper(
        env,
        task_description=(
            "Pick up the cube with the right arm and transfer it to the left arm."
        ),
    )


register(
    id="gym_aloha/AlohaTransferCubeWithTask-v0",
    entry_point="piper_arm.envs:_make_aloha_transfer_cube",
    nondeterministic=True,
)
