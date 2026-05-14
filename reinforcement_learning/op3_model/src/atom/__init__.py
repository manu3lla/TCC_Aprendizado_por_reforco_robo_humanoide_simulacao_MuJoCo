from gymnasium.envs.registration import register

register(
    id="Atom-v1",
    entry_point="atom.env:AtomEnv",
)