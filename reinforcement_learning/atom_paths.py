from pathlib import Path
import sys


RL_ROOT = Path(__file__).resolve().parent
ATOM_SRC = RL_ROOT / "op3_model" / "src"
ATOM_SCENE_XML = ATOM_SRC / "model" / "scene.xml"

BEST_OPTUNA_MODEL = (
    RL_ROOT
    / "saida_optuna_atom"
    / "treino_final_melhor"
    / "melhor_modelo"
    / "best_model.zip"
)
FINAL_OPTUNA_MODEL = (
    RL_ROOT
    / "saida_optuna_atom"
    / "treino_final_melhor"
    / "ppo_atom_optuna_final.zip"
)
VISUAL_TRAINING_MODEL = RL_ROOT / "saida_treino_visual_atom" / "ppo_atom_visual.zip"
LEGACY_BEST_MODEL = RL_ROOT / "saida_treino_atom" / "melhor_modelo" / "best_model.zip"
LEGACY_FINAL_MODEL = RL_ROOT / "saida_treino_atom" / "ppo_atom_final.zip"
BEST_MODELS_DIR = RL_ROOT / "saida_treino_atom" / "conjunto_melhores"

DEFAULT_MODEL_CANDIDATES = (
    BEST_OPTUNA_MODEL,
    FINAL_OPTUNA_MODEL,
    VISUAL_TRAINING_MODEL,
    LEGACY_BEST_MODEL,
    LEGACY_FINAL_MODEL,
)


def register_atom_env():
    src_path = str(ATOM_SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import atom  # noqa: F401


def default_model_path():
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return BEST_OPTUNA_MODEL
