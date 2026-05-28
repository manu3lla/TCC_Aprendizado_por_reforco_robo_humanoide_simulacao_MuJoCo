import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
EXPERIMENTS_DIR = ROOT / "saida_experimentos_tcc1"
SINGLE_RUN_DIR = ROOT / "saida_treino_atom"
OUT_DIR = SINGLE_RUN_DIR / "conjunto_melhores"

def find_best_models():
    found = []

    # search inside saida_experimentos_tcc1/exp_*/melhor_model/best_model.zip
    if EXPERIMENTS_DIR.exists():
        for exp in EXPERIMENTS_DIR.iterdir():
            if exp.is_dir():
                candidate = exp / "melhor_modelo" / "best_model.zip"
                if candidate.exists():
                    found.append((candidate, exp.name))

    # also check single run folder
    single_candidate = SINGLE_RUN_DIR / "melhor_modelo" / "best_model.zip"
    if single_candidate.exists():
        found.append((single_candidate, "saida_treino_atom"))

    # also include any best_model.zip found anywhere under reinforcement_learning
    for p in ROOT.rglob('best_model.zip'):
        parent = p.parent
        # avoid duplicates
        if not any(str(p.resolve()) == str(f[0].resolve()) for f in found):
            found.append((p, parent.name))

    return found

def collect(found):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for idx, (src, tag) in enumerate(found, start=1):
        dst_name = f"{idx:02d}_{tag}_best_model.zip"
        dst = OUT_DIR / dst_name
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")

def main():
    found = find_best_models()
    if not found:
        print("No best_model.zip files found.")
        return
    print(f"Found {len(found)} best models")
    collect(found)
    print(f"Collected models into: {OUT_DIR}")

if __name__ == '__main__':
    main()
