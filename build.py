import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SOURCE_WEIGHTS = PROJECT_ROOT / "weights"
DEFAULT_DIST = PROJECT_ROOT / "dist"


def build_exe(output_dir: Path):
    cmd = [
        "poetry", "run", "pyinstaller",
        "--noconfirm",
        "--windowed",
        "--onefile",
        "--collect-all", "PySide6",
        "--name", "stratosearch",
        "--distpath", str(output_dir),
        "--workpath", str(PROJECT_ROOT / "build"),
        "--specpath", str(PROJECT_ROOT),
        "stratosearch/gui/main.py",
    ]
    subprocess.check_call(cmd)


def copy_weights(target_dir: Path):
    weights_dst = target_dir / "weights"
    if weights_dst.exists():
        shutil.rmtree(weights_dst)
    shutil.copytree(SOURCE_WEIGHTS, weights_dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сборка релизной версии Stratosearch с внешней папкой weights"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_DIST),
        help="Папка, куда будет собран exe и скопированы веса (по умолчанию ./dist)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_path = Path(args.output).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("Сборка exe...")
    build_exe(target_path)

    print("Копирование весов...")
    copy_weights(target_path)

    print("Готово!")


if __name__ == "__main__":
    main()
