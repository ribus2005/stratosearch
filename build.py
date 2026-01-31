import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SOURCE_WEIGHTS = PROJECT_ROOT / "weights"
DEFAULT_DIST = PROJECT_ROOT / "dist"
SOURCE_VIEW = PROJECT_ROOT / "stratosearch" / "gui" / "UI" / "View"
DEFAULT_TYPE = "onefile"


def build_exe(output_dir: Path, type: str):
    cmd = [
        "poetry", "run", "pyinstaller",
        "--noconfirm",
        "--windowed",
        f"--{type}",
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


def copy_styles(target_dir: Path):
    view_dst = target_dir / "UI" / "View"
    if view_dst.exists():
        shutil.rmtree(view_dst)
    shutil.copytree(SOURCE_VIEW, view_dst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сборка релизной версии Stratosearch с внешней папкой weights"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_DIST),
        help="Папка, куда будет собран exe и некоторые необходимые файлы (по умолчанию ./dist)",
    )
    parser.add_argument(
        "--type",
        nargs="?",
        default=str(DEFAULT_TYPE),
        help="Тип сборки - либо onefile, либо onedir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.type not in ["onefile", "onedir"]:
        raise ValueError("Возможны только два вида сборки: onefile, onedir")

    target_path = Path(args.output).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    print("Сборка exe...")
    build_exe(target_path, args.type)

    print("Копирование файлов...")
    copy_files_path = target_path if args.type == "onefile" else target_path / "stratosearch"
    copy_weights(copy_files_path)
    copy_styles(copy_files_path)

    print("Готово!")


if __name__ == "__main__":
    main()
