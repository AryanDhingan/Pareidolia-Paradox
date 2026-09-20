import os
import sys
import runpy


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


if __name__ == "__main__":
    runpy.run_path(
        os.path.join(
            SRC_DIR,
            "final_train.py"
        ),
        run_name="__main__"
    )