import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
output_file = PROJECT_ROOT / "exts.txt"

exts = Counter(
    os.path.splitext(f)[1].lower()
    for d, _, files in os.walk(PROJECT_ROOT)
    for f in files
    if ".git" not in d and "venv" not in d and "__pycache__" not in d
)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(
        "\n".join(
            f"{ext if ext else 'Sin extensión'}: {count}"
            for ext, count in exts.most_common()
        )
    )

print(f"Written to {output_file}")
