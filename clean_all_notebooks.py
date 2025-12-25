import nbformat
from pathlib import Path

# Folder containing notebooks
NOTEBOOK_DIR = Path("https://github.com/ShaikhBorhanUddin/End-to-End-NLP-Pipeline-For-Human-vs-AI-Generated-Text-Classification-System/tree/main/Notebooks")

if not NOTEBOOK_DIR.exists():
    raise FileNotFoundError(f"Folder '{NOTEBOOK_DIR}' not found")

for notebook_path in NOTEBOOK_DIR.glob("*.ipynb"):
    print(f"Cleaning: {notebook_path.name}")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Remove widget metadata from cells
    for cell in nb.cells:
        if "widget" in cell.metadata:
            del cell.metadata["widget"]

    # Save back to the same file (in-place)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

print("✅ All notebooks cleaned successfully.")
