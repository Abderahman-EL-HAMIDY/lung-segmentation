import json
import glob
import os

notebooks = glob.glob("**/*.ipynb", recursive=True)
print(f"Found {len(notebooks)} notebooks")

for path in notebooks:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Remove broken widget metadata
    changed = False
    if "widgets" in nb.get("metadata", {}):
        del nb["metadata"]["widgets"]
        changed = True

    # Also fix per-cell widget state
    for cell in nb.get("cells", []):
        if "widgets" in cell.get("metadata", {}):
            del cell["metadata"]["widgets"]
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"  Fixed: {path}")
    else:
        print(f"  OK:    {path}")

print("Done.")
