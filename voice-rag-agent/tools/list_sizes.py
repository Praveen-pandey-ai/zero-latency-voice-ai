import os
from pathlib import Path

root = Path(__file__).parent.parent
files = []
for p in root.rglob('*'):
    if p.is_file():
        try:
            files.append((p, p.stat().st_size))
        except Exception:
            continue

files.sort(key=lambda x: x[1], reverse=True)
total = sum(s for _, s in files)
for p, s in files[:200]:
    mb = s / 1024 / 1024
    if mb >= 0.01:
        print(f"{p}\t{mb:.2f} MB")

print('\nTOTAL:', round(total/1024/1024,2), 'MB')

print('\nFILES >1MB:')
for p, s in files:
    mb = s / 1024 / 1024
    if mb > 1:
        print(f"{p}\t{mb:.2f} MB")
