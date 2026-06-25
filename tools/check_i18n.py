import os

files = []
for root, dirs, fs in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git"]]
    for f in fs:
        if f.endswith(".py"):
            files.append(os.path.join(root, f))

issues = []
for fp in files:
    with open(fp, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if any("\u4e00" <= c <= "\u9fff" for c in line):
                stripped = line.strip()
                if not stripped.startswith("#") and (
                    "print(" in line or "logger." in line or "logging." in line
                ):
                    issues.append(f"{fp}:{i}: {line.rstrip()}")

if issues:
    print("Remaining Chinese in print/log output:")
    for x in issues:
        print(x)
else:
    print("All print/log messages are now English.")
