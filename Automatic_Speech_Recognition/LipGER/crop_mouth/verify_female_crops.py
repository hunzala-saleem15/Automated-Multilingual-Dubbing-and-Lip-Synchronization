import os
import json

JSON_ROOT = r"E:\ASR\facestar_whisper\formatted_json"

ok = 0
missing = []
bad = []

for file in os.listdir(JSON_ROOT):
    if not file.endswith(".json"):
        continue

    path = os.path.join(JSON_ROOT, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = [data]
    else:
        entries = data

    for entry in entries:
        uid = entry.get("Uid", "UNKNOWN")
        mouth = entry.get("Mouthroi", "")

        if not mouth:
            bad.append((uid, "NO Mouthroi field"))
            continue

        mouth_path = mouth.replace("/", "\\")

        if not os.path.exists(mouth_path):
            missing.append((uid, mouth_path))
        else:
            ok += 1

# ============================
# REPORT
# ============================
print("\n==============================")
print("Valid Mouthroi entries:", ok)
print("Missing Mouthroi files:", len(missing))
print("Bad JSON entries     :", len(bad))

if bad:
    print("\n❌ BAD JSONs:")
    for u, r in bad[:20]:
        print("  ", u, "->", r)

if missing:
    print("\n❌ MISSING FILES:")
    for u, p in missing[:20]:
        print("  ", u, "->", p)

if not missing and not bad:
    print("\n🎉 ALL JSON FILES ARE 100% VALID AND LINKED!")
print("==============================")
