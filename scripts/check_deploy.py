"""Pre-flight check before deploying the app to Streamlit Community Cloud.

Run from the repository root:   python scripts/check_deploy.py
It reports PASS / WARN / FAIL for each check and exits with code 1 if anything failed.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
results = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(level, message):
    results[level] += 1
    print(f"[{level}] {message}")


def tracked_files():
    # files Git would upload; falls back to walking the folder if this is not a Git repository
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, cwd=ROOT, check=True).stdout
        return [ROOT / line for line in out.splitlines() if line.strip()], True
    except Exception:
        skip = {".venv", ".git", "__pycache__", ".ipynb_checkpoints"}
        files = [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & skip)]
        return files, False


# ------------------------------------------------------------------ 1. required files
required = ["app/app.py", "app/requirements.txt", "app/config.json", "app/kb.csv",
            "app/kb_embeddings.npy", ".streamlit/config.toml", ".gitignore"]
for rel in required:
    if (ROOT / rel).exists():
        report("PASS", f"found {rel}")
    else:
        report("FAIL", f"missing {rel} (run this script from the repository root)")

# ------------------------------------------------------------------ 2. config.json
config = {}
try:
    config = json.loads((ROOT / "app/config.json").read_text())
    missing = [k for k in ["threshold", "prefix", "max_len", "model_hub_repo"] if k not in config]
    if missing:
        report("FAIL", f"config.json is missing keys: {missing}")
    else:
        report("PASS", f"config.json ok (threshold {config['threshold']}, max_len {config['max_len']})")
    repo = str(config.get("model_hub_repo", ""))
    if not repo or "YOUR_HF_USERNAME" in repo or "/" not in repo:
        report("FAIL", f"model_hub_repo is not set correctly: {repo!r} (expected alain-ngabo/healthcare-qa-retriever)")
    else:
        report("PASS", f"model_hub_repo = {repo}")
except Exception as error:
    report("FAIL", f"could not read app/config.json: {error}")

# ------------------------------------------------------------------ 3. knowledge base and embeddings agree
try:
    import numpy as np
    import pandas as pd
    kb = pd.read_csv(ROOT / "app/kb.csv")
    emb = np.load(ROOT / "app/kb_embeddings.npy", mmap_mode="r")
    if {"question", "answer"} - set(kb.columns):
        report("FAIL", f"kb.csv needs the columns question and answer, found {list(kb.columns)}")
    elif len(kb) != emb.shape[0]:
        report("FAIL", f"kb.csv has {len(kb)} rows but kb_embeddings.npy has {emb.shape[0]}. Export them together")
    else:
        report("PASS", f"knowledge base ok: {len(kb)} answers, vectors {emb.shape}, dtype {emb.dtype}")
except ImportError:
    report("WARN", "numpy and pandas are not installed here, so kb.csv and the embeddings were not compared")
except Exception as error:
    report("FAIL", f"could not compare kb.csv and kb_embeddings.npy: {error}")

# ------------------------------------------------------------------ 4. requirements and app file
req = (ROOT / "app/requirements.txt").read_text().lower() if (ROOT / "app/requirements.txt").exists() else ""
for package in ["streamlit", "sentence-transformers"]:
    report("PASS" if package in req else "FAIL", f"app/requirements.txt {'lists' if package in req else 'does not list'} {package}")
if "gradio" in req:
    report("WARN", "app/requirements.txt lists gradio, which the Streamlit app does not need")
if "streamlit==" not in req and "streamlit" in req:
    report("WARN", "streamlit is not pinned; pin it (for example streamlit==1.65.0) so the deployed version matches yours")
app_text = (ROOT / "app/app.py").read_text(errors="ignore") if (ROOT / "app/app.py").exists() else ""
if "import streamlit" in app_text:
    report("PASS", "app/app.py is a Streamlit app")
else:
    report("FAIL", "app/app.py does not import streamlit (is it still the Gradio version?)")
if (ROOT / "requirements.txt").exists() and (ROOT / "app/requirements.txt").exists():
    report("PASS", "app/requirements.txt sits next to the app file, so Streamlit Cloud uses it instead of the root file")

# ------------------------------------------------------------------ 5. file sizes
all_files, is_git = tracked_files()
NOISE = {".venv", "venv", "site-packages", "__pycache__", "node_modules", ".ipynb_checkpoints"}
def rel(p):
    return p.relative_to(ROOT).as_posix()
noise = [p for p in all_files if set(rel(p).split("/")) & NOISE]
if noise:
    report("FAIL", f".venv or similar folders would be uploaded ({len(noise)} files). Add .venv/ to .gitignore")
models = [p for p in all_files if rel(p).startswith("models/")]
if models:
    total = sum(p.stat().st_size for p in models if p.exists()) / 1e6
    report("FAIL", f"models/ would be uploaded ({total:.0f} MB of model weights). Add models/ to .gitignore; the model lives on Hugging Face")
files = [p for p in all_files if p not in noise and p not in models]
big = []
for p in files:
    try:
        size_mb = p.stat().st_size / 1e6
    except OSError:
        continue
    if size_mb >= 100:
        report("FAIL", f"{rel(p)} is {size_mb:.0f} MB; GitHub rejects files over 100 MB")
    elif size_mb >= 50:
        report("WARN", f"{rel(p)} is {size_mb:.0f} MB; GitHub warns above 50 MB")
    elif size_mb >= 10:
        big.append(f"{rel(p)} ({size_mb:.0f} MB)")
report("PASS", "no file is over 50 MB" + (f"; largest: {', '.join(big[:4])}" if big else ""))

# ------------------------------------------------------------------ 6. secrets must not be uploaded
rels = {rel(p) for p in files}
leaked = [r for r in rels if r.endswith("secrets.toml") or r == ".env"]
if leaked:
    report("FAIL", f"secret files would be uploaded: {sorted(leaked)}. Delete app/secrets.toml and ignore .streamlit/secrets.toml")
else:
    report("PASS", "no secrets file would be uploaded")
gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
report("PASS" if "secrets.toml" in gi else "FAIL", ".gitignore " + ("ignores" if "secrets.toml" in gi else "does not ignore") + " secrets.toml")
if is_git:
    report("PASS", "this folder is a Git repository (file list taken from Git)")
else:
    report("WARN", "not a Git repository yet (file list taken from the folder); run git init first")

patterns = {"Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"), "Google OAuth secret": re.compile(r"GOCSPX-[A-Za-z0-9_\-]{10,}"),
            "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")}
found = []
for p in files:
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".npy", ".zip", ".pdf", ".docx", ".safetensors", ".bin"}:
        continue
    try:
        if p.stat().st_size > 20_000_000:
            continue
        text = p.read_text(errors="ignore")
    except OSError:
        continue
    for name, pattern in patterns.items():
        if pattern.search(text):
            found.append(f"{rel(p)} ({name})")
if found:
    for item in found[:8]:
        report("FAIL", f"possible secret in {item}: remove it, revoke the key, and clear the notebook output")
else:
    report("PASS", "no tokens or keys found in the files that would be uploaded")

# ------------------------------------------------------------------ 7. the model repository must be public
repo = str(config.get("model_hub_repo", ""))
if repo and "YOUR_HF_USERNAME" not in repo:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=10) as response:
            info = json.loads(response.read().decode())
        names = [s.get("rfilename", "") for s in info.get("siblings", [])]
        if info.get("private"):
            report("FAIL", f"{repo} is private; Streamlit Cloud cannot download it. Make it public on Hugging Face")
        elif "modules.json" not in names or not any(n.endswith("safetensors") for n in names):
            report("FAIL", f"{repo} is reachable but lacks modules.json or model.safetensors; push the model again")
        else:
            report("PASS", f"model repo {repo} is public and has the model files")
    except urllib.error.HTTPError as error:
        report("FAIL", f"model repo {repo} returned HTTP {error.code}: check the name and that it is public")
    except Exception as error:
        report("WARN", f"could not reach Hugging Face to check {repo} ({type(error).__name__}); check it in a browser")

# ------------------------------------------------------------------ summary
print(f"\n{results['PASS']} passed, {results['WARN']} warnings, {results['FAIL']} failed")
if results["FAIL"]:
    print("Fix the FAIL lines before you push and deploy.")
    sys.exit(1)
print("Ready to push. On Streamlit Cloud use: main file path = app/app.py")