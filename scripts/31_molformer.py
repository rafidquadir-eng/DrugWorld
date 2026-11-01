"""MoLFormer-XL embeddings of the parent SMILES (plan section 3; A1 O3).

Before loading, every file of the pinned snapshot that is used (remote code, weights,
tokenizer, config) is hashed and compared with manifests/models.yaml; any mismatch stops the
run. Loaded offline (HF_HUB_OFFLINE), from safetensors, with trust_remote_code at revision
7b12d946 only. Embedding = the model's pooler output (masked mean over tokens, as on the model
card), deterministic_eval=True, CPU, float32, batches of 32.

Outputs data/structures/molformer_<set>.npz (compounds, emb float32 [n, 768]), backed up.
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from drugworld import REPO, backup  # noqa: E402

M = yaml.safe_load((REPO / "manifests" / "models.yaml").read_text())["molformer_xl"]
SNAP = (Path.home() / ".cache" / "huggingface" / "hub" / "models--ibm-research--MoLFormer-XL-both-10pct"
        / "snapshots" / M["revision"])


def verify() -> None:
    want = {**M["file_sha256"], **M["remote_code_sha256"]}
    for name, h in want.items():
        got = hashlib.sha256((SNAP / name).resolve().read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"{name}: SHA-256 {got} differs from manifests/models.yaml")
    print(f"verified {len(want)} files of {M['repo']}@{M['revision'][:8]}")


def main() -> int:
    verify()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModel.from_pretrained(SNAP, trust_remote_code=True, deterministic_eval=True,
                                      use_safetensors=True).eval()
    torch.manual_seed(0)
    for which in ("idr0119", "idr0064"):
        df = pd.read_csv(REPO / "data" / "structures" / f"compounds_{which}.csv")
        embs = []
        with torch.no_grad():
            for i in range(0, len(df), 32):
                enc = tok(list(df.smiles_parent[i:i + 32]), padding=True, return_tensors="pt")
                embs.append(model(**enc).pooler_output.float().numpy())
        emb = np.concatenate(embs)
        if not np.isfinite(emb).all():
            raise SystemExit(f"{which}: non-finite embedding")
        path = REPO / "data" / "structures" / f"molformer_{which}.npz"
        np.savez(path, compounds=df.compound.to_numpy(), emb=emb)
        backup.backup(path)
        print(f"{which}: {emb.shape}, mean norm {np.linalg.norm(emb, axis=1).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
