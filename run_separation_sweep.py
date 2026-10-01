"""Driver: run the full data-gen -> train -> screen -> evaluate cycle for
several GMM-geometry configs, sequentially and unattended.

Usage:
    python run_separation_sweep.py [--dry_run]

--dry_run uses tiny N/epochs (for validating the driver itself, not for
real results). Without it, runs the real N=3000/n=100/100-epoch configs.

Writes each config's data/checkpoint/results under the repo's data/,
checkpoints/, results/ subdirectories, labeled by config name. Never touches
the existing baseline (data/train.npz, data/test.npz, checkpoints/model.pth,
results/*.png at the top level) -- every config gets its own subdirectory.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/Users/riyamehta/ot-dual-potentials")
PYTHON = sys.executable
LOG_DIR = Path(__file__).parent / "sweep_logs"

BIDIR_K_VALUES = "3,5,10,15,20,25,30,35,40,45,50,60,75,100"

CONFIGS = [
    {"label": "sep_0.15", "separation_scale": 0.15, "decoupled": False},
    {"label": "sep_0.3", "separation_scale": 0.3, "decoupled": False},
    {"label": "sep_0.6", "separation_scale": 0.6, "decoupled": False},
    {"label": "sep_2.0", "separation_scale": 2.0, "decoupled": False},
    {"label": "decoupled_sep_1.0", "separation_scale": 1.0, "decoupled": True},
]


def run_stage(label, stage_name, cmd, log_dir):
    log_path = log_dir / f"{label}_{stage_name}.log"
    print(f"[{time.strftime('%H:%M:%S')}] {label}: starting {stage_name}  ({' '.join(cmd)})", flush=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {label}: {stage_name} {status} in {elapsed:.1f}s  (log: {log_path})", flush=True)
    return ok, elapsed


def run_config(cfg, num_instances, n_points, num_epochs, log_dir, status):
    label = cfg["label"]
    data_dir = REPO / "data" / label
    checkpoint_path = REPO / "checkpoints" / label / "model.pth"
    results_dir = REPO / "results" / label

    status[label] = {"stages": {}}
    cfg_t0 = time.time()

    # 1. data-gen + exact solve
    cmd = [
        PYTHON, "ot_solve.py",
        "--num_instances", str(num_instances),
        "--n_points", str(n_points),
        "--seed", "0",
        "--out_dir", str(data_dir),
        "--separation_scale", str(cfg["separation_scale"]),
    ]
    if cfg["decoupled"]:
        cmd.append("--decoupled_target")
    ok, elapsed = run_stage(label, "01_data_gen", cmd, log_dir)
    status[label]["stages"]["data_gen"] = {"ok": ok, "seconds": elapsed}
    if not ok:
        print(f"  {label}: data-gen failed, skipping remaining stages for this config")
        status[label]["total_seconds"] = time.time() - cfg_t0
        return

    # 2. train
    cmd = [
        PYTHON, "train.py",
        "--train_path", str(data_dir / "train.npz"),
        "--test_path", str(data_dir / "test.npz"),
        "--checkpoint_path", str(checkpoint_path),
        "--loss_plot_path", str(results_dir / "loss_curve.png"),
        "--r2_plot_path", str(results_dir / "r2_curve.png"),
        "--metrics_path", str(results_dir / "metrics.csv"),
        "--num_epochs", str(num_epochs),
        "--seed", "0",
    ]
    ok, elapsed = run_stage(label, "02_train", cmd, log_dir)
    status[label]["stages"]["train"] = {"ok": ok, "seconds": elapsed}
    if not ok:
        print(f"  {label}: train failed, skipping remaining stages for this config")
        status[label]["total_seconds"] = time.time() - cfg_t0
        return

    # 3. screen (smoke test, log only)
    cmd = [
        PYTHON, "screen.py",
        "--test_path", str(data_dir / "test.npz"),
        "--checkpoint_path", str(checkpoint_path),
    ]
    ok, elapsed = run_stage(label, "03_screen", cmd, log_dir)
    status[label]["stages"]["screen"] = {"ok": ok, "seconds": elapsed}
    # non-fatal if this fails; proceed to evaluate regardless

    # 4. evaluate (the results that matter: regression + bidirectional sweep)
    cmd = [
        PYTHON, "evaluate.py",
        "--test_path", str(data_dir / "test.npz"),
        "--checkpoint_path", str(checkpoint_path),
        "--results_dir", str(results_dir),
        "--bidir_k_values", BIDIR_K_VALUES,
    ]
    ok, elapsed = run_stage(label, "04_evaluate", cmd, log_dir)
    status[label]["stages"]["evaluate"] = {"ok": ok, "seconds": elapsed}

    status[label]["total_seconds"] = time.time() - cfg_t0
    print(f"[{time.strftime('%H:%M:%S')}] {label}: DONE in {status[label]['total_seconds']:.1f}s total", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only", type=str, default=None, help="comma-separated labels to run (subset)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        num_instances, n_points, num_epochs = 30, 20, 2
    else:
        num_instances, n_points, num_epochs = 3000, 100, 100

    configs = CONFIGS
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c["label"] in wanted]

    status = {}
    sweep_t0 = time.time()
    print(f"=== separation-scale sweep starting: {len(configs)} configs, "
          f"N={num_instances} n={n_points} epochs={num_epochs} ===", flush=True)
    for cfg in configs:
        run_config(cfg, num_instances, n_points, num_epochs, LOG_DIR, status)
        with open(LOG_DIR / "status.json", "w") as f:
            json.dump(status, f, indent=2)

    total = time.time() - sweep_t0
    print(f"=== sweep finished in {total:.1f}s ({total/60:.1f} min) ===", flush=True)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
