from metaflow import FlowSpec, Parameter, step
import subprocess
import os

PY = os.environ.get("PY", "python")


def run(cmd: str) -> None:
    print("[RUN]", cmd)
    subprocess.run(cmd, shell=True, check=True)


class TwoStageRecsysFlow(FlowSpec):
    topn = Parameter("topn", default=500)
    mode = Parameter("mode", default="full", help="full or ci")

    @step
    def start(self):
        self.next(self.baselines)

    @step
    def baselines(self):
        run(f"PYTHONPATH=src {PY} scripts/run_baselines.py")
        self.next(self.als)

    @step
    def als(self):
        run(
            f"PYTHONPATH=src {PY} scripts/run_als_split.py "
            f"--name test --history_path data/processed/train_val.parquet "
            f"--eligible_users_path data/processed/eligible_users_test.parquet "
            f"--holdout_path data/processed/holdout_targets_test.parquet "
            f"--als_dir artifacts/als_test --candidates_out data/processed/candidates_test.parquet "
            f"--topn {self.topn}"
        )
        self.next(self.ranker)

    @step
    def ranker(self):
        run(f"PYTHONPATH=src {PY} scripts/build_ranker_datasets.py")
        run(f"PYTHONPATH=src {PY} scripts/train_ranker_lgbm.py")
        self.next(self.eval_and_gate)

    @step
    def eval_and_gate(self):
        run(f"PYTHONPATH=src {PY} scripts/bootstrap_ci_report.py")
        if self.mode == "ci":
            run(f"{PY} scripts/gate.py --mode ci")
        else:
            run(f"{PY} scripts/gate.py")
        self.next(self.package)

    @step
    def package(self):
        run(f"{PY} scripts/package_bundle.py")
        self.next(self.end)

    @step
    def end(self):
        print("[OK] Metaflow run complete")


if __name__ == "__main__":
    TwoStageRecsysFlow()
