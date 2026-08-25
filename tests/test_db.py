from sqlalchemy.orm import Session

from gpu_corruptnet.db.nosql import AnnotationStore
from gpu_corruptnet.db.sql import best_runs, get_engine, init_db, log_run

FAKE_RESULTS = {
    "config": {"arch": "resnet50", "epochs": 5},
    "device": "cpu",
    "clean_split": "train=100 val=20",
    "val": {"macro_f1": 0.5},
    "seen_test": {"macro_f1": 0.9, "binary_recall": 0.95, "per_class_f1": {"shader": 0.8}},
    "unseen_test": {"macro_f1": 0.7},
}


def test_sql_log_run_and_leaderboard(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path/'meta.db'}")
    init_db(engine)
    with Session(engine) as s:
        rid = log_run(s, FAKE_RESULTS)
        assert rid == 1
        other = {
            **FAKE_RESULTS,
            "config": {"arch": "efficientnet_b4"},
            "unseen_test": {"macro_f1": 0.6},
        }
        log_run(s, other)

        board = best_runs(s, split="unseen_test", metric="macro_f1")
        assert board == [("resnet50", 0.7), ("efficientnet_b4", 0.6)]  # best first

        # per-class f1 was flattened into its own metric rows
        seen = best_runs(s, split="seen_test", metric="f1_shader")
        assert ("resnet50", 0.8) in seen


def test_nosql_annotations_and_aggregation():
    store = AnnotationStore.in_memory()
    n = store.add_many(
        [
            {"image_key": "a.png", "artifacts": ["shader", "discoloration"], "severity": 3},
            {"image_key": "b.png", "artifacts": ["shader"], "severity": 2},
            {"image_key": "c.png", "artifacts": [], "severity": 0},
        ]
    )
    assert n == 3
    assert store.count() == 3
    assert len(store.find_by_artifact("shader")) == 2

    counts = store.artifact_counts()
    assert counts["shader"] == 2
    assert counts["discoloration"] == 1

    mid = store.register_model({"arch": "resnet50", "version": "v1", "s3_uri": "s3://.../m.pt"})
    assert isinstance(mid, str)
