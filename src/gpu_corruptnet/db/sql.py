"""Relational (PostgreSQL) experiment store via SQLAlchemy 2.0.

Portable across engines: pass a Postgres URL in production
(``postgresql+psycopg2://user:pw@host/db``) or the default SQLite file for local dev
and tests. Schema: one Run has many Metrics (per split, per metric name).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    arch: Mapped[str] = mapped_column(String(64))
    device: Mapped[str] = mapped_column(String(32))
    clean_split: Mapped[str] = mapped_column(String(255), default="")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    metrics: Mapped[list[Metric]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    split: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="metrics")


def get_engine(url: str = "sqlite:///runs/metadata.db") -> Engine:
    return create_engine(url, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def log_run(session: Session, results: dict) -> int:
    """Persist a training results dict (from train.run) as a Run + its Metrics."""
    cfg = results.get("config", {})
    run = Run(
        arch=cfg.get("arch", "?"),
        device=results.get("device", "?"),
        clean_split=results.get("clean_split", ""),
        config=cfg,
    )
    for split in ("val", "seen_test", "unseen_test"):
        m = results.get(split) or {}
        for key, val in m.items():
            if isinstance(val, (int, float)):
                run.metrics.append(Metric(split=split, name=key, value=float(val)))
            elif key == "per_class_f1" and isinstance(val, dict):
                for cls, v in val.items():
                    run.metrics.append(Metric(split=split, name=f"f1_{cls}", value=float(v)))
    session.add(run)
    session.commit()
    return run.id


def best_runs(
    session: Session, split: str = "unseen_test", metric: str = "macro_f1"
) -> list[tuple]:
    """(arch, value) rows for a metric, best first — the leaderboard query."""
    rows = session.execute(
        select(Run.arch, Metric.value)
        .join(Metric, Metric.run_id == Run.id)
        .where(Metric.split == split, Metric.name == metric)
        .order_by(Metric.value.desc())
    ).all()
    return [(r[0], r[1]) for r in rows]
