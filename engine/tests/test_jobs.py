"""Jobs: what POST /api/jobs accepts, and the progress a poll of
GET /api/jobs/{id} reads."""
import asyncio
import io
import json
import time

from PIL import Image

from jobs import JobManager


def test_a_runner_sets_the_progress_a_poll_reads():
    async def main():
        jm = JobManager()
        seen = []

        async def runner(job, emit, cancel):
            await emit("progress", {"index": 1, "total": 2, "image": "a.jpg"})
            seen.append(dict(job.progress))
            return {"ok": True}

        jm.register_runner("k", runner)
        job = jm.schedule("k", "p", {})
        orphan = jm.schedule("no-such-kind", "p", {})
        jm.start_worker(1)
        for _ in range(500):
            if job.status == "done" and orphan.status == "failed":
                break
            await asyncio.sleep(0.01)
        for t in jm.worker_tasks:
            t.cancel()
        return seen, job, orphan

    seen, job, orphan = asyncio.run(main())
    assert seen == [{"index": 1, "total": 2, "image": "a.jpg", "phase": "running"}]
    assert job.to_public()["result"] == {"ok": True}
    assert "no runner" in orphan.error


def test_an_augmentation_run_counts_the_copies_it_made(client):
    pid = client.post("/api/v2/projects", data={"name": "Aug progress", "labels": json.dumps(["a"])}).json()["project_id"]
    for i in range(2):
        buf = io.BytesIO()
        Image.new("RGB", (48, 40), (40 * i, 90, 160)).save(buf, "JPEG")
        client.post(f"/api/v2/projects/{pid}/imports/raw", files={"image": (f"{i}.jpg", buf.getvalue(), "image/jpeg")})
    cfg = {"camera": {"enabled": True, "frequency": "all", "noise": 2}}
    jid = client.post(f"/api/v2/projects/{pid}/augment/generate",
                      json={"perImageMode": "1", "config": cfg}).json()["job_id"]
    deadline = time.time() + 120
    while (j := client.get(f"/api/jobs/{jid}").json())["status"] not in ("done", "failed") and time.time() < deadline:
        time.sleep(0.1)
    assert j["status"] == "done"
    assert j["progress"]["index"] == j["progress"]["total"] == 2


def test_a_job_on_a_dataset_whose_dataset_json_is_unreadable_is_not_found(client):
    import store

    pid = client.post("/api/v2/projects", data={"name": "No manifest", "labels": json.dumps(["a"])}).json()["project_id"]
    path = store.dataset_dir(pid) / "dataset.json"
    saved = path.read_bytes()
    path.unlink()
    try:
        r = client.post("/api/jobs", json={"project": pid, "kind": "purge_label", "params": {"label": "a"}})
        assert r.status_code == 404
    finally:
        path.write_bytes(saved)
