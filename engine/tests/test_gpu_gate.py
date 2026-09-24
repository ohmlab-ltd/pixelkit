"""The GPU gate's hand-over: a long holder (training on the labelling
device) lets waiting clicks and labelling images use the card between
its steps, then takes the gate back."""
import asyncio


def test_hand_over_lets_waiting_work_run_and_takes_the_gate_back(app_module):
    gate = app_module.PriorityGPUGate()
    order = []

    async def click():
        async with gate.interactive():
            order.append("click")

    async def labelling_image():
        async with gate:
            order.append("label")

    async def embedding():
        async with gate.background():
            order.append("background")

    async def main():
        async with gate:  # the training run
            assert not gate.contended()
            await gate.hand_over()  # nobody waits: nothing happens
            tasks = [asyncio.create_task(t()) for t in (embedding, labelling_image, click)]
            await asyncio.sleep(0)  # all three queue behind the run
            assert gate.contended()
            order.append("step")
            await gate.hand_over()
            order.append("step")
            # Background work waits for the end of the run, not a step.
            assert not gate.contended()
            assert gate.contended(gate.P_BACKGROUND)
        await asyncio.gather(*tasks)

    asyncio.run(main())
    assert order == ["step", "click", "label", "step", "background"]


def test_a_reference_upload_embeds_off_the_event_loop_under_the_gate(client, app_module, monkeypatch):
    import io
    import json

    import numpy as np
    import v2_dinov2
    from PIL import Image

    seen = []

    def encode(squares):
        try:
            asyncio.get_running_loop()
            seen.append("event loop")
        except RuntimeError:
            seen.append("thread")
        seen.append("gate held" if not app_module.state["gpu_lock"]._free else "gate free")
        return np.zeros((len(squares), 4), np.float32)

    monkeypatch.setattr(v2_dinov2, "is_loaded", lambda: True)
    monkeypatch.setattr(v2_dinov2, "encode_images_batch", encode)
    r = client.post("/api/v2/projects", data={"name": "Refs", "labels": json.dumps(["bolt"])})
    pid = r.json()["project_id"]
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 120, 40)).save(buf, "JPEG")
    r = client.post(f"/api/v2/projects/{pid}/references",
                    files={"image": ("ref.jpg", buf.getvalue(), "image/jpeg")},
                    data={"detections": json.dumps([{"label": "bolt", "score": 1.0,
                                                     "box": [8, 8, 40, 40], "mask": None}])})
    assert r.status_code == 200, r.text
    assert seen == ["thread", "gate held"]
