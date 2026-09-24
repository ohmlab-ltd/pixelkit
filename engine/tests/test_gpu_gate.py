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
