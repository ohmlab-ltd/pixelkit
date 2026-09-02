"""Neuro N6 target facts PixelKit needs at export time.

NeuroN6_MemPool.mpool is a verbatim copy of the board's memory-pool file
from the Arduino core (system/npu/NeuroN6_MemPool.mpool). It is what ST
Edge AI compiles against, so the weights/activation numbers an export
reports are the ones the real build will see. Keep it in sync with the
core; the pool base/size have been stable since the board shipped.
"""
