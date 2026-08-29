"""Smoke checks for mshowgui (no UI, no camera): run in the media env."""
import threading

import numpy as np

import mshowgui as g


# rgba_floats: BGR -> flat RGBA floats, channel order and scale correct
bgr = np.zeros((2, 2, 3), dtype=np.uint8)
bgr[0, 0] = (1, 2, 3)  # B=1, G=2, R=3
out = g.rgba_floats(bgr).reshape(2, 2, 4)
assert out.dtype == np.float32
assert np.allclose(out[0, 0], [3/255, 2/255, 1/255, 1.0]), out[0, 0]
assert np.allclose(out[1, 1], [0, 0, 0, 1.0])
assert g.rgba_floats(bgr).shape == (16,)  # flat for dpg.set_value

# SharedState: slots and flags are thread-consistent
state = g.SharedState()
state.publish(ar=np.zeros((4, 4, 3), dtype=np.uint8))
assert state.get("ar") is not None and state.get("fbo") is None
hits = []


def reader():
    for _ in range(1000):
        hits.append(state.get("denoise"))
    state.set_flag("denoise", True)


t = threading.Thread(target=reader)
t.start()
for i in range(1000):
    state.publish(denoise=i)
t.join()
assert len(hits) == 1000          # no torn reads / exceptions under churn
assert state.get_flag("denoise") is True

print("check_mshowgui: all ok")
