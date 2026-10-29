# Kortyn – summary

Kortyn is an open-source testbed that measures how network latency and a shared GPU server affect learned robot
policies. It runs LeRobot policies against modeled network and queueing delays, in real time or in a validated
simulation-time mode that needs only a CPU. Across five pre-registered phases:

- SmolVLA on two long-horizon LIBERO tasks keeps its success up to about 200 ms of end-to-end latency, then drops
  steeply (47% at 0 ms, 5% at 1,000 ms).
- LeRobot's real-time chunking did not stretch this curve, with long or with short guidance.
- A free Tesla T4 serves a request in 91 ms (fp16, `torch.compile`) but keeps the p99 latency within 150 ms for
  1 robot only. By our pre-registered rule, a shared GPU "Brain" for robot fleets is not viable on it.
- Two infrastructure faults changed earlier results: a rendering fault in multi-robot processes, and a LIBERO bug
  that moved one task's stove with the simulator's history. Both are fixed, and the report shows what changed.

All negative results are reported. Code, data, the technical report and a one-command capacity tool are released
under the Apache-2.0 license: <https://github.com/erithy25/Kortyn-Robotics>
