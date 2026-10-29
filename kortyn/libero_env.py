"""LIBERO environment for the Kortyn Agent (phase 3), built exactly like LeRobot's evaluation env.

LeRobot's ``LiberoEnv`` (``lerobot/envs/libero.py``) wraps LIBERO's ``OffScreenRenderEnv``: per task a
BDDL scene, the fixed initial states of the benchmark (episode ``j`` of a task starts from init state ``j``),
``env.seed(seed)``, ten settling no-op steps after every reset, relative end-effector control at 20 Hz and
``check_success`` as termination. Its processor (``LiberoProcessorStep``) turns the raw observation into
the policy input: both camera images rotated by 180 degrees, and an 8-D state (end-effector position,
orientation as axis-angle, gripper finger positions).

This wrapper keeps LeRobot's class for everything except the cameras. robosuite renders every camera
in every control step, which costs ~80 ms per 360x360 frame in software rendering against ~16 ms for the
rest of a control step. The wrapper builds the simulator without camera observables and renders a frame only
when the agent sends an observation, with robosuite's own camera sensor code. One subtlety decides whether
the images match: robosuite samples its observables, cameras included, at the *first* physics substep of a
control period, not at the end of the step, so the image LeRobot's env returns shows the scene one control
step (minus 2 ms) before the state it reports. The wrapper therefore registers an inactive observable that
copies the simulator state (``mjData``) exactly when robosuite samples, and renders from that copy. Rendering
does not change the simulation, so trajectories and images equal those of LeRobot's env bit for bit
(checked by ``tests/test_libero_env.py``). As for Meta-World, lazy images require ``n_obs_steps = 1``.

Several simulators in one process (fleet validation) need two guards, both about robosuite's GL contexts:
a render makes its simulator's context current first, and so does every hard reset or close, because
robosuite frees a destroyed simulator's render resources in whatever context is current
(:func:`_activate_render_context`).

LIBERO asks interactively for a data directory on its first import and downloads object meshes from the Hub
without a pinned revision. :func:`prepare_libero` answers the question non-interactively with LIBERO's
defaults (what answering "N" writes) and fetches the meshes at a pinned revision beforehand.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kortyn.config import EnvConfig, LiberoEnvConfig
from kortyn.metaworld_env import configure_rendering
from kortyn.sim_env import EnvError, StepResult
from kortyn.typing_utils import Array, FloatArray

# Hub dataset with LIBERO's object meshes; LIBERO's own download (``get_assets_path``) is not pinned.
ASSETS_REPO_ID = "lerobot/libero-assets"
ASSET_SUBDIRS = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects")
GRIPPER_OPEN = -1.0  # LIBERO's gripper command for "open" (LeRobot's settling no-op uses it as well)
ACTION_DIM = 7  # end-effector delta position (3) + delta rotation (3) + gripper (1)
# LIBERO-Long (``libero_10``) instructions in LIBERO's task order: reports name tasks without LIBERO
LIBERO_10_TASKS = (
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
)
# Simulators kept alive per env object. One LIBERO simulator with its render context holds ~1.1-1.5 GB;
# consecutive episodes of a run mostly share a task, so rebuilding on a task switch is rare.
MAX_CACHED_TASKS = 1


def disable_telemetry() -> None:
    """No usage reporting from libraries LIBERO pulls in (Hub telemetry, Weights & Biases)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")


def libero_package_root() -> Path:
    """Directory of LIBERO's ``libero/libero`` package, without importing it (its import prompts)."""
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise EnvError("LIBERO is not installed: run with the dependency group 'libero' (see README)")
    return Path(next(iter(spec.submodule_search_locations))) / "libero"


def libero_default_paths(root: Path) -> dict[str, str]:
    """LIBERO's ``get_default_path_dict()``: the paths its first-run prompt writes when answered with "N"."""
    base = str(root)
    return {
        "benchmark_root": base,
        "bddl_files": os.path.join(base, "./bddl_files"),
        "init_states": os.path.join(base, "./init_files"),
        "datasets": os.path.join(base, "../datasets"),
        "assets": os.path.join(base, "./assets"),
    }


def ensure_libero_config(config_dir: Path | None = None) -> Path:
    """Write LIBERO's config file unless it exists (LIBERO would otherwise prompt on import)."""
    directory = config_dir or Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    config_file = directory / "config.yaml"
    if config_file.exists():
        return config_file
    directory.mkdir(parents=True, exist_ok=True)
    temporary = config_file.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(libero_default_paths(libero_package_root())), encoding="utf-8")
    temporary.replace(config_file)
    return config_file


def libero_assets_dir() -> Path:
    """Where LIBERO looks for downloaded meshes when its package ships none (``download_utils``)."""
    return Path.home() / ".cache" / "libero" / "assets"


def ensure_libero_assets(revision: str, download: Callable[..., Any] | None = None) -> Path:
    """Fetch LIBERO's object meshes at ``revision`` unless the package or the cache already has them."""
    if (libero_package_root() / "assets").exists():
        return libero_package_root() / "assets"
    target = libero_assets_dir()
    if all((target / subdir).is_dir() for subdir in ASSET_SUBDIRS):
        return target
    if download is None:
        from huggingface_hub import snapshot_download

        download = snapshot_download
    download(ASSETS_REPO_ID, repo_type="dataset", revision=revision, local_dir=str(target))
    missing = [subdir for subdir in ASSET_SUBDIRS if not (target / subdir).is_dir()]
    if missing:
        raise EnvError(f"LIBERO assets at {target} lack {missing} after the download")
    return target


def prepare_libero(settings: LiberoEnvConfig) -> None:
    """Everything LIBERO needs before its first import: rendering backend, config file, pinned meshes."""
    disable_telemetry()
    configure_rendering()
    ensure_libero_config()
    ensure_libero_assets(settings.assets_revision)


def task_language(suite: str, task_id: int) -> str:
    """Instruction of a LIBERO task: built in for ``libero_10``, else from LIBERO if installed, else ""."""
    if suite == "libero_10" and 0 <= task_id < len(LIBERO_10_TASKS):
        return LIBERO_10_TASKS[task_id]
    try:
        from lerobot.envs.libero import _get_suite

        return str(_get_suite(suite).get_task(task_id).language)
    except (ImportError, AttributeError, IndexError, KeyError, ValueError):  # no LIBERO, unknown suite/task
        return ""


def episode_slot(settings: LiberoEnvConfig, seed: int) -> tuple[int, int, int]:
    """``(task_id, init_state_id, env_seed)`` of the run episode with this seed (see ``LiberoEnvConfig``)."""
    index = (seed - settings.seed_base) % settings.episodes
    task_id = settings.task_ids[index // settings.episodes_per_task]
    within = index % settings.episodes_per_task
    return task_id, settings.init_state_offset + within, settings.env_seed_start + within


class LazyObservation(dict[str, Array]):
    """Observation whose camera images are rendered on first access, from the step they belong to."""

    def __init__(self, env: LiberoEnv, values: dict[str, Array], step_index: int) -> None:
        super().__init__(values)
        self._env = env
        self._step_index = step_index

    def __contains__(self, key: object) -> bool:
        return key in self._env.image_keys or super().__contains__(key)

    def __missing__(self, key: str) -> Array:
        if key not in self._env.image_keys:
            raise KeyError(key)
        if self._env.step_index != self._step_index:
            raise EnvError(
                f"image of step {self._step_index} requested at step {self._env.step_index}: lazily rendered "
                "observations can only be sent while they are the newest one (n_obs_steps = 1)"
            )
        image = self._env.render_image(key)
        self[key] = image
        return image


class StateSnapshot:
    """Copy of the simulator state taken whenever robosuite samples its observables.

    Registered as an inactive robosuite observable with the camera sampling rate, so ``capture`` runs at the
    very substep at which robosuite would render the cameras. ``render`` swaps the copy into the simulator,
    runs a camera sensor and restores the current state bit for bit.
    """

    def __init__(self, robosuite_env: Any) -> None:
        self._env = robosuite_env
        self._model: Any = None
        self._sampled: Any = None
        self._current: Any = None
        self.captures = 0

    def _buffers(self) -> tuple[Any, Any, Any]:
        import mujoco

        model = self._env.sim.model._model
        if model is not self._model:  # a hard reset builds a new model: new buffers
            self._model = model
            self._sampled = mujoco.MjData(model)
            self._current = mujoco.MjData(model)
        return self._model, self._sampled, self._current

    def capture(self, obs_cache: dict[str, Any]) -> Array:
        import mujoco

        model, sampled, _current = self._buffers()
        mujoco.mj_copyData(sampled, model, self._env.sim.data._data)
        self.captures += 1
        return np.zeros(1)

    def render(self, sensor: Callable[[dict[str, Any]], Any]) -> Array:
        import mujoco

        if self.captures == 0:
            raise EnvError("no simulator state captured yet")
        model, sampled, current = self._buffers()
        live = self._env.sim.data._data
        mujoco.mj_copyData(current, model, live)
        mujoco.mj_copyData(live, model, sampled)
        # robosuite renders into whatever GL context is current and makes its own current only when it creates
        # it: with several simulators in one process (fleet validation) the last one created would be drawn.
        context = getattr(self._env.sim, "_render_context_offscreen", None)
        if context is not None:
            context.gl_ctx.make_current()
        try:
            return np.asarray(sensor({}))
        finally:
            mujoco.mj_copyData(live, model, current)


SNAPSHOT_OBSERVABLE = "kortyn_render_state"


def _activate_render_context(task_env: Any) -> None:
    """Make the GL context of ``task_env``'s live simulator current, before robosuite destroys it.

    A hard reset or close frees the simulator's MuJoCo render resources (``mjr_freeContext``: textures,
    vertex buffers, offscreen framebuffer) in whatever GL context is current, not in the simulator's own.
    With several simulators in one process that is usually the context of the robot that rendered last:
    its resources would be deleted under the same ids, and its images stay wrong until its own next
    reset. Without a live simulator (not built yet) there is nothing to free.
    """
    inner = getattr(task_env, "_env", None)
    sim = getattr(getattr(inner, "env", None), "sim", None)
    context = getattr(sim, "_render_context_offscreen", None)
    if context is not None:
        context.gl_ctx.make_current()


def _reset_property_samplers(task_env: Any) -> None:
    """Undo LIBERO's accumulation of object-property samplers across hard resets.

    LIBERO's BDDL environments create ``object_property_initializers`` once (``__init__``) but append an
    open/close or turn-on/off sampler per articulated initial state in ``_load_model``, i.e. on every hard
    reset. Each sampler draws one number from numpy's global RNG in ``_reset_internal``, before the fixtures
    are placed. After k hard resets the fixture placement therefore uses draws shifted by k, and a fixture
    such as the stove of LIBERO-10 task 8 (placed by writing ``model.body_pos``, which the init state does
    not restore) stands where the simulator's history puts it: the scene depended on how many episodes
    the process had run. Emptying the list before the seeded reset leaves exactly the samplers of that
    reset's model load, so the scene is a function of the episode seed alone. Scenes without articulated
    initial states (e.g. task 0) are unchanged.
    """
    domain = getattr(getattr(task_env, "_env", None), "env", None)
    if domain is not None and hasattr(domain, "object_property_initializers"):
        domain.object_property_initializers = []


def _lerobot_env_class() -> type[Any]:
    """LeRobot's ``LiberoEnv`` with a simulator that skips camera observables (rendered on demand)."""
    from lerobot.envs.libero import LiberoEnv as LeRobotLiberoEnv
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.observables import Observable, sensor

    class NoCameraLiberoEnv(LeRobotLiberoEnv):  # type: ignore[misc]
        snapshot: StateSnapshot | None = None
        _env: Any

        def _ensure_env(self) -> None:
            if self._env is not None:
                return
            # LeRobot's construction with use_camera_obs=False: same scene, cameras, controller and resets.
            env = OffScreenRenderEnv(
                bddl_file_name=self._task_bddl_file,
                camera_heights=self.observation_height,
                camera_widths=self.observation_width,
                control_freq=self.control_freq,
                hard_reset=self.hard_reset,
                use_camera_obs=False,
            )
            env.reset()
            robosuite_env = env.env
            snapshot = StateSnapshot(robosuite_env)

            def capture(obs_cache: dict[str, Any]) -> Array:  # plain function: robosuite tags it
                return snapshot.capture(obs_cache)

            sensor(modality="kortyn")(capture)
            robosuite_env.add_observable(
                Observable(
                    name=SNAPSHOT_OBSERVABLE,
                    sensor=capture,
                    sampling_rate=robosuite_env.control_freq,
                    active=False,
                )
            )
            self.snapshot = snapshot
            self._env = env

        def _format_raw_obs(self, raw_obs: Any) -> dict[str, Any]:
            # LeRobot's robot_state, without the images (not rendered here).
            assert self._env is not None
            eef_pos = raw_obs.get("robot0_eef_pos")
            eef_quat = raw_obs.get("robot0_eef_quat")
            gripper_qpos = raw_obs.get("robot0_gripper_qpos")
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise EnvError("LIBERO observation lacks the end-effector or gripper state")
            return {
                "robot_state": {
                    "eef": {
                        "pos": eef_pos,
                        "quat": eef_quat,
                        "mat": self._env.robots[0].controller.ee_ori_mat,
                    },
                    "gripper": {"qpos": gripper_qpos, "qvel": raw_obs.get("robot0_gripper_qvel")},
                    "joints": {
                        "pos": raw_obs.get("robot0_joint_pos"),
                        "vel": raw_obs.get("robot0_joint_vel"),
                    },
                }
            }

    return NoCameraLiberoEnv


def lerobot_state(robot_state: dict[str, Any]) -> FloatArray:
    """The 8-D policy state as LeRobot computes it (``preprocess_observation``, ``LiberoProcessorStep``)."""
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep
    from lerobot.utils.constants import OBS_STATE

    def convert(value: Any) -> Any:
        # LeRobot's evaluation steps a vector env: every array carries a batch dimension of 1.
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return torch.from_numpy(np.asarray(value)[None]) if isinstance(value, np.ndarray) else value

    processed = LiberoProcessorStep()._process_observation({"observation.robot_state": convert(robot_state)})
    state = processed[OBS_STATE]
    return np.asarray(state[0].numpy(), dtype=np.float32)


class LiberoEnv:
    """LIBERO tasks of one suite with LeRobot's evaluation semantics and lazily rendered cameras.

    One simulator per task (built on first use and kept); ``reset(seed)`` selects task, init state and
    simulator seed of that run episode (:func:`episode_slot`) and sets ``task_description``, the language
    instruction the agent sends with every request.
    """

    def __init__(self, config: EnvConfig) -> None:
        if config.kind != "libero" or config.libero is None:
            raise EnvError("LiberoEnv needs env.kind 'libero' with an env.libero section")
        settings = config.libero
        prepare_libero(settings)
        try:
            from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _get_suite
        except ImportError as exc:
            raise EnvError(f"LIBERO is not installed (lerobot[libero]): {exc}") from exc
        self.config = config
        self.settings = settings
        self._suite = _get_suite(settings.suite)
        n_tasks = len(self._suite.tasks)
        unknown = [task_id for task_id in settings.task_ids if task_id >= n_tasks]
        if unknown:
            raise EnvError(f"task ids {unknown} out of range: suite {settings.suite} has {n_tasks} tasks")
        suite_steps = TASK_SUITE_MAX_STEPS.get(settings.suite)
        if suite_steps is not None and config.max_episode_steps != suite_steps:
            raise EnvError(
                f"env.max_episode_steps={config.max_episode_steps}, but LeRobot evaluates {settings.suite} "
                f"with {suite_steps} steps"
            )
        if abs(config.fps - 20.0) > 1e-9:
            raise EnvError(f"LIBERO runs at robosuite's control rate of 20 Hz, env.fps is {config.fps}")
        self._env_class = _lerobot_env_class()
        self._envs: dict[int, Any] = {}
        self._cameras = dict(settings.cameras)
        self.image_keys = frozenset(self._cameras.values())
        self._camera_by_key = {key: camera for camera, key in self._cameras.items()}
        self._sensors: dict[tuple[int, str], Callable[[dict[str, Any]], Array]] = {}
        self.action_dim = ACTION_DIM
        self.task_id = settings.task_ids[0]
        self.task_description = self._suite.get_task(self.task_id).language
        self.init_state_id = settings.init_state_offset
        self.step_index = 0
        self.renders = 0
        self._last_gripper = GRIPPER_OPEN

    @property
    def fps(self) -> float:
        return self.config.fps

    @property
    def max_episode_steps(self) -> int:
        return self.config.max_episode_steps

    def _task_env(self, task_id: int) -> Any:
        env = self._envs.get(task_id)
        if env is None:
            # A simulator per task costs memory (scene, meshes, render context): keep only a few.
            while len(self._envs) >= MAX_CACHED_TASKS:
                evicted = self._envs.pop(next(iter(self._envs)))
                _activate_render_context(evicted)
                evicted.close()
            settings = self.settings
            env = self._env_class(
                task_suite=self._suite,
                task_id=task_id,
                task_suite_name=settings.suite,
                camera_name=list(self._cameras),
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                observation_height=settings.observation_height,
                observation_width=settings.observation_width,
                init_states=True,
                episode_index=0,
                n_envs=1,
                num_steps_wait=settings.num_steps_wait,
                control_freq=round(self.config.fps),
                control_mode=settings.control_mode,
                hard_reset=settings.hard_reset,
            )
            self._envs[task_id] = env
        else:
            self._envs[task_id] = self._envs.pop(task_id)  # most recently used last
        return env

    @property
    def _sim_env(self) -> Any:
        inner = self._task_env(self.task_id)._env
        if inner is None:
            raise EnvError("LIBERO env used before reset")
        return inner.env  # robosuite environment behind LIBERO's OffScreenRenderEnv

    def render_image(self, key: str) -> Array:
        """Camera image of the newest observation, as LeRobot feeds it to the policy (rotated 180 degrees)."""
        camera = self._camera_by_key[key]
        task_env = self._task_env(self.task_id)
        snapshot = task_env.snapshot
        if snapshot is None or task_env._env is None:
            raise EnvError("LIBERO env used before reset")
        sensor = self._sensors.get((self.task_id, camera))
        if sensor is None:
            camera_name = camera.removesuffix("_image")
            sensors, _names = self._sim_env._create_camera_sensors(
                camera_name,
                cam_w=self.settings.observation_width,
                cam_h=self.settings.observation_height,
                cam_d=False,
                cam_segs=None,
            )
            sensor = sensors[0]
            self._sensors[(self.task_id, camera)] = sensor
        frame = snapshot.render(sensor)
        if frame.dtype != np.uint8 or frame.ndim != 3:
            raise EnvError(f"camera {camera} rendered {frame.dtype} {frame.shape}, expected uint8 (H, W, 3)")
        self.renders += 1
        return np.ascontiguousarray(frame[::-1, ::-1])  # LiberoProcessorStep: flip height and width

    def _observation(self, raw: dict[str, Any]) -> LazyObservation:
        state = lerobot_state(raw["robot_state"])
        return LazyObservation(self, {self.settings.state_key: state}, self.step_index)

    def reset(self, seed: int) -> LazyObservation:
        task_id, init_state_id, env_seed = episode_slot(self.settings, seed)
        self.task_id = task_id
        env = self._task_env(task_id)
        n_states = None if env._init_states is None else len(env._init_states)
        if n_states is not None and init_state_id >= n_states:
            raise EnvError(f"task {task_id} has {n_states} init states, episode needs state {init_state_id}")
        env.init_state_id = init_state_id  # LeRobot: episode j of a task starts from init state j
        env._ensure_env()  # build the simulator on first use, so that its samplers can be reset
        _reset_property_samplers(env)  # the scene must not depend on earlier episodes of this simulator
        _activate_render_context(env)  # the hard reset frees the old simulator's render resources
        raw, _info = env.reset(seed=env_seed)
        # A hard reset rebuilds the simulator: camera sensors bound to the old one are invalid.
        self._sensors = {}
        self.task_description = env.task_description
        self.init_state_id = init_state_id
        self.step_index = 0
        self._last_gripper = GRIPPER_OPEN
        return self._observation(raw)

    def step(self, action: FloatArray) -> StepResult:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (self.action_dim,):
            raise EnvError(f"action must have shape ({self.action_dim},), got {values.shape}")
        raw, reward, terminated, truncated, info = self._task_env(self.task_id).step(values)
        self.step_index += 1
        self._last_gripper = float(values[-1])
        success = bool(info.get("is_success", False))
        return StepResult(
            observation=self._observation(raw),
            reward=float(reward),
            terminated=bool(terminated) or success,
            truncated=bool(truncated),
            success=success,
            coverage=None,
        )

    def hold_action(self, observation: dict[str, Array]) -> FloatArray:
        """Stand still: zero end-effector displacement and rotation, the gripper keeps its last command.

        Before the first action of an episode that is LeRobot's settling no-op (gripper open).
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[-1] = np.float32(self._last_gripper)
        return action

    def close(self) -> None:
        for env in self._envs.values():
            _activate_render_context(env)
            env.close()
        self._envs.clear()
        self._sensors.clear()
