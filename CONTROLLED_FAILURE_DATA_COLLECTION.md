# Controlled Failure Data Collection for Action-Conditioned World Models

## Goal

Build matched trajectory groups that differ by one controlled failure mechanism,
then identify cases where:

```text
simulator ground truth = failure
world-model prediction = visually successful rollout
```

The experimental unit should be a matched set, not an unrelated collection of
successful and failed episodes:

```text
same task
same initial scene seed
same instruction
same embodiment
same cameras
same domain-randomization realization
same successful expert trajectory before intervention
one changed intervention variable
```

This repository's standard collector is designed to retain successful expert
episodes. It must not be used unchanged for intentional failures:

- `script/collect_data.py` accepts planning episodes only when both
  `plan_success` and `check_success()` are true.
- During recording it asserts `check_success()` again.
- Therefore, simply changing a task so that `check_success()` returns true for
  failed variants would corrupt the ground-truth definition.

The recommended design is to preserve the standard collector for generating
successful reference demonstrations and add a separate controlled-failure
replay/recording path.

## Which Files to Copy or Adapt

### 1. Copy the task configuration template

Copy:

```text
task_config/_config_template.yml
```

Suggested destination:

```text
task_config/controlled_failure_clean.yml
```

Use this configuration for the matched experiment. Initially set all nuisance
randomization to fixed values:

```yaml
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
```

Keep both head and wrist cameras enabled. Enable depth and actor segmentation
if storage permits, because they can help determine whether a false-success
prediction comes from RGB ambiguity or actual state ambiguity:

```yaml
camera:
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  depth: true
  endpose: true
  qpos: true
  actor_segmentation: true
```

`save_freq` controls temporal subsampling of simulator control steps. Use the
same value for every member of a matched group. A smaller value captures slips,
brief contact, and post-success instability more reliably.

Do not modify these shared configuration files unless a new robot or camera
model is actually required:

```text
task_config/_embodiment_config.yml
task_config/_camera_config.yml
```

### 2. Copy the target task implementation

Copy the task that you want to study:

```text
envs/<task_name>.py
```

For example:

```text
envs/beat_block_hammer.py
envs/move_can_pot.py
envs/place_object_stand.py
```

Suggested destinations:

```text
envs/beat_block_hammer_controlled.py
envs/move_can_pot_controlled.py
```

The class name must match the new module name because the collection scripts
dynamically import `envs.<task_name>` and retrieve a class with the same name.

This task copy is the main place to expose stable task semantics:

- `load_actors()` defines the initial scene and object identities.
- `play_once()` defines phases such as approach, grasp, transport, placement,
  release, and retreat.
- `check_success()` defines simulator ground truth.

Keep the original success predicate intact or make it stricter and more
diagnostic. Add a separate method such as `get_failure_metrics()` to report
continuous state:

```python
{
    "final_position_error_m": ...,
    "final_orientation_error_deg": ...,
    "target_contact": ...,
    "object_gripper_contact": ...,
    "object_linear_speed_mps": ...,
    "object_angular_speed_radps": ...,
    "object_inside_target": ...,
    "stable_for_steps": ...,
}
```

Do not implement every failure by duplicating the full `play_once()` method.
Split the copied task into named phases and invoke an intervention hook between
phases:

```text
approach -> grasp -> lift -> transport -> place -> release -> verify
```

Example conceptual interface:

```python
self.maybe_intervene("after_grasp")
self.maybe_intervene("before_place")
self.maybe_intervene("after_release")
```

This gives every intervention an explicit phase and makes matching auditable.

### 3. Copy the collection driver

Copy:

```text
script/collect_data.py
```

Suggested destination:

```text
script/collect_controlled_failures.py
```

This is the most important driver to adapt. Retain its useful infrastructure:

- dynamic task loading;
- YAML, embodiment, and camera configuration;
- deterministic scene recreation from a seed;
- loading saved expert joint paths;
- observation recording;
- HDF5 and video conversion.

Change the workflow to:

1. Load seeds and expert trajectories produced by the normal collector.
2. Recreate the same initial scene for every variant.
3. Replay the successful reference variant unchanged.
4. Replay each failed variant with exactly one intervention.
5. Record the final `check_success()` result without asserting success.
6. Reject a nominal failed variant if `check_success()` is still true.
7. Save intervention metadata and continuous ground-truth metrics.

Specifically, the controlled-failure driver must not retain this behavior from
the original collector:

```python
assert TASK_ENV.check_success(), "Collect Error"
```

Instead, enforce the expected label:

```text
success reference: check_success() must be true
failed variant: check_success() must be false
```

Use an explicit manifest rather than encoding experimental meaning only in
directory names.

### 4. Adapt the base recorder only if richer state is needed

Relevant file:

```text
envs/_base_task.py
```

Prefer not to copy the entire base class. Extend the controlled task or add
small shared hooks because all tasks inherit this file.

The useful existing boundaries are:

- `get_obs()` constructs each recorded observation.
- `_take_picture()` writes frame PKLs.
- `save_traj_data()` and `load_tran_data()` store/reload expert joint paths.
- `merge_pkl_to_hdf5_video()` creates HDF5 and an MP4.
- `take_dense_action()` is the lowest-level replay point for arm and gripper
  controls.

Adapt `get_obs()` if per-frame diagnostics are required. Useful additions are:

```text
sim_state/target_object_pose
sim_state/target_pose
sim_state/target_object_velocity
sim_state/contact_flags
sim_state/intervention_active
sim_state/control_step
```

Use `take_dense_action()` for action-noise, gripper-command, or control-timing
interventions. Use task phase hooks for semantic interventions such as wrong
object, near-miss placement, or post-success disturbance.

### 5. Reuse the HDF5/video converter

Relevant file:

```text
envs/utils/pkl2hdf5.py
```

The converter recursively aggregates matching keys from each frame, so
additional numeric arrays inserted into `get_obs()` can flow into HDF5 without
creating a separate storage system.

The standard MP4 contains only head-camera RGB. For camera-ambiguity
experiments, either:

- use the HDF5 camera streams directly; or
- extend the converter to emit separate head, left-wrist, right-wrist, and
  synchronized mosaic videos.

Do not use the head-camera MP4 alone as the canonical record of a multi-camera
experiment.

### 6. Copy a model adapter only as an interface pattern

For connecting the action-conditioned world model, copy:

```text
policy/Your_Policy/deploy_policy.py
```

Suggested destination:

```text
policy/Your_World_Model/infer_rollout.py
```

The existing file is a policy interface, not a world-model evaluator, but its
observation encoding and model-loading separation are useful patterns. Your
adapter should consume:

```text
initial observation or observation context
recorded action trajectory
instruction, if the model is language-conditioned
camera selection
```

It should save:

```text
predicted rollout frames
predicted success score or label
model checkpoint/config identifier
conditioning cameras
inference seed and sampling settings
```

`script/eval_policy.py` may be read for `eval_result` video organization and
model loading, but it should not be copied as the primary data generator. It
evaluates a policy online and selects seeds that the expert can solve; the
experiment here requires offline inference over already matched action
trajectories.

## Documents to Read, Not Copy

### `NOTE.md`

Use this as the main description of data provenance. It explains:

- the planning and recording passes;
- the relation among `seed.txt`, `_traj_data/episodeN.pkl`, HDF5, and MP4;
- camera and robot configuration sources;
- frame sampling and output fields;
- known replay and custom-output-path limitations.

It is background documentation, not an experiment specification.

### `README.md`

Use the data-collection command and high-level task/configuration references.
It does not describe intentional failure collection.

### Policy-specific READMEs

Files such as:

```text
policy/openvla-oft/openvla-oft.md
policy/GO1/README.md
```

are useful only for examples of policy evaluation output under `eval_result`.
They do not preserve matched failed trajectories or causal intervention
metadata.

## Controlled Failure Taxonomy and Implementation Location

| Failure type | One-variable intervention | Best implementation point | Required ground truth |
| --- | --- | --- | --- |
| Late-stage near-miss | Offset final target just beyond the success tolerance | Controlled task, before place | Final position error and target containment/contact |
| Grasp illusion | Reduce closure, open briefly, or perturb object after apparent grasp | Phase hook after grasp or gripper control in `take_dense_action()` | Object-gripper contact, object height, drop time |
| Occluded failure | Trigger a drop while the target object is hidden in one selected view | Phase hook plus camera visibility check | Drop state and per-camera visibility/segmentation |
| Temporal truncation | Stop the recorded/model action sequence before verification | Driver/model input construction | Truncation step and full-trajectory GT outcome |
| Wrong object | Execute the manipulation on a visually similar distractor | Controlled task actor binding | Intended object ID and manipulated object ID |
| Pose tolerance failure | Apply position/orientation error just outside the predicate boundary | Controlled task, placement target | Position and orientation error |
| Post-success instability | Disturb the object or continue simulation until it falls/moves | Hook after transient success | First-success step, final-success label, stability duration |
| Collision-induced failure | Add one controlled obstacle/contact or lateral perturbation | Controlled task during transport/place | Collision pair, impulse/contact time, final pose |
| Action-noise failure | Add a localized action delta over a specified step interval | Replay driver or `take_dense_action()` | Original and executed action, norm, time range |
| Camera-only ambiguity | Keep simulator trajectory fixed and vary visible camera streams | Dataset/view construction, not physics | Identical trajectory ID and selected cameras |

### Intervention calibration

Each intervention should be calibrated relative to the task's actual
`check_success()` boundary. For example, if position tolerance is `0.03 m`,
near-miss offsets should include values just below and just above the boundary:

```text
0.025 m, 0.030 m, 0.035 m, 0.050 m
```

This distinguishes an evaluator-boundary problem from a broad dynamics-model
failure. The same principle applies to orientation, contact duration, release
timing, action perturbation magnitude, and stability duration.

## Recommended Output Layout

Do not place controlled failures inside the standard successful-demo directory.
Use a separate root:

```text
data_controlled_failures/
  <task_name>/
    <task_config>/
      seed_000012/
        manifest.json
        success_demo/
          trajectory.hdf5
          head.mp4
          left_wrist.mp4
          right_wrist.mp4
          model_rollout.mp4
          model_prediction.json
        fail_near_miss_position/
          trajectory.hdf5
          intervention.json
          ground_truth.json
          model_rollout.mp4
          model_prediction.json
        fail_slip_after_grasp/
        fail_wrong_object/
        fail_occluded_drop/
        fail_truncated_before_success/
        fail_post_success_instability/
```

Every variant should point back to the same successful expert source:

```json
{
  "pair_id": "move_can_pot__controlled_failure_clean__seed_000012",
  "task": "move_can_pot_controlled",
  "source_task": "move_can_pot",
  "task_config": "controlled_failure_clean",
  "seed": 12,
  "variant": "fail_slip_after_grasp",
  "source_expert_trajectory": "data/move_can_pot/demo_clean/_traj_data/episode0.pkl",
  "instruction": "Move the can next to the pot.",
  "intervention": {
    "type": "gripper_release",
    "phase": "after_grasp",
    "control_step": 74,
    "parameters": {
      "release_fraction": 0.25,
      "duration_steps": 8
    }
  },
  "ground_truth": {
    "expected_success": false,
    "check_success": false,
    "failure_type": "grasp_illusion",
    "reason": "can slipped before final placement"
  },
  "cameras": ["head_camera", "left_camera", "right_camera"],
  "domain_randomization": {
    "background": "fixed",
    "lighting": "fixed",
    "clutter": "fixed",
    "table_height": "fixed"
  }
}
```

Store the executed action trajectory, not only the intended intervention
parameters. For action-conditioned inference, the model must receive the
commands that were actually applied to the simulator.

## Experimental Matrix

For each selected task and seed:

```text
A. success_demo
B. fail_near_miss_position
C. fail_wrong_orientation
D. fail_slip_after_grasp
E. fail_wrong_object
F. fail_occluded_drop
G. fail_truncated_before_success
H. fail_post_success_instability
I. fail_collision
J. fail_local_action_noise
```

Run world-model inference on every variant, including the successful reference.
Record at least:

```text
simulator ground-truth label
scripted/human failure reason
world-model predicted rollout
success evaluator score on simulator video
success evaluator score on predicted video
per-camera evaluator score, where applicable
intervention type, time, magnitude, and phase
final object/target poses
contact and stability metrics
```

The central false-success indicator is:

```text
false_success =
    (simulator_ground_truth == false)
    and (predicted_rollout_success == true)
```

Also separate two sources of error:

```text
world-model error:
  predicted video depicts success although the simulator failed

evaluator error:
  predicted video still depicts failure, but the success evaluator labels it success
```

Human review or a task-specific scripted state inspector should annotate a
small validation subset to verify this distinction.

## Collection Procedure

1. Use the normal `collect_data.sh` path to produce successful expert seeds and
   `_traj_data` files.
2. Freeze a list of accepted source seeds. Never search for different seeds per
   failure type.
3. For each seed, recreate the scene and record an unchanged success reference.
4. Recreate the same scene again for each intervention.
5. Apply one intervention at a named phase or control-step interval.
6. Continue simulation through a fixed verification horizon. This is required
   for slips and post-success instability.
7. Compute task ground truth from simulator state after the verification
   horizon.
8. Reject failed variants that accidentally remain successful, but retain a
   calibration log so intervention strength is not silently cherry-picked.
9. Run world-model inference with the exact executed action trajectory.
10. Score simulator and predicted rollouts with the same evaluator and save all
    raw scores.
11. Only after the controlled experiment is complete, sweep background,
    lighting, clutter, table height, language, and camera perturbations.

## Confounds and Repository-Specific Pitfalls

- The standard collector filters for success. Its failed planning attempts are
  not matched, recorded failure data.
- The standard recording pass asserts success, so intentional failures require
  a separate driver.
- `seed.txt` indexes successful source episodes, not arbitrary failed variants.
  Give variants their own manifest and identifiers.
- NumPy and Torch are seeded in `Base_Task`; Python's `random` seed is commented
  out. Avoid the standard `random` module in controlled intervention logic, or
  seed it explicitly.
- The MP4 converter uses only `head_camera`. Use HDF5 or extend video output for
  wrist-camera comparisons.
- `joint_action` in recorded observations is the current joint state. Preserve
  the actual commanded action separately if the world model conditions on
  commands rather than observed qpos.
- Do not change nuisance randomization and failure mechanism in the same matched
  comparison.
- Do not use only a final binary label. Save continuous errors and contact,
  visibility, and stability state.
- Do not define a failure merely by forcing a label. The physical simulator
  state must violate the original task success condition.
- Do not truncate the simulator trajectory when testing temporal truncation
  without retaining the full counterfactual outcome. The model input may be
  truncated, but the ground-truth episode must establish that the action
  sequence does not achieve verified success.

## Minimal File Plan

The smallest maintainable implementation is:

```text
task_config/controlled_failure_clean.yml
envs/<task_name>_controlled.py
script/collect_controlled_failures.py
policy/Your_World_Model/infer_rollout.py
data_controlled_failures/<task>/<config>/...
```

Read and reuse infrastructure from:

```text
NOTE.md
script/collect_data.py
envs/_base_task.py
envs/utils/pkl2hdf5.py
script/eval_policy.py
policy/Your_Policy/deploy_policy.py
```

The experimental principle is:

> Do not merely collect failed trajectories. Construct attributable failed
> trajectories by changing one mechanism variable per matched group.

In Chinese:

> 我们不是要“收集失败轨迹”，而是要“构造可归因的失败轨迹”。每组失败轨迹只改变一个机制变量，比如 grasp slip、near-miss、wrong object、occlusion 或 temporal truncation。这样才能判断 world model 把失败预测成成功，到底是因为视觉捷径、时序外推、物体绑定错误，还是 evaluator 本身的判别边界问题。
