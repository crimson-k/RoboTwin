from ._base_task import Base_Task
from .utils import *
import sapien
import math


class adjust_bottle_controlled(Base_Task):
    PHASE_IDS = {
        "setup": 0,
        "after_grasp": 1,
        "after_lift": 2,
        "after_place": 3,
        "verification": 4,
    }

    intervention_types = ["none", "unstable_grasp" , "trajectory_perturbation", "move_waypoint", "grasp_pose_perturbation"]

    def setup_demo(self, **kwags):
        self.configure_intervention(kwags.get("intervention", {"type": "none"}))
        self.control_step = 0
        self.intervention_active = False
        super()._init_task_env_(**kwags)

    def configure_intervention(self, spec):
        spec = spec or {"type": "none"}
        self.intervention = {
            "type": spec.get("type", "none"),
            "phase": spec.get("phase"),
            "parameters": dict(spec.get("parameters", {})),
            "verification_steps": int(spec.get("verification_steps", 20)),
        }
        self.current_phase = "setup"
        self.intervention_applied = False
        self.intervention_active = False

    def _move_with_online_planning(self, actions, keep_online_planning=False):
        """Plan an intervention while replay is using saved joint paths."""
        previous_need_plan = self.need_plan
        previous_plan_success = self.plan_success
        left_path_len = len(self.left_joint_path)
        right_path_len = len(self.right_joint_path)
        self.need_plan = True
        try:
            move_succeeded = self.move(actions)
            intervention_plan_succeeded = bool(
                move_succeeded and self.plan_success
            )
        finally:
            if keep_online_planning:
                self.need_plan = True
            else:
                del self.left_joint_path[left_path_len:]
                del self.right_joint_path[right_path_len:]
                self.need_plan = previous_need_plan
                self.plan_success = previous_plan_success
        return intervention_plan_succeeded


    def choose_grasp_pose(
        self,
        actor: Actor,
        arm_tag: ArmTag,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id: list | float = None,
    ):
        res_pre_pose, res_pose = super().choose_grasp_pose(actor, arm_tag, pre_dis, target_dis, contact_point_id)
        if self.intervention["type"] == "grasp_pose_perturbation":
            grasp_displacement_dim = int(self.intervention["parameters"].get("grasp_displacement_dim", 0))
            if grasp_displacement_dim not in [0, 1, 2]:
                raise ValueError("grasp_displacement_dim must be 0, 1, or 2")
            grasp_displacement = float(self.intervention["parameters"].get("grasp_displacement", 0.0))
            res_pre_pose[grasp_displacement_dim] += grasp_displacement
            res_pose[grasp_displacement_dim] += grasp_displacement
            return res_pre_pose, res_pose
        else:
            return res_pre_pose, res_pose    

    def grasp_actor(self, actor: Actor, arm_tag: ArmTag, pre_grasp_dis=0.1, target_dis=0, contact_point_id: list | float = None):
        gripper_perturb = self.intervention["parameters"].get("grasp_gripper_opening")
        if self.intervention["type"] == "grasp_pose_perturbation" and gripper_perturb is not None:
            Action(arm_tag, "close", target_gripper_pos=gripper_perturb)
        return super().grasp_actor(actor, arm_tag, pre_grasp_dis, target_dis, contact_point_id=contact_point_id)

    def maybe_intervene(self, phase, arm_tag):
        self.current_phase = phase
        if self.intervention["type"] == "none":
            return
        if self.intervention_applied or self.intervention["phase"] != phase:
            return
        if self.intervention["type"] not in self.intervention_types:
            raise ValueError(
                f"Unsupported intervention type: {self.intervention['type']}"
            )

        parameters = self.intervention["parameters"]
        gripper_position = float(parameters.get("gripper_position", 0.35))
        hold_steps = int(parameters.get("hold_steps", 20))
        hold_at_pose = bool(parameters.get("hold_at_pose", False))
        x_perturb = float(parameters.get("x_perturb", 0.05))
        y_perturb = float(parameters.get("y_perturb", 0.05))
        z_perturb = float(parameters.get("z_perturb", 0.05))
        target_pose_left = list(parameters.get("target_pose_left"), [0.0, 0.0, 0.0])
        target_pose_right = list(parameters.get("target_pose_right"), [0.0, 0.0, 0.0])
        quaternion = list(parameters.get("target_pose_right"), None)
        
        if not 0.0 <= gripper_position <= 1.0:
            raise ValueError("gripper_position must be in [0, 1]")
        if hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")

        self.intervention_applied = True
        self.intervention_active = True
        if self.intervention["type"] == "unstable_grasp":
            self.move(self.open_gripper(arm_tag, pos=gripper_position))
        elif self.intervention["type"] == "trajectory_perturbation":
            perturbation_succeeded = self._move_with_online_planning(
                self.move_by_displacement(
                    arm_tag,
                    move_axis="world",
                    x=x_perturb,
                    y=y_perturb,
                    z=z_perturb,
                ),
                keep_online_planning = self.need_plan
            )
            if not perturbation_succeeded:
                self.intervention_active = False
                raise RuntimeError(
                    "trajectory_perturbation planning failed; refusing to "
                    "record an episode without the requested perturbation"
                )
            
        elif self.intervention["type"] == "move_waypoint":
            target_pose = np.asarray(
                self.get_arm_pose(arm_tag=arm_tag),
                dtype=np.float64,
            )
            if arm_tag == 'left':
                target_pose[:3] = target_pose_left
            else:
                target_pose[:3] = target_pose_right

            pose_norm = np.linalg.norm(target_pose[3:])
            if pose_norm == 0:
                raise ValueError("move_waypoint target quaternion must be non-zero")
            target_pose[3:] = target_pose[3:] / pose_norm

            move_succeeded = self._move_with_online_planning(
                self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose),
                keep_online_planning=self.need_plan,
            )
            if not move_succeeded:
                self.intervention_active = False
                raise RuntimeError(
                    "move_waypoint planning failed; refusing to record an "
                    "episode without the requested intervention"
                )

        self._advance_simulation(hold_steps)
        self.intervention_active = False

    def _advance_simulation(self, steps):
        for step in range(steps):
            self.scene.step()
            self.control_step += 1
            if self.render_freq and step % self.render_freq == 0:
                self._update_render()
                self.viewer.render()
            if self.save_freq is not None and step % self.save_freq == 0:
                self._take_picture()

    def run_verification_horizon(self):
        self.current_phase = "verification"
        self._advance_simulation(self.intervention["verification_steps"])

    def take_dense_action(self, control_seq, save_freq=-1):
        control_lengths = []
        for key in ("left_arm", "right_arm"):
            action = control_seq[key]
            if action is not None:
                control_lengths.append(action["position"].shape[0])
        for key in ("left_gripper", "right_gripper"):
            action = control_seq[key]
            if action is not None:
                control_lengths.append(action["num_step"])

        result = super().take_dense_action(control_seq, save_freq=save_freq)
        self.control_step += max(control_lengths, default=0)
        return result

    def load_actors(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]

        self.model_id = np.random.choice([13, 16])

        self.bottle = rand_create_actor(
            self,
            xlim=xlims[self.qpose_tag],
            ylim=[-0.13, -0.08],
            zlim=[0.752],
            rotate_rand=True,
            qpos=qposes[self.qpose_tag],
            modelname="001_bottle",
            convex=True,
            rotate_lim=(0, 0, 0.4),
            model_id=self.model_id,
        )
        self.delay(4)
        self.add_prohibit_area(self.bottle, padding=0.15)
        self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

    def play_once(self):
        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        target_pose = (
            self.right_target_pose
            if self.qpose_tag == 1
            else self.left_target_pose
        )

        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
            )
        )

        self.maybe_intervene("after_grasp", arm_tag)

        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.1,
                move_axis="arm",
            )
        )
        self.maybe_intervene("after_lift", arm_tag)

        self.move(
            self.place_actor(
                self.bottle,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                is_open=False,
            )
        )
        self.maybe_intervene("after_place", arm_tag)
        self.run_verification_horizon()

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def _bottle_velocity(self):
        for component in self.bottle.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return (
                    np.asarray(component.get_linear_velocity(), dtype=np.float64),
                    np.asarray(component.get_angular_velocity(), dtype=np.float64),
                )
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def get_failure_metrics(self):
        bottle_position = np.asarray(self.bottle.get_pose().p, dtype=np.float64)
        functional_point = np.asarray(
            self.bottle.get_functional_point(0)[:3],
            dtype=np.float64,
        )
        linear_velocity, angular_velocity = self._bottle_velocity()
        correct_side = (
            functional_point[0] < -0.15
            if self.qpose_tag == 0
            else functional_point[0] > 0.15
        )

        return {
            "bottle_position": bottle_position.tolist(),
            "functional_point_position": functional_point.tolist(),
            "correct_side": bool(correct_side),
            "above_height_threshold": bool(functional_point[2] > 0.9),
            "gripper_contact": bool(
                self.get_gripper_actor_contact_position(self.bottle.get_name())
            ),
            "linear_speed_mps": float(np.linalg.norm(linear_velocity)),
            "angular_speed_radps": float(np.linalg.norm(angular_velocity)),
        }

    def get_obs(self):
        obs = super().get_obs()
        if hasattr(self, "bottle"):
            bottle_pose = self.bottle.get_pose()
            linear_velocity, angular_velocity = self._bottle_velocity()
            obs["sim_state"] = {
                "bottle_pose": np.asarray(
                    bottle_pose.p.tolist() + bottle_pose.q.tolist(),
                    dtype=np.float64,
                ),
                "bottle_velocity": np.concatenate(
                    (linear_velocity, angular_velocity)
                ),
                "intervention_active": np.asarray(
                    self.intervention_active,
                    dtype=np.bool_,
                ),
                "phase_id": np.asarray(
                    self.PHASE_IDS[self.current_phase],
                    dtype=np.int64,
                ),
                "control_step": np.asarray(
                    self.control_step,
                    dtype=np.int64,
                ),
            }
        return obs

    def check_success(self):
        target_hight = 0.9
        bottle_pose = self.bottle.get_functional_point(0)
        return (
            (
                self.qpose_tag == 0
                and bottle_pose[0] < -0.15
                or self.qpose_tag == 1
                and bottle_pose[0] > 0.15
            )
            and bottle_pose[2] > target_hight
        )
