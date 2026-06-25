from .utils import *
import sapien
import math

class InterventionMixin():

    intervention_types = [
        "unstable_grasp" ,
        "trajectory_perturbation", 
        "move_waypoint", 
        "grasp_pose_perturbation"
        ]
    
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
        self.applied_waypoints = set()
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
        if self.intervention["type"] not in self.intervention_types:
            raise ValueError(
                f"Unsupported intervention type: {self.intervention['type']}"
            )

        parameters = self.intervention["parameters"]
        phase_spec = parameters.get("phase", self.intervention.get("phase"))
        phase_list = phase_spec if isinstance(phase_spec, list) else [phase_spec]
        if phase not in phase_list:
            return
        if self.intervention["type"] != "move_waypoint" and self.intervention_applied:
            return

        gripper_position = float(parameters.get("gripper_position", 0.35))
        hold_steps = int(parameters.get("hold_steps", 20))
        hold_at_pose = bool(parameters.get("hold_at_pose", False))
        x_perturb = float(parameters.get("x_perturb", 0.05))
        y_perturb = float(parameters.get("y_perturb", 0.05))
        z_perturb = float(parameters.get("z_perturb", 0.05))
        if self.intervention["type"] == "move_waypoint":
            num_of_waypoints = int(parameters.get("num_of_waypoints", 0))
        
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
            applied_this_phase = False
            for i in range(num_of_waypoints):
                waypoint_key = f"waypoint {i}"
                waypoint = parameters.get(waypoint_key)
                if waypoint is None:
                    raise ValueError(f"Missing move_waypoint config: {waypoint_key}")
                waypoint = dict(waypoint)
                intervention_phase = waypoint.get(
                    "phase",
                    phase_list[i] if i < len(phase_list) else phase_list[-1],
                )
                if intervention_phase != phase:
                    continue
                if i in self.applied_waypoints:
                    continue
                target_pose = np.asarray(
                    self.get_arm_pose(arm_tag=arm_tag),
                    dtype=np.float64,
                )
                target_pose_left = waypoint.get("target_pose_left")
                target_pose_right = waypoint.get("target_pose_right")
                gripper_angle = waypoint.get("gripper_angle")

                if arm_tag == 'left':
                    if target_pose_left is None:
                        raise ValueError(f"{waypoint_key}.target_pose_left is required")
                    target_pose[:3] = target_pose_left
                else:
                    if target_pose_right is None:
                        raise ValueError(f"{waypoint_key}.target_pose_right is required")
                    target_pose[:3] = target_pose_right

                if not gripper_angle:
                    pose_norm = np.linalg.norm(target_pose[3:])
                    if pose_norm == 0:
                        raise ValueError("move_waypoint target gripper_angle must be non-zero")
                    target_pose[3:] = target_pose[3:] / pose_norm
                else:
                    target_pose[3:] = transforms.euler_expr_to_quat(gripper_angle)

                move_succeeded = self._move_with_online_planning(
                    self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose),
                    keep_online_planning=self.need_plan,
                )
                if not move_succeeded:
                    self.intervention_active = False
                    raise RuntimeError(
                        "move_waypoint planning failed; refusing to record an episode without the requested intervention"
                    )
                self.applied_waypoints.add(i)
                applied_this_phase = True
            if not applied_this_phase:
                self.intervention_active = False
                return

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
    
    def _actor_velocity(self):
        for component in self.target_actor.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return (
                    np.asarray(component.get_linear_velocity(), dtype=np.float64),
                    np.asarray(component.get_angular_velocity(), dtype=np.float64),
                )
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def get_obs(self):
        obs = super().get_obs()
        if hasattr(self, "target_actor"):
            actor_pose = self.target_actor.get_pose()
            linear_velocity, angular_velocity = self._actor_velocity()
            obs["sim_state"] = {
                "actor_pose": np.asarray(
                    actor_pose.p.tolist() + actor_pose.q.tolist(),
                    dtype=np.float64,
                ),
                "actor_velocity": np.concatenate(
                    (linear_velocity, angular_velocity)
                ),
                "intervention_active": np.asarray(
                    self.intervention_active,
                    dtype=np.bool_,
                ),
                "control_step": np.asarray(
                    self.control_step,
                    dtype=np.int64,
                ),
            }
        return obs