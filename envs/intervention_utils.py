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
        self.num_of_interventions = spec.get("num_of_interventions", 0)
        self.intervention_list = {}
        for i in range(self.num_of_interventions):
            intervention_data = spec.get(f"intervention {i}", {"type": None})
            intervention_data["intervention_applied"] = False
            intervention_data["intervention_active"] = False
            self.intervention_list[f"intervention {i}"] = intervention_data
        self.current_phase = "setup"
        self.current_intervention_id = 0
        self.intervention = self.intervention_list.get(f"intervention {self.current_intervention_id}")

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
            self.intervention_active = True
            grasp_displacement_dim = int(self.intervention["parameters"].get("grasp_displacement_dim", 0))
            if grasp_displacement_dim not in [0, 1, 2]:
                raise ValueError("grasp_displacement_dim must be 0, 1, or 2")
            grasp_displacement = float(self.intervention["parameters"].get("grasp_displacement", 0.0))
            res_pre_pose[grasp_displacement_dim] += grasp_displacement
            res_pose[grasp_displacement_dim] += grasp_displacement
            
            self.intervention_applied = True
            self.intervention_active = False
            if self.current_intervention_id == (self.num_of_interventions - 1):
                return
            self.current_intervention_id += 1
            self.intervention = self.intervention_list.get(f"intervention {self.current_intervention_id}")
            return res_pre_pose, res_pose
        else:
            return res_pre_pose, res_pose

    def grasp_actor(self, actor: Actor, arm_tag: ArmTag, pre_grasp_dis=0.1, target_dis=0, contact_point_id: list | float = None):
        gripper_perturb = self.intervention["parameters"].get("grasp_gripper_opening")
        if self.intervention["type"] == "grasp_pose_perturbation" and gripper_perturb is not None:
            self.intervention_active = True
            Action(arm_tag, "close", target_gripper_pos=gripper_perturb)
        return super().grasp_actor(actor, arm_tag, pre_grasp_dis, target_dis, contact_point_id=contact_point_id)
    
    def maybe_intervene(self, phase, arm_tag):
        self.current_phase = phase
        if self.intervention["type"] == "none" or "grasp_pose_perturbation":
            return
        if self.intervention["type"] not in self.intervention_types:
            raise ValueError(
                f"Unsupported intervention type: {self.intervention['type']}"
            )
        if self.intervention.get("intervention_applied"):
            return
        if self.intervention["phase"] != self.current_phase:
            return
        
        parameters = self.intervention["parameters"]
        hold_steps = int(parameters.get("hold_steps", 20))
        if hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        self.intervention_active = True
        
        if self.intervention["type"] == "unstable_grasp":
            gripper_position = parameters.get("gripper_position", 0.0)
            if not 0.0 <= gripper_position <= 1.0:
                raise ValueError("gripper_position must be in [0, 1]")
            self.move(self.open_gripper(arm_tag, pos=gripper_position))

        elif self.intervention["type"] == "trajectory_perturbation":
            x_perturb = parameters.get("x_perturb", 0.0)
            y_perturb = parameters.get("y_perturb", 0.0)
            z_perturb = parameters.get("z_perturb", 0.0)
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
            target_pose_left = parameters.get("target_pose_left")
            target_pose_right = parameters.get("target_pose_right")
            gripper_angle = parameters.get("gripper_angle")

            if arm_tag == 'left':
                if target_pose_left is None:
                    target_pose_left == target_pose[:3]
                target_pose[:3] = target_pose_left
            else:
                if target_pose_right is None:
                    target_pose_right == target_pose[:3]
                target_pose[:3] = target_pose_right

            if not gripper_angle:
                pose_norm = np.linalg.norm(target_pose[3:])
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

        self._advance_simulation(hold_steps)

        self.intervention_applied = True
        self.intervention_active = False
        if self.current_intervention_id == (self.num_of_interventions - 1):
            return
        self.current_intervention_id += 1
        self.intervention = self.intervention_list.get(f"intervention {self.current_intervention_id}")
    
    def _advance_simulation(self, steps):
        for step in range(steps):
            self.scene.step()
            self.control_step += 1
            if self.render_freq and step % self.render_freq == 0:
                self._update_render()
                self.viewer.render()
            if self.save_freq is not None and step % self.save_freq == 0:
                self._take_picture()

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
