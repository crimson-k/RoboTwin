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
            self.intervention_list[f"intervention {i}"] = intervention_data
        self.current_phase = "setup"
        self.current_intervention_id = 0
        self.intervention = self.intervention_list.get(f"intervention {self.current_intervention_id}")

    def _summarize_plan_result(self, result):
        if not isinstance(result, dict):
            return f"type={type(result).__name__}"
        parts = [f"status={result.get('status')}"]
        if "position" in result:
            position = result["position"]
            shape = getattr(position, "shape", None)
            parts.append(f"position_shape={shape}")
            if shape is not None and len(shape) > 0 and shape[0] > 0:
                parts.append(f"first_qpos={np.round(position[0], 4).tolist()}")
                parts.append(f"last_qpos={np.round(position[-1], 4).tolist()}")
        extra_keys = sorted(k for k in result.keys() if k not in ("status", "position", "velocity"))
        if extra_keys:
            parts.append(f"extra_keys={extra_keys}")
        return ", ".join(parts)

    def _format_pose_debug(self, pose):
        if isinstance(pose, sapien.Pose):
            return np.round(np.asarray(pose.p.tolist() + pose.q.tolist()), 4).tolist()
        return np.round(np.asarray(pose, dtype=np.float64), 4).tolist()

    def _move_with_online_planning(self, actions, keep_online_planning=False):
        """Plan an intervention while replay is using saved joint paths."""
        previous_need_plan = self.need_plan
        previous_plan_success = self.plan_success
        left_path_len = len(self.left_joint_path)
        right_path_len = len(self.right_joint_path)
        self.need_plan = True
        self.last_online_plan_debug = None
        move_succeeded = None
        intervention_plan_succeeded = False
        try:
            move_succeeded = self.move(actions)
            intervention_plan_succeeded = bool(
                move_succeeded and self.plan_success
            )
        finally:
            new_left_results = self.left_joint_path[left_path_len:]
            new_right_results = self.right_joint_path[right_path_len:]
            self.last_online_plan_debug = {
                "move_succeeded": move_succeeded,
                "plan_success_after_move": self.plan_success,
                "new_left_results": [self._summarize_plan_result(result) for result in new_left_results],
                "new_right_results": [self._summarize_plan_result(result) for result in new_right_results],
            }
            if keep_online_planning:
                self.need_plan = True
            else:
                del self.left_joint_path[left_path_len:]
                del self.right_joint_path[right_path_len:]
                self.need_plan = previous_need_plan
                self.plan_success = previous_plan_success
        return intervention_plan_succeeded

    def _print_move_waypoint_debug(
        self,
        arm_tag,
        current_pose,
        target_pose,
        transformed_target_pose,
        parameters,
    ):
        arm = str(arm_tag)
        qpos = self.robot.left_entity.get_qpos() if arm == "left" else self.robot.right_entity.get_qpos()
        print("  [move_waypoint debug]")
        print(f"    intervention_id: {self.current_intervention_id}")
        print(f"    phase: {self.current_phase}")
        print(f"    arm: {arm}")
        print(f"    parameters: {parameters}")
        print(f"    current_gripper_pose: {self._format_pose_debug(current_pose)}")
        print(f"    requested_gripper_pose: {self._format_pose_debug(target_pose)}")
        print(f"    planner_endlink_goal: {self._format_pose_debug(transformed_target_pose)}")
        table_height = 0.74 + getattr(self, "table_z_bias", 0)
        print(f"    table_height: {table_height:.4f}")
        if target_pose[2] <= table_height + 0.02:
            print("    warning: requested gripper z is at or near table height")
        print(f"    current_qpos: {np.round(np.asarray(qpos, dtype=np.float64), 4).tolist()}")
        plan_debug = getattr(self, "last_online_plan_debug", None)
        if not plan_debug:
            print("    planner_results: no planner result was recorded")
            return
        print(f"    move_returned: {plan_debug['move_succeeded']}")
        print(f"    plan_success_after_move: {plan_debug['plan_success_after_move']}")
        for result in plan_debug["new_left_results"]:
            print(f"    left_result: {result}")
        for result in plan_debug["new_right_results"]:
            print(f"    right_result: {result}")
    
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
        actions = super().grasp_actor(actor, arm_tag, pre_grasp_dis, target_dis, gripper_pos = gripper_perturb, contact_point_id=contact_point_id)
        if self.current_intervention_id != (self.num_of_interventions - 1):
            self.current_intervention_id += 1
            self.intervention = self.intervention_list.get(f"intervention {self.current_intervention_id}")
        return actions

    def maybe_intervene(self, phase, arm_tag):
        self.current_phase = phase
        if self.intervention["type"] == "none" or self.intervention["type"] == "grasp_pose_perturbation":
            return
        if self.intervention["type"] not in self.intervention_types:
            raise ValueError(
                f"Unsupported intervention type: {self.intervention['type']}"
            )
        if self.intervention["phase"] != self.current_phase:
            return
        
        parameters = self.intervention["parameters"]
        hold_steps = int(parameters.get("hold_steps", 20))
        if hold_steps < 0:
            raise ValueError("hold_steps must be non-negative")
        
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
                raise RuntimeError(
                    "trajectory_perturbation planning failed"
                )
            
        elif self.intervention["type"] == "move_waypoint":
            current_pose = np.asarray(
                self.get_arm_pose(arm_tag=arm_tag),
                dtype=np.float64,
            )
            target_pose = current_pose.copy()
            target_pose_left = parameters.get("target_pose_left")
            target_pose_right = parameters.get("target_pose_right")
            gripper_angle = parameters.get("gripper_angle")

            if arm_tag == 'left':
                if target_pose_left is None:
                    target_pose_left = target_pose[:3]
                target_pose[:3] = target_pose_left
            else:
                if target_pose_right is None:
                    target_pose_right = target_pose[:3]
                target_pose[:3] = target_pose_right

            if not gripper_angle:
                pose_norm = np.linalg.norm(target_pose[3:])
                target_pose[3:] = target_pose[3:] / pose_norm
            else:
                target_pose[3:] = transforms.euler_expr_to_quat(gripper_angle)

            transformed_target_pose = self.robot._trans_from_gripper_to_endlink(
                target_pose,
                arm_tag=str(arm_tag),
            )
            move_succeeded = self._move_with_online_planning(
                self.move_to_pose(arm_tag=arm_tag, target_pose=target_pose),
                keep_online_planning=self.need_plan,
            )
            if not move_succeeded:
                self._print_move_waypoint_debug(
                    arm_tag,
                    current_pose,
                    target_pose,
                    transformed_target_pose,
                    parameters,
                )
                raise RuntimeError(
                    "move_waypoint planning failed"
                )

        self._advance_simulation(hold_steps)

        if self.current_intervention_id == (self.num_of_interventions - 1):
            return
        self.current_intervention_id += 1
        self.intervention = self.intervention_list.get(f"intervention {self.current_intervention_id}")
        if self.intervention.get("phase") == self.current_phase:
            self.maybe_intervene(phase, arm_tag)
    
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
                "control_step": np.asarray(
                    self.control_step,
                    dtype=np.int64,
                ),
            }
        return obs
