from ._base_task import Base_Task
from .intervention_utils import InterventionMixin
from .utils import *


class adjust_bottle_controlled(InterventionMixin, Base_Task):

    def configure_intervention(self, spec):
        return super().configure_intervention(spec)
    
    def setup_demo(self, **kwags):
        self.configure_intervention(kwags)
        self.control_step = 0
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0.707, 0.0, 0.0, -0.707], [0.707, 0.0, 0.0, 0.707]]
        xlims = [[-0.12, -0.08], [0.08, 0.12]]

        self.model_id = np.random.choice([13, 16])

        self.target_actor = rand_create_actor(
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
        self.add_prohibit_area(self.target_actor, padding=0.15)
        self.left_target_pose = [-0.25, -0.12, 0.95, 0, 1, 0, 0]
        self.right_target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

    def maybe_intervene(self, phase, arm_tag):
        return super().maybe_intervene(phase, arm_tag)
    
    def grasp_actor(self, actor, arm_tag, pre_grasp_dis=0.1, target_dis=0, contact_point_id = None):
        return super().grasp_actor(actor, arm_tag, pre_grasp_dis, target_dis, contact_point_id)
    
    def play_once(self):
        arm_tag = ArmTag("right" if self.qpose_tag == 1 else "left")
        target_pose = (
            self.right_target_pose
            if self.qpose_tag == 1
            else self.left_target_pose
        )

        self.maybe_intervene("before_grasp", arm_tag)

        self.move(
            self.grasp_actor(
                self.target_actor,
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
                self.target_actor,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.0,
                is_open=False,
            )
        )
        self.maybe_intervene("after_place", arm_tag)

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.model_id}",
            "{a}": str(arm_tag),
        }
        return self.info
    
    def check_success(self):
        target_hight = 0.9
        bottle_pose = self.target_actor.get_functional_point(0)
        return (
            (
                self.qpose_tag == 0
                and bottle_pose[0] < -0.15
                or self.qpose_tag == 1
                and bottle_pose[0] > 0.15
            )
            and bottle_pose[2] > target_hight
        )
    
