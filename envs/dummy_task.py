from ._base_task import Base_Task
from .intervention_utils import InterventionMixin
from .utils import *

class dummy_task(InterventionMixin, Base_Task):
    
    def configure_intervention(self, spec):
        return super().configure_intervention(spec)
    
    def setup_demo(self, **kwags):
        self.configure_intervention(kwags)
        self.control_step = 0
        super()._init_task_env_(**kwags)

    def load_actors(self):
        if self.actor_source is None:
            return

        import importlib
        module = importlib.import_module(f"envs.{self.actor_source}")
        source_cls = getattr(module, self.actor_source)
        source_cls.load_actors(self)

    def maybe_intervene(self, phase, arm_tag):
        return super().maybe_intervene(phase, arm_tag)
    
    def play_once(self):
        arm_tag = ArmTag("left")
        
        self.maybe_intervene("phase", arm_tag)

        if self.actor_source == "adjust_bottle" and hasattr(self, "model_id"):
            self.info["info"] = {
                "{A}": f"001_bottle/base{self.model_id}",
                "{a}": str(arm_tag),
            }

        return self.info
    
    def check_success(self):
        return True
