from ._base_task import Base_Task
from .intervention_utils import InterventionMixin
from .utils import *
import ast
import importlib
import inspect
import textwrap


def _literal_int_list(node):
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    ids = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, int):
            return None
        ids.append(item.value)
    return ids


def _choice_literal_pool(node):
    if isinstance(node, ast.Subscript):
        node = node.value
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "choice" or not node.args:
        return None
    return _literal_int_list(node.args[0])


def get_actor_source_model_ids(actor_source):
    module = importlib.import_module(f"envs.{actor_source}")
    source_cls = getattr(module, actor_source)
    source = textwrap.dedent(inspect.getsource(source_cls.load_actors))
    tree = ast.parse(source)

    assigned_pools = {}
    model_ids = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pool = _choice_literal_pool(node.value)
            if pool is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    assigned_pools[target.attr] = pool
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg not in ("model_id", "modelid"):
                    continue
                value = keyword.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id == "self":
                    model_ids.extend(assigned_pools.get(value.attr, []))
                elif isinstance(value, ast.Constant) and isinstance(value.value, int):
                    model_ids.append(value.value)

    deduped = []
    for model_id in model_ids:
        if model_id not in deduped:
            deduped.append(model_id)
    return deduped


class dummy_task(InterventionMixin, Base_Task):
    
    def configure_intervention(self, spec):
        return super().configure_intervention(spec)
    
    def setup_demo(self, **kwags):
        self.actor_model_sweep = kwags.get("actor_model_sweep", False)
        self.actor_model_repeat = int(kwags.get("actor_model_repeat", 8))
        self.actor_model_sweep_index = kwags.get("intervention_variant_id", kwags.get("now_ep_num", 0))
        self.actor_model_ids = (
            get_actor_source_model_ids(kwags.get("actor_source"))
            if self.actor_model_sweep and kwags.get("actor_source") is not None
            else []
        )
        self.configure_intervention(kwags)
        self.control_step = 0
        super()._init_task_env_(**kwags)

    def load_actors(self):
        if self.actor_source is None:
            return

        module = importlib.import_module(f"envs.{self.actor_source}")
        source_cls = getattr(module, self.actor_source)

        if not self.actor_model_sweep or not self.actor_model_ids:
            source_cls.load_actors(self)
            return

        forced_model_id = self.actor_model_ids[
            (self.actor_model_sweep_index // self.actor_model_repeat) % len(self.actor_model_ids)
        ]
        old_choice = np.random.choice

        def choice_with_forced_model_id(a, *args, **kwargs):
            choices = np.asarray(a).tolist()
            if choices == self.actor_model_ids:
                size = kwargs.get("size", args[0] if args else None)
                if size is not None:
                    return np.full(size, forced_model_id)
                return forced_model_id
            return old_choice(a, *args, **kwargs)

        try:
            np.random.choice = choice_with_forced_model_id
            source_cls.load_actors(self)
        finally:
            np.random.choice = old_choice

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
