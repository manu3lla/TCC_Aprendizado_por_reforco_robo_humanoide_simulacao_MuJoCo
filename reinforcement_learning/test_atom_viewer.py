import mujoco
import mujoco.viewer

from atom_paths import ATOM_SCENE_XML

print("Usando XML em:", ATOM_SCENE_XML)

model = mujoco.MjModel.from_xml_path(str(ATOM_SCENE_XML))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
