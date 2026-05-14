import os
import mujoco
import mujoco.viewer

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "op3_model",
    "src",
    "model",
    "scene.xml"
)

MODEL_PATH = os.path.abspath(MODEL_PATH)
print("Usando XML em:", MODEL_PATH)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()