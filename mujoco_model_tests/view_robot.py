import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("robot_scene.xml")
# model = mujoco.MjModel.from_xml_path("robot.urdf")
data = mujoco.MjData(model)


print("Number of actuators:", model.nu)


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
    
        data.ctrl[1] = 2.1   # elbow_right

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.01)