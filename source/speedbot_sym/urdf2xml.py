import mujoco

model = mujoco.MjModel.from_xml_path('speedbot.urdf')
mujoco.mj_saveLastXML('speedbot.xml', model)

'''
<mujoco>
    <compiler
        meshdir="./meshes"
        balanceinertia="true"
        discardvisual="false"
    />
</mujoco>
'''