import json
import os
import math

# Load geometry data
DATA_FILE = "geometry_data.json"
URDF_FILE = "rocket_lander.urdf"

def quaternion_to_rpy(q):
    """
    Convert quaternion (x, y, z, w) to roll-pitch-yaw Euler angles.
    """
    x, y, z, w = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

def generate_urdf():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    scale = 0.001 
    
    MESH_MAP = {
        "rocket_main_body": "meshes/rocket_main_body.stl",
        "rocket_thruster_main_engine": "meshes/rocket_thruster_main.stl",
        "rocket_thruster_holder": "meshes/rocket_thruster_holder.stl",
        "rocket_foot": "meshes/rocket_foot.stl",
        "rocket_rcs_yaw": "meshes/rocket_rcs_yaw.stl",
        "rocket_rcs_roll_pitch": "meshes/rocket_rcs_roll_pitch.stl"
    }

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append('<robot name="rocket_lander">')
    
    # Materials
    xml.append('  <material name="grey"><color rgba="0.7 0.7 0.7 1"/></material>')
    xml.append('  <material name="dark"><color rgba="0.3 0.3 0.3 1"/></material>')

    # 1. BASE LINK (Main Body)
    xml.append('  <link name="base_link">')
    xml.append('    <visual>')
    xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
    xml.append('      <geometry>')
    xml.append(f'        <mesh filename="{MESH_MAP["rocket_main_body"]}" scale="0.001 0.001 0.001"/>')
    xml.append('      </geometry>')
    xml.append('      <material name="grey"/>')
    xml.append('    </visual>')
    xml.append('    <collision>')
    xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
    xml.append('      <geometry>')
    xml.append('        <cylinder radius="0.45" length="5.0"/>') 
    xml.append('      </geometry>')
    xml.append('    </collision>')
    xml.append('    <inertial>')
    xml.append('      <mass value="100.0"/>')
    xml.append('      <inertia ixx="10.0" ixy="0.0" ixz="0.0" iyy="10.0" iyz="0.0" izz="2.0"/>')
    xml.append('    </inertial>')
    xml.append('  </link>')

    # 2. Iterate Components
    for key, props in data.items():
        if key == "rocket_main_body": continue
            
        link_name = key
        pos = [p * scale for p in props['position']]
        quat = props['rotation_quaternion']
        r, p, y = quaternion_to_rpy(quat)
        
        # Mapping
        mesh_file = None
        for prefix, filename in MESH_MAP.items():
            if link_name.startswith(prefix):
                mesh_file = filename
                break
        
        if not mesh_file:
            print(f"Warning: No mesh mapping for {link_name}")
            continue

        # --- SPECIAL LOGIC: MAIN ENGINE GIMBAL ---
        if "thruster_main" in key:
            gimbal_link = f"{link_name}_gimbal_intermed"
            
            # Pitch Joint (Base -> Intermed) - Y Axis
            xml.append(f'  <link name="{gimbal_link}">')
            xml.append('    <inertial><mass value="0.1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>')
            xml.append('  </link>')
            xml.append(f'  <joint name="joint_{link_name}_pitch" type="revolute">')
            xml.append('    <parent link="base_link"/>')
            xml.append(f'    <child link="{gimbal_link}"/>')
            xml.append(f'    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{r} {p} {y}"/>')
            xml.append('    <axis xyz="0 1 0"/>') 
            xml.append('    <limit lower="-0.35" upper="0.35" effort="1000" velocity="10.0"/>')
            xml.append('  </joint>')
            
            # Roll Joint (Intermed -> Engine) - X Axis (Renamed from Yaw as requested)
            xml.append(f'  <link name="{link_name}">')
            xml.append('    <visual>')
            xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            xml.append('      <geometry>')
            xml.append(f'        <mesh filename="{mesh_file}" scale="0.001 0.001 0.001"/>')
            xml.append('      </geometry>')
            xml.append('      <material name="dark"/>')
            xml.append('    </visual>')
            xml.append('    <collision>')
            xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
            xml.append('      <geometry>')
            xml.append(f'        <mesh filename="{mesh_file}" scale="0.001 0.001 0.001"/>')
            xml.append('      </geometry>')
            xml.append('    </collision>')
            xml.append('    <inertial>')
            xml.append('      <mass value="5.0"/>')
            xml.append('      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>')
            xml.append('    </inertial>')
            xml.append('  </link>')
            xml.append(f'  <joint name="joint_{link_name}_roll" type="revolute">')
            xml.append(f'    <parent link="{gimbal_link}"/>')
            xml.append(f'    <child link="{link_name}"/>')
            xml.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')
            xml.append('    <axis xyz="1 0 0"/>') 
            xml.append('    <limit lower="-0.35" upper="0.35" effort="1000" velocity="10.0"/>')
            xml.append('  </joint>')
            continue

        # --- SPECIAL LOGIC: LEGS ---
        joint_type = "fixed"
        axis = "0 0 1"
        lower_limit = 0
        upper_limit = 0
        
        if "foot" in key:
            joint_type = "revolute"
            # Use LOCAL Y Axis for rotation
            axis = "0 1 0" 
            # Revised Range: +/- 1.6 rad (approx pi/2)
            lower_limit = -1.6 
            upper_limit = 1.6

        # Standard Link
        xml.append(f'  <link name="{link_name}">')
        xml.append('    <visual>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append('      <geometry>')
        xml.append(f'        <mesh filename="{mesh_file}" scale="0.001 0.001 0.001"/>')
        xml.append('      </geometry>')
        xml.append('      <material name="dark"/>')
        xml.append('    </visual>')
        xml.append('    <collision>')
        xml.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        xml.append('      <geometry>')
        xml.append(f'        <mesh filename="{mesh_file}" scale="0.001 0.001 0.001"/>')
        xml.append('      </geometry>')
        xml.append('    </collision>')
        xml.append('    <inertial>')
        xml.append('      <mass value="5.0"/>')
        xml.append('      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>')
        xml.append('    </inertial>')
        xml.append('  </link>')

        xml.append(f'  <joint name="joint_{link_name}" type="{joint_type}">')
        xml.append('    <parent link="base_link"/>')
        xml.append(f'    <child link="{link_name}"/>')
        xml.append(f'    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{r} {p} {y}"/>')
        if joint_type == "revolute":
            xml.append(f'    <axis xyz="{axis}"/>')
            xml.append(f'    <limit lower="{lower_limit}" upper="{upper_limit}" effort="100" velocity="1.0"/>')
        xml.append('  </joint>')

    xml.append('</robot>')
    
    with open(URDF_FILE, 'w') as f:
        f.write('\n'.join(xml))
    print(f"URDF generated: {os.path.abspath(URDF_FILE)}")

if __name__ == "__main__":
    generate_urdf()
