import struct
import numpy as np

def parse_stl(filename):
    with open(filename, 'rb') as f:
        # Binary STL header
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        print(f"Num triangles: {num_triangles}")
        
        vertices = []
        for _ in range(num_triangles):
            # Normal (3 floats)
            f.read(12)
            # 3 Vertices (3 floats each)
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            v3 = struct.unpack('<fff', f.read(12))
            # Attribute byte count
            f.read(2)
            
            vertices.extend([v1, v2, v3])
            
        verts = np.array(vertices)
        v_min = verts.min(axis=0)
        v_max = verts.max(axis=0)
        size = v_max - v_min
        print(f"Min: {v_min}")
        print(f"Max: {v_max}")
        print(f"Size: {size}")

if __name__ == "__main__":
    import sys
    parse_stl("rocket_lander/envs/assets/meshes/rocket_main_body.stl")
    print("\n--- THRUSTER ---")
    parse_stl("rocket_lander/envs/assets/meshes/rocket_thruster_main.stl")
    print("\n--- FOOT ---")
    parse_stl("rocket_lander/envs/assets/meshes/rocket_foot.stl")
    print("\n--- RCS ROLL PITCH ---")
    parse_stl("rocket_lander/envs/assets/meshes/rocket_rcs_roll_pitch.stl")
    print("\n--- RCS YAW ---")
    parse_stl("rocket_lander/envs/assets/meshes/rocket_rcs_yaw.stl")
    print("\n--- HOLDER ---")
    parse_stl("rocket_lander/envs/assets/meshes/rocket_thruster_holder.stl")
