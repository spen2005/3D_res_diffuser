import os
import trimesh

def convert_ply_to_obj(ply_path, obj_path):
    mesh = trimesh.load(ply_path, force='mesh')
    mesh.export(obj_path)
    print(f"Converted {ply_path} -> {obj_path}")

def write_urdf(folder, obj_filename):
    urdf_path = os.path.join(folder, f"{os.path.basename(folder)}.urdf")
    urdf_content = f"""<?xml version="1.0"?>
<robot name="design">
  <material name="obj_color">
      <color rgba="0.2 0.3 0.5 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)
    print(f"Written URDF to {urdf_path}")

def main():
    base_dir = os.getcwd()

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.ply'):
                ply_path = os.path.join(root, file)
                obj_filename = os.path.splitext(file)[0] + '.obj'
                obj_path = os.path.join(root, obj_filename)

                convert_ply_to_obj(ply_path, obj_path)
                write_urdf(root, obj_filename)

            elif file.endswith('.obj'):
                # If .obj already exists but no urdf, just write the urdf
                obj_path = os.path.join(root, file)
                urdf_path = os.path.join(root, f"{os.path.basename(root)}.urdf")
                if not os.path.exists(urdf_path):
                    write_urdf(root, file)

if __name__ == "__main__":
    main()
