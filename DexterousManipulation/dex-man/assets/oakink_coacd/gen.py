import os

# URDF template with placeholder for filename
URDF_TEMPLATE = '''<?xml version="1.0"?>
<robot name="design">
  <material name="obj_color">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
'''

def generate_urdf_for_obj(obj_path):
    dir_name = os.path.dirname(obj_path)
    base_name = os.path.basename(obj_path)
    name_without_ext = os.path.splitext(base_name)[0]
    urdf_path = os.path.join(dir_name, f"{name_without_ext}.urdf")

    with open(urdf_path, "w") as f:
        f.write(URDF_TEMPLATE.format(filename=base_name))
    print(f"Generated: {urdf_path}")

def main():
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(".obj"):
                obj_full_path = os.path.join(root, file)
                generate_urdf_for_obj(obj_full_path)

if __name__ == "__main__":
    main()
