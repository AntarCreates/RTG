# Adding a DAE Mesh to Gazebo Sim

This guide explains how to take a DAE (Collada) mesh file and create a functional Gazebo Sim world around it. The process is specific to the newer Gazebo Sim (gz sim) which has a different workflow than classic Gazebo.

## Prerequisites

- A DAE mesh file
- Gazebo Sim installed on your system
- Basic familiarity with XML and SDF format

## Step 1: Prepare Your Directory Structure

Gazebo Sim looks for models in specific directories. The recommended location is:

```bash
# Create the standard Gazebo Sim model directory
mkdir -p ~/.gz/sim/models/my_mesh_model/meshes
```

## Step 2: Copy Your Mesh File

```bash
# Replace path/to/your/mesh.dae with the actual path to your DAE file
cp path/to/your/mesh.dae ~/.gz/sim/models/my_mesh_model/meshes/
```

## Step 3: Create the Model Configuration File

```bash
nano ~/.gz/sim/models/my_mesh_model/model.config
```

Add the following content:

```xml
<?xml version="1.0" ?>
<model>
  <name>my_mesh_model</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>
    A model based on a DAE mesh file for Gazebo simulation.
  </description>
</model>
```

## Step 4: Create the Model SDF File

```bash
nano ~/.gz/sim/models/my_mesh_model/model.sdf
```

Add the following content:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="my_mesh_model">
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <!-- Adjust the pose as needed for your mesh orientation -->
      <pose>0 0 0 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://my_mesh_model/meshes/mesh.dae</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>model://my_mesh_model/meshes/mesh.dae</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

Make sure to replace "mesh.dae" with your actual mesh filename.

## Step 5: Create the World SDF File

Create a directory for your world files:

```bash
mkdir -p ~/gz_worlds
nano ~/gz_worlds/my_mesh_world.sdf
```

Add the following content:

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="my_mesh_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Your Mesh Model -->
    <model name="my_mesh_model">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <pose>0 0 0 0 0 0</pose>
        <visual name="visual">
          <geometry>
            <mesh>
              <uri>model://my_mesh_model/meshes/mesh.dae</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <mesh>
              <uri>model://my_mesh_model/meshes/mesh.dae</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>10 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-1 0 -0.9</direction>
    </light>

    <light type="directional" name="ambient">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.2 0.2 0.2 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>0 0 -1</direction>
    </light>
  </world>
</sdf>
```

## Step 6: Launch with Gazebo Sim

Make sure Gazebo Sim can find your models by setting the environment variable:

```bash
# Set the model path environment variable
export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}:~/.gz/sim/models

# Launch Gazebo Sim with your world
gz sim ~/gz_worlds/my_mesh_world.sdf
```

For convenience, you can add the export command to your `.bashrc` file to make it permanent:

```bash
echo 'export GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}:~/.gz/sim/models' >> ~/.bashrc
source ~/.bashrc
```

## Alternative Method: Using Absolute Paths

If you're having trouble with the model path, you can use absolute paths as a more reliable alternative:

```bash
nano ~/gz_worlds/my_mesh_world_absolute.sdf
```

Replace the mesh URI section with absolute paths:

```xml
<uri>file:///home/username/.gz/sim/models/my_mesh_model/meshes/mesh.dae</uri>
```

Then launch with:

```bash
gz sim ~/gz_worlds/my_mesh_world_absolute.sdf
```

## Custom Lighting Example

To create a scene with custom lighting (such as a sun-like light source at a scaled distance), use:

```xml
<!-- Sun-like light source scaled to moon distance (0.001 scale) -->
<!-- Average Earth-Sun distance is about 149.6 million km -->
<!-- Scaled sun distance is approximately 149.6 km -->
<light type="directional" name="sun">
  <cast_shadows>true</cast_shadows>
  <pose>149.6 0 100 0 0 0</pose>
  <diffuse>0.8 0.8 0.8 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <attenuation>
    <range>1000</range>
    <constant>0.9</constant>
    <linear>0.01</linear>
    <quadratic>0.001</quadratic>
  </attenuation>
  <direction>-1 0 -0.9</direction>
</light>
```

## Tips for Working with Mesh Files

1. **Mesh Orientation**: If your mesh appears with incorrect orientation, adjust the pose values in the model's link section.

2. **Scaling**: If your mesh is too large or small, adjust the scale values in the mesh section.

3. **Mesh Complexity**: High-polygon meshes may cause performance issues. Consider simplifying complex meshes.

4. **Material and Texture**: Make sure any textures referenced in your DAE file are accessible at the paths specified in the DAE file.

5. **Mesh Validation**: If you have issues loading the mesh, check its validity with tools like Blender or MeshLab.

## Troubleshooting

1. **Mesh Not Found**:
   - Verify the path in the URI is correct
   - Check file permissions
   - Try using absolute paths instead of relative paths

2. **Model Appears at the Wrong Location**:
   - Adjust the pose values in the model and link sections

3. **Missing Textures**:
   - Check if your DAE file contains embedded textures or references external files
   - Ensure texture files are in the correct location

4. **Performance Issues**:
   - Simplify complex meshes
   - Reduce the complexity of collision geometries

5. **World Doesn't Load**:
   - Check for syntax errors in your SDF files
   - Verify that Gazebo Sim can find all referenced resources
   - Check Gazebo Sim logs for specific error messages: `journalctl -b -u gz-sim`
