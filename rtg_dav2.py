# Author: Antar Mazumder
# Email: antar_mazumder@mines.edu
# Description: This script processes images to generate depth maps, 
# converts them into point clouds, and creates meshes using depth estimation.
# The point clouds and meshes are saved in multiple formats such as .ply, .pcd, and .xyz.
# Dependencies: Open3D, NumPy, PyTorch, Pillow, OpenCV, DepthAnythingV2

import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch
import traceback

from depth_anything_v2.dpt import DepthAnythingV2

# Configuration - All parameters can be modified directly in this dictionary
CONFIG = {
    # Model parameters
    'encoder': 'vitb',  
    'load_from': "checkpoints/depth_anything_v2_vitb.pth",
    'max_depth': 2500,  # Maximum depth value for scaling
    
    # Input/Output parameters
    'img_path': 'mars.png',  # Path to input image or directory
    'outdir': './vis_pointcloud',  # Directory to save results
    'save_formats': ['ply', 'xyz'],  # File formats to save point clouds
    
    # Camera parameters
    'focal_length_x': 470.4,  # Focal length for X dimension
    'focal_length_y': 470.4,  # Focal length for Y dimension
    
    # Point cloud processing parameters
    'depth_filter_threshold': 0.002,  # Threshold for filtering background points (0.01-0.1)
    'min_points_after_filter': 100,  # Minimum number of points to continue processing
    'voxel_size': 0.0010,  # Voxel size for downsampling (smaller = more detail)
    'remove_outliers': True,  # Enable outlier removal
    
    # Statistical outlier removal parameters
    'outlier_nb_neighbors': 35,  # Number of neighbors for outlier detection
    # increase this if you find holes!
    'outlier_std_ratio': 3.5,  # Standard deviation ratio for outlier detection; higher = more detailed
    
    # Normal estimation parameters
    'normal_radius': 0.1,  # Radius for normal estimation
    'normal_max_nn': 40,  # Maximum number of neighbors for normal estimation
    
    # Mesh creation parameters
    'poisson_depth': 11,  # Octree depth for Poisson reconstruction (8-12, higher = more detail)
    'poisson_scale': 1.3,  # Scale factor for Poisson reconstruction (1.0-2.0)
    'density_threshold_percentile': 0.01,  # Percentile for removing low-density vertices (0.01-0.2); lower--> more low density vertices kept
    'fill_holes': True,  # Enable hole filling in the mesh
    'max_hole_size': 300,  # Maximum boundary vertices to fill in a hole
    'mesh_cleanup': True,  # Enable mesh cleanup operations
    'smoothing_iterations': 1,  # Number of smoothing iterations (0-3, 0 = no smoothing)
}

def process_point_cloud(pcd, config):
    """
    Process point cloud to improve quality and prepare for mesh generation
    
    Args:
        pcd: Open3D point cloud object
        config: Dictionary with processing parameters
        
    Returns:
        Processed Open3D point cloud
    """
    # Ensure we have points before processing
    if len(pcd.points) == 0:
        print("WARNING: Point cloud has no points. Skipping processing.")
        return pcd
    
    # Remove statistical outliers if enabled
    if config['remove_outliers']:
        # Check if we have enough points for statistical outlier removal
        if len(pcd.points) > config['outlier_nb_neighbors']:
            print(f"Removing statistical outliers (neighbors: {config['outlier_nb_neighbors']}, std ratio: {config['outlier_std_ratio']})")
            pcd, outlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=config['outlier_nb_neighbors'], 
                std_ratio=config['outlier_std_ratio']
            )
            print(f"Removed {len(outlier_indices)} outlier points")
        else:
            print(f"WARNING: Not enough points ({len(pcd.points)}) for statistical outlier removal.")
    
    # Voxel downsampling only if we have points and voxel size > 0
    if len(pcd.points) > 0 and config['voxel_size'] > 0:
        print(f"Downsampling with voxel size: {config['voxel_size']}")
        points_before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=config['voxel_size'])
        print(f"Downsampled from {points_before} to {len(pcd.points)} points")
    
    # Estimate normals if we have points left
    if len(pcd.points) > 0:
        print(f"Estimating normals (radius: {config['normal_radius']}, max_nn: {config['normal_max_nn']})")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config['normal_radius'], 
                max_nn=config['normal_max_nn']
            )
        )
        print("Normal estimation complete")
    else:
        print("WARNING: No points left after filtering. Cannot estimate normals.")
        # Create a minimal valid point cloud to avoid errors
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))
        pcd.estimate_normals()
    
    return pcd

def create_mesh_from_points(pcd, config):
    """
    Convert point cloud to mesh using surface reconstruction
    
    Args:
        pcd: Open3D point cloud object with normals
        config: Dictionary with mesh creation parameters
        
    Returns:
        Open3D triangle mesh
    """
    # Check if we have enough points for mesh creation
    if len(pcd.points) < 4:
        print("ERROR: Not enough points to create a mesh.")
        # Return an empty mesh instead of failing
        return o3d.geometry.TriangleMesh()
    
    # Ensure we have normals before proceeding
    if not pcd.has_normals():
        print("Estimating normals as they were not present")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=config['normal_radius'], 
                max_nn=config['normal_max_nn']
            )
        )
    
    # Only orient normals if we have them
    if pcd.has_normals():
        try:
            print("Orienting normals towards camera")
            pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        except Exception as e:
            print(f"WARNING: Could not orient normals: {str(e)}")
            # Re-estimate normals as fallback
            print("Re-estimating normals as fallback")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=config['normal_radius'], 
                    max_nn=config['normal_max_nn']
                )
            )
    
    # Poisson surface reconstruction with error handling
    try:
        print(f"Creating mesh with Poisson reconstruction (depth: {config['poisson_depth']}, scale: {config['poisson_scale']})")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=config['poisson_depth'],
            width=0,  # Automatic width estimation
            scale=config['poisson_scale'],
            linear_fit=False
        )
        
        print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        # Only remove vertices if we have a valid mesh
        if len(mesh.vertices) > 0 and len(densities) > 0:
            density_threshold = config['density_threshold_percentile']
            if len(densities) < 10:  # Adjust threshold for small point clouds
                density_threshold = 0.01
                
            print(f"Removing low-density vertices (threshold percentile: {density_threshold})")
            vertices_before = len(mesh.vertices)
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            print(f"Removed {vertices_before - len(mesh.vertices)} low-density vertices")
        
            # Cleanup mesh if enabled
            if config['mesh_cleanup']:
                print("Cleaning up mesh (removing degenerate triangles, duplicates, etc.)")
                triangles_before = len(mesh.triangles)
                vertices_before = len(mesh.vertices)
                
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                print(f"Cleanup removed {triangles_before - len(mesh.triangles)} triangles and {vertices_before - len(mesh.vertices)} vertices")
            
            # Try to fill holes if enabled
            if config['fill_holes']:
                try:
                    print(f"Filling holes (max hole size: {config['max_hole_size']})")
                    holes_filled = mesh.fill_holes(config['max_hole_size'])
                    print(f"Holes filled: {holes_filled}")
                    
                    # Apply smoothing if iterations > 0
                    if config['smoothing_iterations'] > 0:
                        print(f"Smoothing mesh ({config['smoothing_iterations']} iterations)")
                        mesh = mesh.filter_smooth_simple(number_of_iterations=config['smoothing_iterations'])
                        
                except Exception as e:
                    print(f"WARNING: Hole filling failed: {str(e)}")
        
        return mesh
    
    except Exception as e:
        print(f"ERROR in mesh creation: {str(e)}")
        return o3d.geometry.TriangleMesh()

def save_point_cloud(pcd, base_path, formats=['ply']):
    """
    Save point cloud in multiple formats
    
    Args:
        pcd: Open3D point cloud object
        base_path: Base file path without extension
        formats: List of file formats to save
    """
    for format in formats:
        out_path = f"{base_path}.{format}"
        if format == 'ply':
            o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
        elif format == 'pcd':
            o3d.io.write_point_cloud(out_path, pcd)
        elif format == 'xyz':
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Check if normals exist
            has_normals = pcd.has_normals()
            if has_normals:
                normals = np.asarray(pcd.normals)
            
            with open(out_path, 'w') as f:
                for i in range(len(points)):
                    x, y, z = points[i]
                    r, g, b = colors[i]
                    
                    if has_normals:
                        nx, ny, nz = normals[i]
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")
                    else:
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        print(f"Saved point cloud as: {out_path}")

def save_mesh(mesh, base_path, formats=['ply']):
    """
    Save mesh in multiple formats
    
    Args:
        mesh: Open3D triangle mesh object
        base_path: Base file path without extension
        formats: List of file formats to save
    """
    for format in formats:
        out_path = f"{base_path}_mesh.{format}"
        if format == 'ply':
            o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=True)
        elif format == 'obj':
            o3d.io.write_triangle_mesh(out_path, mesh)
        print(f"Saved mesh as: {out_path}")

def main():
    """
    Main function to process images and create 3D models
    """
    print("=== Starting Depth Map to 3D Conversion ===")
    
    # Check device for PyTorch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Model configurations for different ViT sizes
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Load depth estimation model
    print(f"Loading depth model: {CONFIG['encoder']} from {CONFIG['load_from']}")
    depth_anything = DepthAnythingV2(**model_configs[CONFIG['encoder']])
    depth_anything.load_state_dict(torch.load(CONFIG['load_from'], map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    print("Depth model loaded successfully")

    # Get list of image files to process
    if os.path.isfile(CONFIG['img_path']):
        filenames = [CONFIG['img_path']]
    else:
        filenames = glob.glob(os.path.join(CONFIG['img_path'], '**/*'), recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filter for images

    if not filenames:
        print(f"ERROR: No valid image files found in {CONFIG['img_path']}")
        return

    # Create output directory
    os.makedirs(CONFIG['outdir'], exist_ok=True)
    print(f"Will process {len(filenames)} images and save results to {CONFIG['outdir']}")

    # Process each image
    for k, filename in enumerate(filenames):
        print(f"\n=== Processing {k+1}/{len(filenames)}: {filename} ===")

        try:
            # Load image
            print("Loading image")
            color_image = Image.open(filename).convert('RGB')
            width, height = color_image.size
            print(f"Image size: {width}x{height}")

            image = cv2.imread(filename)
            if image is None:
                print(f"ERROR: Failed to load image: {filename}")
                continue

            # Generate depth map
            print("Generating depth map")
            pred = depth_anything.infer_image(image, height)
            
            # Scale depth and apply gamma correction for better depth detail
            pred = (pred / pred.max()) * CONFIG['max_depth']
            pred = np.power(pred / CONFIG['max_depth'], 0.8) * CONFIG['max_depth']  # Gamma correction
            print(f"Depth range: {pred.min():.2f} to {pred.max():.2f}")

            # Resize depth map to match original image size
            resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

            # Create 3D coordinates
            print("Creating 3D point cloud from depth map")
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x = (x - width / 2) / CONFIG['focal_length_x']
            y = (y - height / 2) / CONFIG['focal_length_y']
            z = np.array(resized_pred)

            # Filter out background points using depth threshold
            depth_threshold = CONFIG['depth_filter_threshold']
            mask = z > (z.max() * depth_threshold)
            points_after_filter = np.sum(mask)
            print(f"Filtered points: {points_after_filter} remaining out of {width*height} (threshold: {depth_threshold})")
            
            # Check if we have enough points after filtering
            if points_after_filter < CONFIG['min_points_after_filter']:
                print(f"WARNING: Very few points ({points_after_filter}) after depth filtering. Adjusting threshold.")
                # Try a more permissive threshold
                depth_threshold = 0.01
                mask = z > (z.max() * depth_threshold)
                points_after_filter = np.sum(mask)
                print(f"Adjusted filtering: {points_after_filter} points (threshold: {depth_threshold})")
                
                if points_after_filter < CONFIG['min_points_after_filter']:
                    print(f"ERROR: Still too few points ({points_after_filter}). Check your depth map.")
                    continue
            
            # Apply mask to get filtered points
            x = x[mask]
            y = y[mask]
            z = z[mask]
            colors = np.array(color_image)[mask]

            # Create point cloud with 3D coordinates
            points = np.stack((np.multiply(x, z.flatten()), 
                             np.multiply(y, z.flatten()), 
                             z.flatten()), axis=-1)
            colors = colors.reshape(-1, 3) / 255.0
            
            # Verify we have valid points before creating point cloud
            if len(points) == 0 or np.isnan(points).any():
                print("ERROR: No valid points to create point cloud")
                continue
                
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Ensure the point cloud has points before processing
            if len(pcd.points) == 0:
                print("ERROR: Point cloud has no points after creation")
                continue

            # Process the point cloud
            print("Processing point cloud")
            pcd = process_point_cloud(pcd, CONFIG)
            
            # Create mesh from point cloud
            print("Creating mesh from point cloud")
            mesh = create_mesh_from_points(pcd, CONFIG)
            
            # Only save if we have a valid point cloud and mesh
            if len(pcd.points) > 0 and len(mesh.vertices) > 0:
                # Save both point cloud and mesh
                base_path = os.path.join(CONFIG['outdir'], 
                                      os.path.splitext(os.path.basename(filename))[0])
                save_point_cloud(pcd, base_path, CONFIG['save_formats'])
                save_mesh(mesh, base_path, ['ply', 'obj'])
                print(f"Successfully saved point cloud and mesh for {filename}")
            else:
                print(f"WARNING: Empty point cloud or mesh for {filename}. Not saving.")

        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
            print("Full error traceback:")
            traceback.print_exc()
            continue
    
    print("\n=== Processing complete ===")

if __name__ == '__main__':
    main()