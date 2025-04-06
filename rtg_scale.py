from PIL import Image
import depth_pro
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import open3d as o3d

def depth_to_pointcloud(
    image_path, 
    camera_altitude=None, 
    scale_factor=1.0,
    # Post-processing options
    remove_outliers=True,
    outlier_nb_neighbors=20,
    outlier_std_ratio=2.0,
    voxel_size=0.01,
    estimate_normals=True,
    normal_radius=0.1,
    normal_max_nn=30,
    # Mesh options
    create_mesh=True,
    mesh_depth=9,
    mesh_scale=1.1,
    density_quantile=0.1,
    # Background filtering
    filter_background=True,
    background_threshold=0.9,  # Much more permissive
    depth_filter_type='absolute',  # 'relative' or 'absolute'
    absolute_depth_max=None,  # Maximum depth to keep (in units after scaling)
    # Output options
    save_formats=['ply'],
    # Color adjustment options
    gamma_correction=1.0,
    # Debug options
    save_depth_histogram=True
):
    """
    Convert depth data to colored point cloud PLY file and calculate physical dimensions
    """
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    
    # Load and preprocess an image
    image, _, f_px = depth_pro.load_rgb(image_path)
    img_np = np.array(image)
    image_for_depth = transform(image)
    
    # Run inference with f_px parameter
    prediction = model.infer(image_for_depth, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m]
    focal_length_px = prediction["focallength_px"]  # Focal length in pixels
    
    # Convert depth tensor to numpy array
    depth_np = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.array(depth)

    # Save depth histogram before any adjustments for diagnostic purposes
    if save_depth_histogram:
        plt.figure(figsize=(10, 6))
        plt.hist(depth_np.flatten(), bins=50)
        plt.xlabel('Depth (m)')
        plt.ylabel('Frequency')
        plt.title('Raw Depth Distribution Before Adjustments')
        plt.savefig(f"{os.path.splitext(image_path)[0]}_raw_depth_hist.png")
        plt.close()
        print(f"Raw depth histogram saved (before adjustments)")
        print(f"Raw depth stats - Min: {np.min(depth_np):.2f}m, Max: {np.max(depth_np):.2f}m, Mean: {np.mean(depth_np):.2f}m")

    # Apply gamma correction to depth values if needed
    if gamma_correction != 1.0:
        max_depth = np.max(depth_np)
        depth_np = np.power(depth_np / max_depth, gamma_correction) * max_depth
        print(f"Applied gamma correction with value: {gamma_correction:.2f}")
    
    # Apply altitude correction if provided
    original_depth_np = depth_np.copy()  # Save for comparison
    if camera_altitude is not None:
        # Calculate a correction factor based on the ratio of known altitude to predicted average depth
        avg_predicted_depth = np.mean(depth_np)
        correction_factor = camera_altitude / avg_predicted_depth
        
        # Apply the correction factor to all depth values
        depth_np = depth_np * correction_factor 
        print(f"Applied altitude correction with factor: {correction_factor:.4f}")
    
    # Apply additional scale factor if needed
    if scale_factor != 1.0:
        depth_np = depth_np * scale_factor
        print(f"Applied manual scale factor: {scale_factor:.4f}")
    
    # Save depth histogram after adjustments for diagnostic purposes
    if save_depth_histogram:
        plt.figure(figsize=(10, 6))
        plt.hist(depth_np.flatten(), bins=50)
        plt.xlabel('Depth (scaled units)')
        plt.ylabel('Frequency')
        plt.title('Depth Distribution After Adjustments')
        plt.savefig(f"{os.path.splitext(image_path)[0]}_adjusted_depth_hist.png")
        plt.close()
        print(f"Adjusted depth histogram saved")
        print(f"Adjusted depth stats - Min: {np.min(depth_np):.2f}, Max: {np.max(depth_np):.2f}, Mean: {np.mean(depth_np):.2f}")

    # Get image dimensions
    height_px, width_px = depth_np.shape
    
    # Create meshgrid for pixel coordinates
    xx, yy = np.meshgrid(np.arange(width_px), np.arange(height_px))
    
    # Calculate 3D coordinates from depth
    cx, cy = width_px / 2, height_px / 2
    x = (xx - cx) * depth_np / focal_length_px
    y = (yy - cy) * depth_np / focal_length_px
    z = depth_np
    
    # Calculate physical dimensions (as in original code)
    width_at_depth = width_px * depth_np / focal_length_px
    height_at_depth = height_px * depth_np / focal_length_px
    
    min_depth = np.min(depth_np)
    max_depth = np.max(depth_np)
    avg_depth = np.mean(depth_np)
    
    width_at_min_depth = width_px * min_depth / focal_length_px
    height_at_min_depth = height_px * min_depth / focal_length_px
    
    width_at_max_depth = width_px * max_depth / focal_length_px
    height_at_max_depth = height_px * max_depth / focal_length_px
    
    width_at_avg_depth = width_px * avg_depth / focal_length_px
    height_at_avg_depth = height_px * avg_depth / focal_length_px
    
    if isinstance(width_at_depth, torch.Tensor):
        width_at_depth = width_at_depth.detach().cpu().numpy()
    if isinstance(height_at_depth, torch.Tensor):
        height_at_depth = height_at_depth.detach().cpu().numpy()

    min_width = np.min(width_at_depth)
    max_width = np.max(width_at_depth)
    min_height = np.min(height_at_depth)
    max_height = np.max(height_at_depth)
    
    scene_dimensions = {
        "min_depth_m": min_depth,
        "max_depth_m": max_depth,
        "avg_depth_m": avg_depth,
        "width_at_min_depth_m": width_at_min_depth,
        "height_at_min_depth_m": height_at_min_depth,
        "width_at_max_depth_m": width_at_max_depth,
        "height_at_max_depth_m": height_at_max_depth,
        "width_at_avg_depth_m": width_at_avg_depth,
        "height_at_avg_depth_m": height_at_avg_depth,
        "min_width_m": min_width,
        "max_width_m": max_width,
        "min_height_m": min_height,
        "max_height_m": max_height,
    }
    
    # Flatten the arrays for PLY format
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    
    # Get RGB values from the original image
    if len(img_np.shape) == 3:  # Color image
        r = img_np[:, :, 0].flatten()
        g = img_np[:, :, 1].flatten()
        b = img_np[:, :, 2].flatten()
    else:  # Grayscale image
        r = g = b = img_np.flatten()
    
    # IMPORTANT: Modified depth filtering approach with multiple options
    valid_indices = np.ones_like(z_flat, dtype=bool)  # Start with all points valid
    
    if filter_background:
        if depth_filter_type == 'relative':
            # Use relative threshold (% of max depth)
            depth_threshold = max_depth * background_threshold
            valid_indices = z_flat < depth_threshold
            print(f"Relative depth filtering: keeping points with depth < {depth_threshold:.2f} ({background_threshold*100}% of max)")
        
        elif depth_filter_type == 'absolute':
            # Use absolute threshold value (in units after scaling)
            if absolute_depth_max is None:
                # Default to 110% of mean depth if not specified
                absolute_depth_max = avg_depth * 1.1
            
            valid_indices = z_flat < absolute_depth_max
            print(f"Absolute depth filtering: keeping points with depth < {absolute_depth_max:.2f}")
        
        elif depth_filter_type == 'percentile':
            # Keep points below a certain percentile of depth
            depth_threshold = np.percentile(z_flat, background_threshold * 100)
            valid_indices = z_flat < depth_threshold
            print(f"Percentile depth filtering: keeping points below {background_threshold*100}th percentile ({depth_threshold:.2f})")
        
        else:  # 'none' or invalid value
            print("No depth filtering applied")
    
    # Apply the filter
    points_before = len(z_flat)
    x_flat = x_flat[valid_indices]
    y_flat = y_flat[valid_indices]
    z_flat = z_flat[valid_indices]
    r = r[valid_indices]
    g = g[valid_indices]
    b = b[valid_indices] 
    
    points_after = len(z_flat)
    print(f"Filtering point cloud: keeping {points_after} of {points_before} points ({points_after/points_before*100:.1f}%)")
    
    # Check if we have any points left after filtering
    if points_after == 0:
        print("ERROR: No points remain after filtering. Try adjusting filter parameters.")
        # Create empty point cloud
        pcd = o3d.geometry.PointCloud()
    else:
        # Create Open3D point cloud with filtered points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack([x_flat, y_flat, z_flat]))
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([r, g, b]) / 255.0)
        
        # Process the point cloud
        if len(pcd.points) > 0:
            if remove_outliers:
                # Adjust parameters based on point count
                if len(pcd.points) < 100:
                    print("Few points remain, using relaxed outlier removal parameters")
                    outlier_nb_neighbors = min(outlier_nb_neighbors, 10)
                    outlier_std_ratio = 3.0  # More permissive
                
                pcd, _ = pcd.remove_statistical_outlier(
                    nb_neighbors=outlier_nb_neighbors, 
                    std_ratio=outlier_std_ratio
                )
                print(f"After outlier removal: {len(pcd.points)} points")
            
            # Only downsample if we have enough points
            if len(pcd.points) > 1000:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                print(f"After voxel downsampling: {len(pcd.points)} points")
            
            # Estimate normals if requested and if we have enough points
            if estimate_normals and len(pcd.points) > 10:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=normal_radius, 
                        max_nn=normal_max_nn
                    )
                )
    
    # Save results
    base_name = os.path.splitext(image_path)[0]
    scaling_suffix = f"_alt{camera_altitude}_scale{scale_factor}" if camera_altitude else f"_scale{scale_factor}"
    
    # Create and save a visualization of the depth map
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_np, cmap='viridis')
    plt.colorbar(label='Depth (scaled units)')
    plt.title('Depth Map (After Scaling)')
    depth_vis_file = f"{base_name}{scaling_suffix}_depth_vis.png"
    plt.savefig(depth_vis_file)
    plt.close()
    print(f"Depth visualization saved to {depth_vis_file}")
    
    # Save the point cloud if we have points
    output_file = f"{base_name}{scaling_suffix}_pointcloud.ply"
    
    if len(pcd.points) > 0:
        # Save using Open3D
        for format in save_formats:
            out_path = f"{base_name}{scaling_suffix}_pointcloud.{format}"
            if format == 'ply':
                o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
            elif format == 'pcd':
                o3d.io.write_point_cloud(out_path, pcd)
            elif format == 'xyz':
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)
                with open(out_path, 'w') as f:
                    for i in range(len(points)):
                        x, y, z = points[i]
                        r, g, b = colors[i]
                        nx, ny, nz = normals[i]
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f} {nx:.6f} {ny:.6f} {nz:.6f}\n")
            print(f"Saved point cloud as: {out_path}")
        
        # Create and save mesh if enabled and we have enough points
        if create_mesh and len(pcd.points) > 100:
            try:
                # Ensure we have normals
                if not pcd.has_normals():
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=normal_radius,
                            max_nn=normal_max_nn
                        )
                    )
                
                # Try to orient normals
                try:
                    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
                except Exception as e:
                    print(f"Warning: Could not orient normals: {e}")
                
                # Create mesh
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd,
                    depth=mesh_depth,
                    scale=mesh_scale,
                    linear_fit=False
                )
                
                # Remove low-density vertices
                vertices_to_remove = densities < np.quantile(densities, density_quantile)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
                # Cleanup mesh
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                # Save mesh
                for format in ['ply', 'obj']:
                    mesh_path = f"{base_name}{scaling_suffix}_mesh.{format}"
                    o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=True)
                    print(f"Saved mesh as: {mesh_path}")
            
            except Exception as e:
                print(f"Mesh creation failed: {e}")
    else:
        print("WARNING: No points in point cloud, skipping mesh creation.")
    
    # Always save direct PLY output as backup
    # Create PLY header and data for direct output
    ply_header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".format(len(x_flat))
    
    # Prepare data for PLY file
    vertices = np.column_stack([x_flat, y_flat, z_flat, r, g, b])
    
    # Save to PLY file directly
    direct_output_file = f"{base_name}{scaling_suffix}_pointcloud_direct.ply"
    with open(direct_output_file, 'w') as f:
        f.write(ply_header)
        np.savetxt(f, vertices, fmt='%f %f %f %d %d %d')
    
    print(f"Direct point cloud saved to {direct_output_file}")
    
    # Save scene dimensions information to a text file
    dimensions_file = f"{base_name}{scaling_suffix}_dimensions.txt"
    with open(dimensions_file, 'w') as f:
        f.write("Scene Physical Dimensions:\n")
        f.write("==========================\n\n")
        f.write(f"Image Resolution: {width_px}x{height_px} pixels\n")
        f.write(f"Focal Length: {focal_length_px} pixels\n\n")
        
        if camera_altitude:
            f.write(f"Camera Altitude: {camera_altitude} meters\n")
        if scale_factor != 1.0:
            f.write(f"Manual Scale Factor: {scale_factor}\n\n")
        
        f.write("Depth Information:\n")
        f.write(f"  Minimum Depth: {min_depth:.4f} units\n")
        f.write(f"  Maximum Depth: {max_depth:.4f} units\n")
        f.write(f"  Average Depth: {avg_depth:.4f} units\n\n")
        
        f.write("Scene Dimensions:\n")
        f.write(f"  Width at Minimum Depth: {width_at_min_depth:.4f} units\n")
        f.write(f"  Height at Minimum Depth: {height_at_min_depth:.4f} units\n\n")
        
        f.write(f"  Width at Maximum Depth: {width_at_max_depth:.4f} units\n")
        f.write(f"  Height at Maximum Depth: {height_at_max_depth:.4f} units\n\n")
        
        f.write(f"  Width at Average Depth: {width_at_avg_depth:.4f} units\n")
        f.write(f"  Height at Average Depth: {height_at_avg_depth:.4f} units\n\n")
        
        f.write("Overall Scene Bounds:\n")
        f.write(f"  Minimum Width: {min_width:.4f} units\n")
        f.write(f"  Maximum Width: {max_width:.4f} units\n")
        f.write(f"  Minimum Height: {min_height:.4f} units\n")
        f.write(f"  Maximum Height: {max_height:.4f} units\n")
    
    print(f"Scene dimensions saved to {dimensions_file}")
    
    return direct_output_file, depth_np, focal_length_px, scene_dimensions

# Usage
if __name__ == "__main__":
    image_path = "data/moon.png"
    
    # For lunar orbital photos, specify the camera altitude
    camera_altitude = 100000*5  # 100 km in meters
    
    # Scale factor - use 1.0 initially to debug the filtering
    scale_factor = 1.0
    
    try:
        output_file, depth, focal_length, dimensions = depth_to_pointcloud(
            image_path, 
            camera_altitude=camera_altitude, 
            scale_factor=scale_factor,
            
            # Post-processing options - use default values initially
            remove_outliers=True,
            outlier_nb_neighbors=20,
            outlier_std_ratio=2.0,
            voxel_size=0.05,
            
            # Background filtering - try absolute threshold approach
            filter_background=True,
            depth_filter_type='percentile',  # Use percentile-based filtering
            background_threshold=0.95,  # Keep bottom 95% of points
            
            # Mesh options
            create_mesh=True,
            mesh_depth=8,
            
            # Output options
            save_formats=['ply'],
            
            # Debug options
            save_depth_histogram=True,
            
            # Color adjustment options
            gamma_correction=0.8
        )
        
        print(f"Processing complete for {image_path}")
        
        # After successful run, try with desired scale factor
        if os.path.exists(output_file):
            print("\nFirst attempt successful. Now trying with smaller scale factor...")
            
            # Now try with smaller scale factor
            output_file2, depth2, focal_length2, dimensions2 = depth_to_pointcloud(
                image_path, 
                camera_altitude=camera_altitude, 
                scale_factor=0.001,  # Now try the smaller scale
                
                # Keep same filtering parameters
                filter_background=True,
                depth_filter_type='percentile',
                background_threshold=0.95,
                
                # Mesh options
                create_mesh=True,
                
                # Output options
                save_formats=['ply'],
                
                # Debug options
                save_depth_histogram=True,
                
                # Color adjustment options
                gamma_correction=0.8
            )
            
            print(f"Second processing run complete for {image_path}")
    
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()