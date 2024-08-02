from gradslam.structures.pointclouds import Pointclouds
import open3d as o3d

def main():   
    scene = "0046" 
    dataset_path = f"/datasets/Scannet/scans/scene{scene}_00/rgb_cloud"
    pointcloud = Pointclouds.load_pointcloud_from_h5(dataset_path)
    pcd_xyz = pointcloud.points_padded[0].numpy()
    pcd_color = pointcloud.colors_padded[0].numpy()/255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
    pcd.colors = o3d.utility.Vector3dVector(pcd_color)
    # o3d.visualization.draw_geometries([pcd])
    # print(pcd_xyz.numpy().shape)
    # print(pcd_color)
    o3d.io.write_point_cloud(f"scene{scene}_00_rgb.ply", pcd.voxel_down_sample(voxel_size=0.1))


if __name__ == '__main__':
    main()