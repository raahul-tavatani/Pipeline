from pypcd.pypcd import PointCloud

pc = PointCloud.from_path("C:/Pipeline/saved_data/single_frame.pcd")

print("Fields:", pc.fields)
print("Sample x values:", pc.pc_data['x'][:5])
