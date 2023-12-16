import h5py

if __name__ == '__main__':

    # 用你的.h5文件路径替换'your_file.h5'
    with h5py.File('pems-bay.h5', 'r') as file:
        # 检查'speed'是否是一个组
        if 'speed' in file:
            speed_group = file['speed']
            # 列出'speed'组内的所有数据集
            for item in speed_group:
                print(f"在'speed'组内发现对象：{item}")
                # 如果这个项目是一个数据集，打印出它的形状
                if isinstance(speed_group[item], h5py.Dataset):
                    print(f"数据集 '{item}' 的形状：{speed_group[item].shape}")

            # 假设 'block0_values' 是包含数据的二维数组
            if "block0_values" in file['speed']:
                # 读取第一行数据
                first_row_values = file['speed']["block0_values"][0, :]
                print("第一行的速度数据:", first_row_values)

            # 如果你也想读取 'block0_items'，它可能包含列的标签（sensor_id）
            if "block0_items" in file['speed']:
                sensor_ids = file['speed']["block0_items"][:]
                print("传感器ID:", sensor_ids)

            # 将sensor_ids 按照逗号分隔符连接成一个字符串，储存在文件中
            with open("sensor_graph/graph_sensor_ids_pems.txt", "w") as f:
                sensor_ids_str = [str(id) for id in sensor_ids]
                f.write(",".join(sensor_ids_str))