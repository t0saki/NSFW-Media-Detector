import pandas as pd
import os

# 定义你的CSV文件路径
file_path = 'nsfw_detection_results.csv'  # 请将 'your_file.csv' 替换为你的实际文件路径
# prefix 已使用Unix风格分隔符
prefix = "Compressed/2025-09-15/DSMPhotos_Converted"

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg",
                              ".png", ".bmp", ".webp", ".heic", ".heif", ".avif"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误: 找不到文件 {file_path}")
else:
    try:
        # 从文件中读取CSV，指定编码为 'utf-8'
        df = pd.read_csv(file_path, encoding='utf-8')

        # 过滤 'NSFW Probability' > 0.95 的行
        filtered_df = df[df['NSFW Probability'] > 0.8].copy()

        # --- 新增逻辑：识别并排除 Live Photo 中的视频部分 ---

        # 获取所有高概率文件路径的列表
        all_paths = filtered_df['File Path'].tolist()

        # 提取所有 .HEIC 文件的基础名称（路径+文件名，不含后缀），用于识别Live Photo
        # 例如: 'F:\\DSMPhotos\\album\\IMG_1234.HEIC' -> 'F:\\DSMPhotos\\album\\IMG_1234'
        live_photo_basenames = {os.path.splitext(
            p)[0] for p in all_paths if os.path.splitext(p)[1].lower() == '.heic'}

        # 创建一个新的列表，用于存放需要保留的路径
        paths_to_keep = []
        for path in all_paths:
            base_name, ext = os.path.splitext(path)
            ext_lower = ext.lower()

            # 检查当前文件是否为视频，并且其基础名称是否存在于 live_photo_basenames 集合中
            is_live_photo_video = ext_lower in SUPPORTED_VIDEO_EXTENSIONS and base_name in live_photo_basenames

            # 如果不是 Live Photo 关联的视频，则保留
            if not is_live_photo_video:
                paths_to_keep.append(path)

        # 基于过滤后的路径列表，创建最终的DataFrame
        final_df = filtered_df[filtered_df['File Path'].isin(
            paths_to_keep)].copy()

        # 对筛选后的DataFrame按照 'File Path' 进行排序
        sorted_df = final_df.sort_values(by='File Path')

        # --- 结束新增逻辑 ---

        # 定义转换路径的函数 (修改为使用 '/' 分隔符)

        def transform_path(file_path):
            # 移除 'F:\', 并统一输入的分隔符为'\'以便分割
            relative_path = file_path.replace(
                'F:\\DSMPhotos\\', '').replace('/', '\\')

            # 分割路径以获取最后一个目录和文件名
            path_parts = relative_path.split('\\')
            if len(path_parts) > 1:
                last_dir = path_parts[-2]
                path_parts.insert(-1, last_dir)
                # 使用 Unix 风格的 '/' 连接路径
                transformed_path = '/'.join(path_parts)
            else:
                transformed_path = relative_path

            return transformed_path

        # --- 新增逻辑：应用所有修改并生成最终输出列表 ---

        output_lines = []

        for path in sorted_df['File Path']:
            # 1. 应用原始的路径转换
            transformed = transform_path(path)

            # 2. 修改后缀
            # 注意：os.path.splitext 在处理'/'时表现良好
            base, original_ext = os.path.splitext(transformed)

            new_ext = ''
            if original_ext.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                new_ext = '.avif'
            elif original_ext.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                new_ext = '.mp4'
            else:
                # 如果有其他类型文件，保留原后缀
                new_ext = original_ext

            # 3. 添加前缀并组合成最终路径 (使用 '/' 分隔符)
            final_path = f"{prefix}/{base}{new_ext}"
            output_lines.append(final_path)

        # 将转换后的路径保存到新的txt文件
        output_file = 'nsfw_paths.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')

        # --- 结束新增逻辑 ---

        print(f"已成功将符合条件的路径保存到 {output_file} 文件中。")
        print("以下是文件内容:")

        # 打印文件内容以供验证
        with open(output_file, 'r', encoding='utf-8') as f:
            print(f.read())

    except UnicodeDecodeError:
        print("处理文件时发生编码错误，请检查你的CSV文件是否为 UTF-8 编码。")
    except Exception as e:
        print(f"处理文件时发生其他错误: {e}")
