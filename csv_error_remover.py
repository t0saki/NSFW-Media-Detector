import pandas as pd


def remove_error_predictions(input_file, output_file):
    """
    读取CSV文件，删除'Prediction'列中包含'ERROR'的行，
    并将清理后的数据保存到新的CSV文件中。

    参数:
    input_file (str): 输入的CSV文件名。
    output_file (str): 输出的CSV文件名。
    """
    try:
        # 读取CSV文件到pandas DataFrame中
        df = pd.read_csv(input_file)
        print(f"成功读取文件: {input_file}")
        print(f"原始数据共有 {len(df)} 行。")

        # 过滤掉'Prediction'列包含'ERROR'的行
        # 使用 .str.contains() 来查找子字符串'ERROR'
        # `na=False`确保非字符串数据不会导致错误
        # `~` 操作符用于反转条件，即选择不包含'ERROR'的行
        cleaned_df = df[~df['Prediction'].str.contains('ERROR', na=False)]

        # 将清理后的DataFrame写入新的CSV文件
        # index=False 表示不将DataFrame的索引写入CSV文件
        cleaned_df.to_csv(output_file, index=False)

        print(f"清理后的数据共有 {len(cleaned_df)} 行。")
        print(f"已成功将清理后的数据保存到: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_file}'。请检查文件名和路径是否正确。")
    except KeyError:
        print(f"错误：输入文件中找不到名为 'Prediction' 的列。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


# --- 使用示例 ---
# 定义输入和输出文件名
input_filename = 'nsfw_detection_results.csv'
output_filename = 'output.csv'

try:
    remove_error_predictions(input_filename, output_filename)

except Exception as e:
    print(f"创建示例文件时出错: {e}")
