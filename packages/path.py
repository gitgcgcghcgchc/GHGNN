#
# import os
# import threading
#
# def make_path(path):
#     # 确保路径存在
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)
#
#     new_folder_name = str(subfolder_count)
#     new_folder_path = os.path.join(path, new_folder_name)
#     # # 初始化锁
#     # lock = threading.Lock()
#     #
#     # with lock:  # 确保线程安全
#     #     # 获取当前文件夹数量
#     #     subfolder_count = len([entry for entry in os.scandir(path) if entry.is_dir()])
#     #
#     #     # 创建新的文件夹，文件夹名称为文件夹数量
#     #     new_folder_name = str(subfolder_count)
#     #     new_folder_path = os.path.join(path, new_folder_name)
#     #
#     #     # 检查新文件夹是否已经存在并创建新的文件夹
#     #     # 如果存在，继续增加计数，确保每次运行创建新的文件夹
#         while os.path.exists(new_folder_path):
#             subfolder_count += 1
#             new_folder_name = str(subfolder_count)
#             new_folder_path = os.path.join(path, new_folder_name)
#
#         try:
#             # 创建新文件夹
#             os.makedirs(new_folder_path)
#         except OSError as e:
#             print(f"Error creating folder {new_folder_path}: {e}")
#             return None
#
#     return new_folder_path
#
#
import os
import threading
from datetime import datetime

def make_path(path):
    # 确保路径存在
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # 初始化锁
    lock = threading.Lock()

    with lock:  # 确保线程安全
        # 获取当前时间并格式化为字符串，精确到分钟
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        new_folder_name = current_time
        new_folder_path = os.path.join(path, new_folder_name)

        try:
            # 创建新文件夹
            os.makedirs(new_folder_path)
        except OSError as e:
            print(f"Error creating folder {new_folder_path}: {e}")
            return None

    return new_folder_path