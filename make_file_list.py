import os

# 指定训练集路径和测试集路径
train_path = '/media/lz-4060ti-linux/SE/SE/VB_DEMAND_16K(copy)/clean_train'
test_path = '/media/lz-4060ti-linux/SE/SE/VB_DEMAND_16K/clean_test'

# 获取训练集路径下的所有wav文件名（不包含后缀）
train_file_names = [os.path.splitext(file)[0] for file in os.listdir(train_path) if file.endswith('.wav')]

# 生成training.txt文件并将文件名写入
with open('training.txt', 'w') as train_file:
    train_file.write('\n'.join(train_file_names))

# 获取测试集路径下的所有wav文件名（不包含后缀）
test_file_names = [os.path.splitext(file)[0] for file in os.listdir(test_path) if file.endswith('.wav')]

# 生成test.txt文件并将文件名写入
with open('test.txt', 'w') as test_file:
    test_file.write('\n'.join(test_file_names))
