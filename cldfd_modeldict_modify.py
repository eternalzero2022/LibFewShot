import torch

# 加载模型的状态字典
state_dict = torch.load('./results/CLDFD-miniImageNet--ravi-ResNet10-5-1-Jan-15-2025-14-05-56/checkpoints/model_best.pth')

# # 打印当前的 keys
# print("Original state_dict keys:", state_dict.keys())

# 删除所有以 'old_student' 开头的键
keys_to_remove = [key for key in state_dict.keys() if key.startswith('old_student')]

for key in keys_to_remove:
    del state_dict[key]

# # 打印修改后的 keys
# print("Modified state_dict keys:", state_dict.keys())

# 保存修改后的状态字典
torch.save(state_dict, './results/CLDFD-miniImageNet--ravi-ResNet10-5-1-Jan-15-2025-14-05-56/checkpoints/model_best.pth')