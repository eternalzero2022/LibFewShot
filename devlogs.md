## 添加的模块
- config/backbones/resnet10.yaml
- config/classifiers/CLDFD_pretrain.yaml
- config/cldfd_pretrain.yaml
- core/model/backbone/resnet_10.py
- core/model/finetuning/cldfd_pretrain.py

## 修改的模块
- core/model/backbone/\_\_init\_\_.py
- core/model/finetuning/\_\_init\_\_.py

## 为了实现成功运行做的工作：
- 在core/model/finetuning/\_\_init\_\_.py中添加语句from .cldfd import CLDFD
- 取消了Trainer的依赖，将Trainer中获取当前epoch的代码单独抽象出一个类和一个文件，放在core/utils/runtime_data.py中，防止循环依赖
- 修改了权重读取时候状态字典的每个键，去掉前缀emb_func
- 修改了权重读取路径，将model_best.pth改为emb_func_best.pth
- 将core/data/collates/contrib/autoaugment.py中第175行的astype(np.int)改为astype(int)
- cldfd.py的set_forward_loss中将图像先转变为pil的形式然后再转变为tensor的形式