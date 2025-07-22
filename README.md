## 自用的comfyui节点汇总，有BUG但可以不改嘿嘿
![image](https://github.com/user-attachments/assets/276c33dd-5fda-4d75-bbf4-9cbc312c6068)
<img width="3361" height="1470" alt="image" src="https://github.com/user-attachments/assets/1d310277-74d3-482e-9424-dfb2d22af989" />

![image](https://github.com/user-attachments/assets/f80ad400-1cc8-4dc3-b356-3f43b53ec696)
![image](https://github.com/user-attachments/assets/747b5ac8-86f2-4896-9a62-5de6a7e6ddf0)
![image](https://github.com/user-attachments/assets/4f8bbbf3-3e0f-4b0c-92ed-edbffac26a04)
![image](https://github.com/user-attachments/assets/a864adfb-46ce-49d8-ab22-7c675756c24d)
![image](https://github.com/user-attachments/assets/b790341d-22cd-42bf-b1ce-e93e726c8801)
<img width="2752" height="1722" alt="image" src="https://github.com/user-attachments/assets/25367bae-77c6-4f48-b687-edcfa20a2381" />






## 功能迭代备忘（预防老年痴呆专用）
- 按最长边等比例调整图像大小
- 按某个边按自选比例（最大限度）裁切图像
- 字符串串联
- 整数计算
- 图像联结（第三张非必选）
- 图像去色节点
- 人脸检测和裁切节点（输出mask和裁切区域，未检测到输出原图，检测到多个输出最大的脸）
- 拼图工具拼角色卡用
- 拼图+动态角色选择
- 字符串的搜索替换
- 无缝拼接图像（模糊融合）
- kontext_pro/max（审核可以调成6）
- 从工作流json解析变量
- 从url加载图片
- 从url加载json
- 新增xlsx文件转json的脚本
- 从批次ID读取json
- 双角色动态判断系统
- 文本转图像就是出一个黑底白字的图
- 拼图工具增强版增加对于kontext输出的RGBA的支持
- 小说批次验证（通过批次ID截取json）
- RGBA转RGB
- 保存txt到output（/\都支持但是必须输出路径展示为文本才可以执行到这个环节）
- 图像转像素化的无聊功能
- 动态提示词支持多级嵌套（备注下用法{|/\},嗯我必能想起来咋用）
- 边缘纯色扩充（单方向扩充H,双方向从中心扩充H/2）
- 亮度检测
- 亮度补偿
- PIP 自定义扩图（mask）
- 新增了字节的Seed-X-Instruct-7B，模型去huggingface下，部分配置文件有调整，速度挺快的，支持多语言
- PIP 性别判断动态起手式（男/女）



## 记一些蛇皮问题

--force-reinstall（强制安装--适用于python环境复杂的情况）

.\python_embeded\python.exe -m pip install "C:\NMDtorch\torch-2.7.1+cu128-cp312-cp312-win_amd64.whl" --force-reinstall

.\python_embeded\python.exe -m pip install "C:\NMDtorch\torchvision-0.22.1+cu128-cp312-cp312-win_amd64.whl" --force-reinstall --no-deps

.\python_embeded\python.exe -m pip install "C:\NMDtorch\torchaudio-2.7.1+cu128-cp312-cp312-win_amd64.whl" --force-reinstall

.\python_embeded\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
