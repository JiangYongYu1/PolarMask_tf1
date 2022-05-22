# Polar Mask tensorflow 1.14版本

### 工程代码讲解
主干实现：resnet18 resnet34 resnet50
        
        具体实现：详见 resnet_18_34.py, resnet_50.py, 里面的每个函数的具体功能都有注释
    
模型整体实现：
        
        具体实现：详见 model.py, 里面的每个函数的具体功能都有注释

loss函数：
        
        具体实现：详见 loss_ad.py，里面的每个函数的具体功能都有注释

真实数据label生成部分

        具体实现：详见 data_generator_for_polar_mask.py，里面的每个函数的具体功能都有注释

### 模型配置
主要配置文件：
        
        config.py 




### 模型训练
单GPU训练
    
        python train.py --gpu_list 0 
        参数都可以在flags里面配置

多GPU训练

        python multi_gpu_train.py
        参数也都可以配置

### 模型验证
        python eval_polar_coco_mmcv_centerness.py
### 模型测试
        python test_polar.py
        
### 模型冻结
第一步：
        
      运行python test_polar.py得到简化且加上滑动平均后的模型

第二步：

      运行 ckpt_show.py 得到tensorboard log 可以看到模型结构图，从中找到输出节点名

第三步：
        
      修改ckpt_to_pb.py里面的路径，生成pb模型

第四步：
     
      运行 test_polar_pb.py
