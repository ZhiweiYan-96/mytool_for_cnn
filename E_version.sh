cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_300x300/log1.log  /home/yanzhiwei/mytool_for_cnn/log1.log
 cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_300x300/VGG_VOC0712_SSD_up_lessparam_fusion_300x300.log  /home/yanzhiwei/mytool_for_cnn/fusion.log



python merge_logs.py log1.log fusion.log
mv total_log.txt ssd_up_fusion.log




python visualize_multiple_produced_loss.py ssd_up_fusion.log 