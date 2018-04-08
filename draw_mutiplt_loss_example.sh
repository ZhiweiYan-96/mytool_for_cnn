cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_0407_alltrain_300x300/VGG_VOC0712_0406_alltrain_SSD_up_lessparam_fusion_0407_alltrain_300x300.log  /home/yanzhiwei/mytool_for_cnn/0407.log

cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_300x300/log1.log  /home/yanzhiwei/mytool_for_cnn/ssd_up_fusion.log
cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_300x300/VGG_VOC0712_SSD_up_lessparam_fusion_300x300.log  /home/yanzhiwei/mytool_for_cnn/ssd_up_fusion1.log 

cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_0406_alltrain_300x300/log1.log  /home/yanzhiwei/mytool_for_cnn/0406.log
cp /home/yanzhiwei/caffe/jobs/VGGNet/VOC0712/SSD_up_lessparam_fusion_0406_alltrain_300x300/VGG_VOC0712_0406_alltrain_SSD_up_lessparam_fusion_0406_alltrain_300x300.log  /home/yanzhiwei/mytool_for_cnn/0406_1.log

python merge_logs.py ssd_up_fusion.log ssd_up_fusion1.log
mv total_log.txt ssd_up_fusion.log

python merge_logs.py 0406.log  0406_1.log
mv total_log.txt 0406.log

python merge_logs.py 0407.log
mv total_log.txt 0407.log

python visualize_multiple_produced_loss.py ssd_up_fusion.log 0406.log 0407.log