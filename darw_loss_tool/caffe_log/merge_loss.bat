python merge_logs.py ssd_up_first.log ssd_up_second.log  ssd_up_third.log ssd_up_fourth.log
ren total_log.txt ssd.txt
python merge_logs.py dsod_first.log dsod_second.log dsod_third.log
ren total_log.txt dsod.txt
python visualize_produced_loss.py ssd.txt dsod.txt
