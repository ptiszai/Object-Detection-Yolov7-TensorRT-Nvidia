cd ./x64/Release
ImageDetector-yolov7-tensorRT.exe -model=../../models/yolov7-tiny-norm.trt -class_names=../../models/coco_classes.txt -image=../../images/bus.jpg -wr=1
pause