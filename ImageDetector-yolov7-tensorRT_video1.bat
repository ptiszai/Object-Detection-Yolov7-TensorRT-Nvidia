cd ./x64/Release
ImageDetector-yolov7-tensorRT.exe -model=../../models/yolov7-tiny-norm.trt -class_names=../../models/coco_classes.txt -video=../../images/cars.mp4 -wr=1
pause