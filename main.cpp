#include <iostream>
#include <filesystem>
#include <thread>
#include <semaphore>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include"yolo7_normal.h"
#include "QueueFPS.hpp"

//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

using namespace std;
using namespace cv;

const string WinName = "Deep learning object detection in Nvidia";
const char* keys =
{
	"{help h ?| | show help message}{model|| <x.rt> model_trt.engine}{class_names|coco_classes.txt|class names file}{image|| <*.png,jpg,bmp> image file}{video|| <*.mp4>}{path|.| path to file}{wr|0|writing to file}{gpu|1| Default gpu }{end2end|0| Default normal }"
};
bool process = true;

typedef struct ObjectF {
	Mat frame;
	vector<Object> objects;
} ObjectF;
QueueFPS<Mat> framesAndProcesszingQueue;
counting_semaphore<1> frameSignal(0);

//----------------------------------------------------
// Frames capturing thread
thread frameAndProcesszingThread(VideoCapture& cap, Yolo7_normal* yolo7)
{
	cout << "Frames and processzing capturing thread" << endl;
	thread framesAndProcesszingThread([&]() {
		Mat frame;
		Mat fclone;
		//int ee = 0;
		while (process)
		{
			ObjectF objectf;
		//	ee++;
			if (cap.grab()) {
				cap >> frame;
				if (!frame.empty())
				{	
					framesAndProcesszingQueue.push(frame);
					frameSignal.acquire();
					this_thread::sleep_for(5ms);
				}
				else
				{
					cerr << "END or ERROR:video frame empty!" << endl;					
					break;
				}
			}
			else {
				cerr << "ERROR: bad grab" << endl;
				this_thread::sleep_for(100ms);
			}
		}
	});
	return framesAndProcesszingThread;
}

// functions
static void help(int argc, const char** argv)
{

	for (int ii = 1; ii < argc; ii++) {
		cout << argv[ii] << endl;
	}
}

//----------------------------------------------------
// MAIN
//----------------------------------------------------

int main(int argc, const char** argv) {
	/* Examples:
		"ImageDetector-yolov7-tensorRT.exe -h"
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny-norm.trt -class_names=models/coco_classes.txt -image=images/bus.jpg" // normal, not end2end, gpu, only read image
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny.trt -class_names=models/coco_classes.txt -image=images/bus.jpg -end2end=1 -wr=1" // normal, end2end, gpu, created out image bus_e2e.jpg
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny-norm.trt -class_names=models/coco_classes.txt -image=images/bus.jpg  -wr=1" // normal, not end2end, gpu,  created out image bus_norm.jpg
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny-norm.trt -class_names=models/coco_classes.txt -video=images/car.mp4" // normal, not end2end, gpu, only read video
		etc.
	*/
	CommandLineParser parser(argc, argv, keys);
	help(argc, argv);
	string model_name = parser.get<string>("model");
	if (model_name.empty()) {
		parser.printMessage();
		return 1;
	}

	parser.about("Trying commandline parser");	

	if (parser.has("help"))
	{
		parser.printMessage();
		//parser.printErrors();
		return 0;
	}
	string path_name = parser.get<string>("path");
	if (path_name == ".") {
		path_name = filesystem::current_path().string();
	}

	//string model_name = parser.get<string>("model");
	string model_path = path_name + "/" + model_name;
	if (!filesystem::exists(model_path)) {
		cerr << "ERROR: model file not exist" << endl;
		return -1;
	}

	string class_names = parser.get<string>("class_names");
	string classes_path = path_name + "/" + class_names;
	if (!filesystem::exists(classes_path)) {
		cerr << "ERROR: classes file not exist" << endl;
		return -1;
	}

	string image_name = parser.get<string>("image");	
	string video_name = parser.get<string>("video");
	bool wr = (bool)parser.get<int>("wr");
	bool gpu = (bool)parser.get<int>("gpu");
	bool end2end = (bool)parser.get<int>("end2end");
	string img_path = "";
	string video_path = "";
	bool image = false;
	bool mp4 = false;

	Utils utils;

	if (gpu) {
		if (utils.GetCurrentCUDADevice() < 0)
		{
			cerr << "ERROR: not founded GPU or/and CUDA" << endl;
			return -1;
		}
		utils.GetCurrentCUDADeviceProperties();
	}

	if (!image_name.empty()) {
		img_path = path_name + "/" + image_name;
		string ext = filesystem::path(img_path).extension().string();
		if ((ext == ".png") || (ext == ".jpg") || (ext == ".bmp")) {
			image = true;
		}
		else {
			cerr << "ERROR: image ext. is not png or jpg or bmp" << video_name << endl;
			return -1;
		}
		if (!filesystem::exists(img_path)) {
			cerr << "ERROR: image file not exist" << endl;
			return -1;
		}
		cout << "image:" << image_name << endl;
	}
	else 
	if (!video_name.empty()) {
		video_path = path_name + "/" + video_name;
		string ext = std::filesystem::path(video_path).extension().string();
		if (ext == ".mp4") {
			mp4 = true;
		}
		else {
			cerr << "ERROR: video ext. is not mp4" << video_name << endl;
			return -1;
		}
		if (!filesystem::exists(video_path)) {
			cerr << "ERROR: video file not exist" << endl;
			return -1;
		}
		cout << "video:" << video_name << endl;
	}
	else {
		cerr << "ERROR:image name or video name is empty" << endl;
		return -1;
	}

	
	Yolo7_normal* yolo7_normal = NULL;	
	if (!end2end) {
		yolo7_normal = new Yolo7_normal(classes_path, &utils);
		//(Yolo7_normal*)yolov7 = yolo7_normal;
		if (!yolo7_normal->readModel(model_path)) {
			cerr << "ERROR: model read failer:" << model_path << endl;
			return -1;			
		}
	}
	else {
		cout << "Under Developing, exit program !!!!!!!!!!!!!!!!!!!!!" << endl;
		return 1;
	}

	namedWindow(WinName, WINDOW_AUTOSIZE);
	VideoCapture cap;
	VideoWriter video_raw;

	// one image
	if (image) { 
		// Open an image file.
		Mat img = imread(img_path);
		if (img.total() == 0) {
			cerr << "ERRON:image array length == 0" << endl;
			return -1;
		}

		utils.Timer(true);
		if (!end2end) {
			vector<Object> objects = yolo7_normal->detect_img(img);
			if (objects.empty()) {
				cerr << "ERRON:image objects vector length == 0" << endl;
				return -1;
			}
			Mat imag = yolo7_normal->drawPreds(img, objects);
			cout << "Done:" << utils.Timer(false) << "ms" << endl;
			string label = "Inference process time: ";
			label.append(utils.Timer(false));
			label.append(" ms");
			putText(imag, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
			if (wr) {
				string filename = filesystem::path(img_path).stem().string();
				imwrite(filename + "_o.png", imag); // result image to file.
			}
			imshow(WinName, imag);
			waitKey(0);
		}
	}
	else
	// video
	if (mp4) {
		// Open a video file or an image file.		
		cap.open(video_path);
		if (!cap.isOpened()) {
			cerr << "ERRON:Unable to open video" << endl;
			return -1;
		}
		if (wr) {
			int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			string filename = std::filesystem::path(video_path).stem().string();
			if (!video_raw.open(filename + "_o.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 10, Size(frame_width, frame_height))) {
				cerr << "ERRON:VideoWriter opened failed!" << endl;
				return 1;
			}
		}
		thread frmThread = frameAndProcesszingThread(cap, yolo7_normal);
		//std::thread prThread = processingThread(yolov7);

		//Postprocessing and rendering loop
		while (waitKey(1) < 0) {
			if (framesAndProcesszingQueue.empty()) {
				this_thread::sleep_for(20ms);
				continue;
			}			
			Mat frame = framesAndProcesszingQueue.get();
			utils.Timer(true);
			vector<Object> objects = yolo7_normal->detect_img(frame);
			Mat imag = yolo7_normal->drawPreds(frame, objects);	
			frameSignal.release();
			//cout << "Done:" << utils.Timer(false) << "ms" << endl;
			string label = "Inference process time: ";
			label.append(utils.Timer(false));
			label.append(" ms");
			putText(imag, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));							
			if (wr) {
				if (!video_raw.isOpened()) {
					cout << "ERRON:VideoWriter opened failed!" << endl;
					return 1;
				}
				video_raw << imag;
			}
			imshow(WinName, imag);
		}
		frameSignal.release();
		process = false;
		frmThread.join();		
	}
	
	if (cap.isOpened()) {
		cap.release();
	}
	if (video_raw.isOpened()) {
		video_raw.release();
	}	
	return 0;
}

