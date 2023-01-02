#include <iostream>
#include <filesystem>
//https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
#include <opencv2/opencv.hpp>
#include "utils.h"
#include"yolo7_normal.h"

using namespace std;
using namespace cv;

static const std::string WinName = "Deep learning object detection in Nvidia";
static const char* keys =
{
	"{help h ?| | show help message}{model|| <x.rt> model_trt.engine}{image|| <*.png,jpg,bmp> image file}{video|| <*.mp4>}{path|.| path to file}{wr|0|writing to file}{gpu|1| Default gpu }{end2end|0| Default normal }"
};

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
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny-norm.trt -image=images/bus.jpg" // normal, not end2end, gpu, only read image
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny.trt -image=images/bus.jpg -end2end=1 -wr=1" // normal, end2end, gpu, created out image bus_e2e.jpg
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny-norm.trt -image=images/bus.jpg  -wr=1" // normal, not end2end, gpu,  created out image bus_norm.jpg
		"ImageDetector-yolov7-tensorRT.exe -model=models/yolov7-tiny-norm.trt -video=images/images/cat.mp4" // normal, not end2end, gpu, only read video
		etc.
	*/
	cv::CommandLineParser parser(argc, argv, keys);

	parser.about("Trying commandline parser");
	help(argc, argv);
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

	string model_name = parser.get<string>("model");
	string model_path = path_name + "/" + model_name;
	if (!filesystem::exists(model_path)) {
		cout << "ERROR: model file not exist" << endl;
		return 1;
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
			cout << "not founded GPU or/and CUDA" << endl;
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
			cout << "image ext. is not png or jpg or bmp" << video_name << endl;
			return 1;
		}
		if (!filesystem::exists(img_path)) {
			cout << "ERROR: image file not exist" << endl;
			return 1;
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
			cout << "video ext. is not mp4" << video_name << endl;
			return 1;
		}
		if (!filesystem::exists(video_path)) {
			cout << "ERROR: video file not exist" << endl;
			return 1;
		}
		cout << "video:" << video_name << endl;
	}
	else {
		cout << "ERROR:image name or video name is empty" << endl;
		return 1;
	}

	Yolo7_normal* yolo7_normal = NULL;
	if (!end2end) {
		yolo7_normal = new Yolo7_normal(model_path);

	}
	/*std::vector<std::string> class_names = utils.LoadNames(class_path);//read classes
	if (class_names.empty()) {
		cout << "LoadNames is failed:" << class_path << endl;
		return false;
	}
	if (yolov7.readModel(net, model_path, class_names, config_path, gpu)) {
		cout << "read net ok!" << endl;
		net.setPreferableBackend(backend);
		net.setPreferableTarget(target);
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}
	*/

	namedWindow(WinName, WINDOW_NORMAL);
	VideoCapture cap;
	VideoWriter video_raw;

	// one image
	if (image) { 
		// Open an image file.
		Mat img = imread(img_path);
		utils.Timer(true);

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
			int frame_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
			int frame_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			string filename = std::filesystem::path(video_path).stem().string();
			if (!video_raw.open(filename + "_o.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 10, Size(frame_width, frame_height))) {
				cout << "ERRON:VideoWriter opened failed!" << endl;
				return 1;
			}
		}
	}
	
	cout << "Success" << endl;
	
	if (cap.isOpened()) {
		cap.release();
	}
	if (video_raw.isOpened()) {
		video_raw.release();
	}
	//system("pause");		
	return 0;
}

