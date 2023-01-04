#include "utils.h"
#include <string>
#include <fstream>
#include <iostream>

#include "cuda_runtime_api.h"

#include <opencv2//opencv.hpp>
#include <opencv2/core/cuda.hpp>
using namespace std;
using namespace cv;

int Utils::GetCurrentCUDADevice() {
	int curDev = -1;
	CHECK(cudaGetDevice(&curDev));
	return curDev;
}

struct cudaDeviceProp* Utils::GetCurrentCUDADeviceProperties() {
	int numDevs = 0;
	cudaGetDeviceCount(&numDevs);
	int currentComputeDevice = GetCurrentCUDADevice();
	cudaDeviceProp prop;
	//cudaGetDeviceProperties(&devProp, currentComputeDevice);
	CHECK(cudaGetDeviceProperties(&prop, currentComputeDevice));
	printf("Device id:                     %d\n", currentComputeDevice);
	printf("Major revision number:         %d\n", prop.major);
	printf("Minor revision number:         %d\n", prop.minor);
	printf("Name:                          %s\n", prop.name);
	printf("Total global memory:           %lu\n", (unsigned long)prop.totalGlobalMem);
	printf("Total shared memory per block: %lu\n", (unsigned long)prop.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", prop.regsPerBlock);
	printf("Warp size:                     %d\n", prop.warpSize);
	printf("Maximum memory pitch:          %lu\n", (unsigned long)prop.memPitch);
	printf("Maximum threads per block:     %d\n", prop.maxThreadsPerBlock);
	printf("Maximum dimension of block:    %d, %d, %d\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("Maximum dimension of grid:     %d, %d, %d\n",
		prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("Clock rate:                    %d\n", prop.clockRate);
	printf("Total constant memory:         %lu\n", (unsigned long)prop.totalConstMem);
	printf("Texture alignment:             %lu\n", (unsigned long)prop.textureAlignment);
	printf("Concurrent copy and execution: %s\n",
		(prop.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", prop.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n",
		(prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return &prop;

}

std::vector<std::string> Utils::LoadNames(const std::string& path = "") {
	// load class names
	std::vector<std::string> class_names;
	//std::string Path = "SS";
	std::ifstream infile(path);
	if (infile.is_open()) {

		std::string line;
		while (std::getline(infile, line)) {

			class_names.emplace_back(line);
		}
		infile.close();
	}
	else {
		std::cerr << "Error loading the class names!\n";
	}
	return class_names;
}

std::string Utils::Timer(bool start)
{
	static std::chrono::high_resolution_clock::time_point t0;
	
	std::string result = "";
	if (start)
	{
		t0 = std::chrono::high_resolution_clock::now();		
	}
	else
	{ //stop
		auto t1 = std::chrono::high_resolution_clock::now();
		auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);		
		result = to_string(int_ms.count());
	}
	return result; // ms
}

Mat Utils::static_resize(cv::Mat& img, int input_w, int input_h) {
	float r = min(input_w / (img.cols * 1.0), input_h / (img.rows * 1.0));
	int unpad_w = r * img.cols;
	int unpad_h = r * img.rows;
	Mat re(unpad_h, unpad_w, CV_8UC3);	
	resize(img, re, re.size());
	Mat out(input_w, input_h, CV_8UC3, Scalar(114, 114, 114));
	re.copyTo(out(Rect(0, 0, re.cols, re.rows)));	
	return out;
}

float* Utils::blobFromImage(Mat& img) {
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	float* blob = new float[img.total() * 3];
	int channels = 3;
	int img_h = img.rows;
	int img_w = img.cols;
	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < img_h; h++)
		{
			for (size_t w = 0; w < img_w; w++)
			{
				blob[c * img_w * img_h + h * img_w + w] = (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
			}
		}
	}
	return blob;
}

void Utils::qsort_descent_inplace_(std::vector<Object>& faceobjects, int left, int right) {
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (faceobjects[i].prob > p)
			i++;

		while (faceobjects[j].prob < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(faceobjects[i], faceobjects[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j) qsort_descent_inplace_(faceobjects, left, j);
		}
#pragma omp section
		{
			if (i < right) qsort_descent_inplace_(faceobjects, i, right);
		}
	}
}

void Utils::qsort_descent_inplace(std::vector<Object>& objects){
	if (objects.empty())
		return;

	qsort_descent_inplace_(objects, 0, objects.size() - 1);
}

static inline float intersection_area(const Object& a, const Object& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

void Utils::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) {
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = faceobjects[picked[j]];

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

void Utils::generate_yolo_proposals(float* feat_blob, int output_size, float prob_threshold, std::vector<Object>& objects) {
	const int num_class = 80;
	auto dets = output_size / (num_class + 5);
	for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
	{
		const int basic_pos = boxs_idx * (num_class + 5);
		float x_center = feat_blob[basic_pos + 0];
		float y_center = feat_blob[basic_pos + 1];
		float w = feat_blob[basic_pos + 2];
		float h = feat_blob[basic_pos + 3];
		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;
		float box_objectness = feat_blob[basic_pos + 4];
		// std::cout<<*feat_blob<<std::endl;
		for (int class_idx = 0; class_idx < num_class; class_idx++)
		{
			float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
			float box_prob = box_objectness * box_cls_score;
			if (box_prob > prob_threshold)
			{
				Object obj;
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = w;
				obj.rect.height = h;
				obj.label = class_idx;
				obj.prob = box_prob;

				objects.push_back(obj);
			}

		} // class loop
	}

}


