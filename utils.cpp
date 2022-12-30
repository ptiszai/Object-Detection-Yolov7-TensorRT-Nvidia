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
	CUDA_CHECK(cudaGetDevice(&curDev));
	return curDev;
}

struct cudaDeviceProp* Utils::GetCurrentCUDADeviceProperties() {
	int numDevs = 0;
	cudaGetDeviceCount(&numDevs);
	int currentComputeDevice = GetCurrentCUDADevice();
	cudaDeviceProp prop;
	//cudaGetDeviceProperties(&devProp, currentComputeDevice);
	CUDA_CHECK(cudaGetDeviceProperties(&prop, currentComputeDevice));
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
/*bool Utils::IsCUDA()
{
	int gpucount = cuda::getCudaEnabledDeviceCount();
	if (gpucount != 0) {
		cout << "no. of gpu = " << gpucount << endl;
	}
	else
	{
		cout << "There is no CUDA supported GPU" << endl;
		return false;

	}
	cuda::DeviceInfo deviceinfo;
	int id = deviceinfo.cuda::DeviceInfo::deviceID();
	cuda::setDevice(id);
	cuda::resetDevice();
	//enum cuda::FeatureSet arch_avail;
	//if (cuda::TargetArchs::builtWith(arch_avail))
	//	cout << "yes, this Gpu arch is supported" << endl;

	//cuda::DeviceInfo deviceinfo;
	cout << "GPU: " << deviceinfo.cuda::DeviceInfo::name() << endl;
	return true;
}*/

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