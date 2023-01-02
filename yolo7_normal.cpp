#include"yolo7_normal.h"
#include <fstream>
#include <iostream>
#include <string>

#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"
#include "NvInferPlugin.h"
#include "logging.h"

//https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#unique_383286676

using namespace std;
using namespace cv;

using namespace nvinfer1;
static Logger gLogger;

Yolo7_normal::Yolo7_normal(std::string engine_file_path)
{
    size_t size{ 0 };
    char* trtModelStream{ nullptr };

    std::ifstream file(engine_file_path, std::ios::in | std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    std::cout << "engine init finished" << std::endl;
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); // https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/20
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    for (int j = 0; j < out_dims.nbDims; j++) {
        this->output_size *= out_dims.d[j];
    }
    this->prob = new float[this->output_size];
}


Yolo7_normal::~Yolo7_normal()
{
    std::cout << "yolo destroy" << std::endl;
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();

}