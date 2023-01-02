#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"

using namespace nvinfer1;

class Yolo7_normal
{
public:
    Yolo7_normal(std::string engine_file_path);
    virtual ~Yolo7_normal();
    void detect_img(std::string image_path);
    void detect_video(std::string video_path);
    cv::Mat static_resize(cv::Mat& img);
    float* blobFromImage(cv::Mat& img);
    void doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape);

private:
    static const int INPUT_W = 640;
    static const int INPUT_H = 640;
    const char* INPUT_BLOB_NAME = "image_arrays";
    const char* OUTPUT_BLOB_NAME = "outputs";
    float* prob;
    int output_size = 1;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;

};