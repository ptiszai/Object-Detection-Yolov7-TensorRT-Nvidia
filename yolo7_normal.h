#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"
#include "utils.h"

using namespace nvinfer1;

class Yolo7_normal {
public:
    Yolo7_normal(std::string& engine_file_path, Utils* utils_a);
    ~Yolo7_normal();
    bool readModel(std::string& engine_file_path);
    std::vector<Object> detect_img(cv::Mat& img);
    void detect_video(std::string& video_path);
    cv::Mat drawPreds(const cv::Mat& bgr, const std::vector<Object>& objects/*, std::string image_path*/);

private:
    Utils* utils;
    std::vector<std::string> class_names;
    static const int INPUT_W = 640;
    static const int INPUT_H = 640;
    const char* INPUT_BLOB_NAME = "images";
    const char* OUTPUT_BLOB_NAME = "output";
    const float NMS_THRESH = 0.45;
    const float BBOX_CONF_THRESH = 0.2;
    float* prob;
    int output_size = 1;
    ICudaEngine* engine;
    IRuntime* runtime;
    IExecutionContext* context;
    bool doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape);
    void decode_outputs(float* prob, int output_size, std::vector<Object>& objects, float scale, const int img_w, const int img_h);
};