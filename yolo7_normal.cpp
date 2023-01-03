#include"yolo7_normal.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"
#include "NvInferPlugin.h"
#include "logging.h"

//https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#unique_383286676

using namespace std;
using namespace cv;

using namespace nvinfer1;
static Logger gLogger;

Yolo7_normal::Yolo7_normal(string& class_path) {
    utils = new Utils();
    class_names = utils->LoadNames(class_path);//read classes
}

bool Yolo7_normal::readModel(string& engine_file_path) {
    if (class_names.empty()) {
        cerr << "ERROR:class_names is failed:" << endl;
        return false;
    }
    size_t size{ 0 };
    char* trtModelStream{ nullptr };

    try {
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
        //assert(runtime != nullptr);
        if (runtime == nullptr) {
            throw "createInferRuntime()";
        }
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        //assert(engine != nullptr); // https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/20
        if (engine == nullptr) {
            throw "deserializeCudaEngine()";
        }
        context = engine->createExecutionContext();
        //assert(context != nullptr);
        if (context == nullptr) {
            throw "createExecutionContext()";
        }
        delete[] trtModelStream;
        auto out_dims = engine->getBindingDimensions(1);
        for (int j = 0; j < out_dims.nbDims; j++) {
            this->output_size *= out_dims.d[j];
        }
        this->prob = new float[this->output_size];
    }
    catch (const char* msg) {
        cerr << "ERROR:readModel():"<< msg << std::endl;
        return false;
    }
    return true;
}

Yolo7_normal::~Yolo7_normal() {
    std::cout << "yolo destroy" << std::endl;
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
}

vector<Object> Yolo7_normal::detect_img(cv::Mat& img) {
      //  cv::Mat img = cv::imread(image_path);
        int img_w = img.cols;
        int img_h = img.rows;
        cv::Mat pr_img = utils->static_resize(img, this->INPUT_W, this->INPUT_H);
     //   std::cout << "blob image" << std::endl;

        float* blob;
        blob = utils->blobFromImage(pr_img);
        float scale = std::min(this->INPUT_W / (img.cols * 1.0), this->INPUT_H / (img.rows * 1.0));

        // run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, blob, this->prob, output_size, pr_img.size());
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<Object> objects;
        decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h);
     //   draw_objects(img, objects, image_path);
        delete blob; 
        return objects;
}

void Yolo7_normal::decode_outputs(float* prob, int output_size, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
    std::vector<Object> proposals;
    utils->generate_yolo_proposals(prob, output_size, BBOX_CONF_THRESH, proposals);
    std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

    utils->qsort_descent_inplace(proposals);

    std::vector<int> picked;
    utils->nms_sorted_bboxes(proposals, picked, NMS_THRESH);

    int count = picked.size();

    std::cout << "num of boxes: " << count << std::endl;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

bool Yolo7_normal::doInference(IExecutionContext& context, float* input, float* output, const int output_size, cv::Size input_shape) {
    const ICudaEngine& engine = context.getEngine();
    try {
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbBindings() == 2);
        //if (engine.getNbBindings() != 2) {
        //    throw "engine.getNbBindings()";
        //}
        void* buffers[2];
        int number = engine.getNbBindings();
        for (int ii = 0; ii < number; ii++) {
            cout << engine.getBindingName(ii) << endl;
        }
        //char const* b1 = engine.getBindingName(0);
        //auto in_dims = engine.getBindingDimensions(engine.getBindingIndex("image_arrays"));
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

        assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
        //if (engine.getBindingDataType(inputIndex) != nvinfer1::DataType::kFLOAT) {
        //    throw "engine.getBindingDataType 1";
        //}

        const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
        assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
        //if (engine.getBindingDataType(outputIndex) != nvinfer1::DataType::kFLOAT) {
        //    throw "engine.getBindingDataType 2";
        //}
        int mBatchSize = engine.getMaxBatchSize();

        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
        //if (cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)) != 0) {
         //   throw "cudaMalloc() 1";
        //}

        CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));
        //if (cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)) != 0) {
        //    throw "cudaMalloc() 2";
        //}

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        //if (cudaStreamCreate(&stream) != 0) {
        //    throw "cudaStreamCreat()";
        //}

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
        //if (cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream) != 0) {
        //    throw "cudaMemcpyAsync()";
        //}
        context.enqueue(1, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        //if (cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream) != 0) {
        //    throw "cudaMemcpyAsync()";
        //}
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        //CHECK(cudaFree(buffers[inputIndex]));
        if (cudaFree(buffers[inputIndex]) != 0) {
            throw "cudaFree() 1";
        }
        //CHECK(cudaFree(buffers[outputIndex]));
        if (cudaFree(buffers[outputIndex]) != 0) {
            throw "cudaFree() 2";
        }
    }
    catch (const char* msg) {
        cerr << "ERROR:doInference():" << msg << std::endl;
        return false;
    }
    return true;
}

Mat Yolo7_normal::drawPreds(const cv::Mat& bgr, const std::vector<Object>& objects/*, std::string image_path*/) {
    /*static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };*/

    Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, obj.rect, color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    //cv::imwrite("det_res.jpg", image);
    //fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
    return image;
}
