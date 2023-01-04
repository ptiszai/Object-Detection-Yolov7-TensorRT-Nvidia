#include"yolo7_normal.h"
#include <fstream>
#include <string>
#include <vector>

#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "logging.h"

//https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#unique_383286676

using namespace std;
using namespace cv;

using namespace nvinfer1;
static Logger gLogger;

Yolo7_normal::Yolo7_normal(string& class_path, Utils* utils_a) {
    utils = utils_a;
    class_names = utils->LoadNames(class_path);//read classes
    return;
}

bool Yolo7_normal::readModel(string& engine_file_path) {
    if (class_names.empty()) {
        cerr << "ERROR:class_names is failed:" << endl;
        return false;
    }
    size_t size{ 0 };
    char* trtModelStream{ nullptr };

    try {
        ifstream file(engine_file_path, ios::in | ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
       // std::cout << "engine init finished" << std::endl;
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
        runtime = createInferRuntime(gLogger);       
        if (runtime == nullptr) {
            throw "createInferRuntime()";
        }
        engine = runtime->deserializeCudaEngine(trtModelStream, size);
        // https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/20
        if (engine == nullptr) {
            throw "deserializeCudaEngine()";
        }
        context = engine->createExecutionContext();        
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

vector<Object> Yolo7_normal::detect_img(Mat& img) {     
        int img_w = img.cols;
        int img_h = img.rows;
        //cout << "detect_img" << endl;
        Mat pr_img = utils->static_resize(img, this->INPUT_W, this->INPUT_H);

        float* blob = NULL;
        blob = utils->blobFromImage(pr_img);
        float scale = std::min(this->INPUT_W / (img.cols * 1.0), this->INPUT_H / (img.rows * 1.0));

        // run inference
        auto start = chrono::system_clock::now();
        doInference(*context, blob, this->prob, output_size, pr_img.size());
        auto end = chrono::system_clock::now();
        std::cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::vector<Object> objects;
        decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h);    
        delete blob; 
        return objects;
}

void Yolo7_normal::decode_outputs(float* prob, int output_size, vector<Object>& objects, float scale, const int img_w, const int img_h) {
    vector<Object> proposals;
    utils->generate_yolo_proposals(prob, output_size, BBOX_CONF_THRESH, proposals);
    //cout << "num of boxes before nms: " << proposals.size() << endl;

    utils->qsort_descent_inplace(proposals);

    vector<int> picked;
    utils->nms_sorted_bboxes(proposals, picked, NMS_THRESH);

    int count = picked.size();

    //cout << "num of boxes: " << count << endl;

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
    try {
        const ICudaEngine& engine = context.getEngine();
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        //assert(engine.getNbBindings() == 2);
        if (engine.getNbBindings() != 2) {
            throw "engine.getNbBindings()";
        }
        void* buffers[2];
       /* int number = engine.getNbBindings();
        for (int ii = 0; ii < number; ii++) {
            cout << engine.getBindingName(ii) << endl;
        }*/
        //char const* b1 = engine.getBindingName(0);
        //auto in_dims = engine.getBindingDimensions(engine.getBindingIndex("image_arrays"));
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

        //assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
        if (engine.getBindingDataType(inputIndex) != nvinfer1::DataType::kFLOAT) {
            throw "engine.getBindingDataType 1";
        }

        const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
        //assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
        if (engine.getBindingDataType(outputIndex) != nvinfer1::DataType::kFLOAT) {
            throw "engine.getBindingDataType 2";
        }
        int mBatchSize = engine.getMaxBatchSize();

        // Create GPU buffers on device
        //CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
        if (cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)) != 0) {
            throw "cudaMalloc() 1";
        }

        CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));
        if (cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)) != 0) {
            throw "cudaMalloc() 2";
        }

        // Create stream
        cudaStream_t stream;
        //CHECK(cudaStreamCreate(&stream));
        if (cudaStreamCreate(&stream) != 0) {
            throw "cudaStreamCreat()";
        }

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        //CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
        if (cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream) != 0) {
            throw "cudaMemcpyAsync()";
        }
        context.enqueue(1, buffers, stream, nullptr);
        //CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        if (cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream) != 0) {
            throw "cudaMemcpyAsync()";
        }
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

Mat Yolo7_normal::drawPreds(const Mat& bgr, const vector<Object>& objects) {
    Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5) {
            txt_color = cv::Scalar(0, 0, 0);
        }
        else {
            txt_color = cv::Scalar(255, 255, 255);
        }

        rectangle(image, obj.rect, color * 255, 2);
       
        char buffer[20];  // maximum expected length of the float        
        snprintf(buffer, 20, "%.2f", obj.prob * 100);
        string text = class_names.at(obj.label) + ":" + string(buffer);

        int baseLine = 0;
        Size label_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.rect.x;
        int y = obj.rect.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        rectangle(image, Rect(Point(x, y), Size(label_size.width, label_size.height + baseLine)),txt_bk_color, -1);
        putText(image, text, Point(x, y + label_size.height),FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }
    return image;
}
