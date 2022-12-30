#pragma once
#include <string>
#include <vector>

#define CUDA_CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

class Utils {
public:
    Utils() {}
    ~Utils() {}
    std::vector<std::string> LoadNames(const std::string& path);
    int GetCurrentCUDADevice();
    struct cudaDeviceProp* GetCurrentCUDADeviceProperties();
    //bool IsCUDA();
    std::string Timer(bool start);
private:
};
