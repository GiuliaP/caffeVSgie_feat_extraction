#ifndef CAFFEFEATEXTRACTOR_H_
#define CAFFEFEATEXTRACTOR_H_

#include <string>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

// Boost
#include "boost/algorithm/string.hpp"
#include "boost/make_shared.hpp"

// Caffe
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"

using namespace std;
using namespace caffe;

template<class Dtype>
class CaffeFeatExtractor {

    string caffemodel_file;
    string prototxt_file;

    caffe::shared_ptr<Net<Dtype> > feature_extraction_net;

    int mean_width;
    int mean_height;
    int mean_channels;

    string blob_name;

    bool gpu_mode;
    int device_id;

public:

    bool timing;

    CaffeFeatExtractor(string _caffemodel_file,
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_names,
            string _compute_mode,
            int _device_id,
            bool _timing_extraction);

    bool extract_singleFeat_1D(cv::Mat &image, vector<Dtype> &features, float (&times)[2]);
};

template <class Dtype>
CaffeFeatExtractor<Dtype>::CaffeFeatExtractor(string _caffemodel_file,
        string _prototxt_file, int _resizeWidth, int _resizeHeight,
        string _blob_name,
        string _compute_mode,
        int _device_id,
        bool _timing) {

    // Setup the GPU or the CPU mode for Caffe
    if (strcmp(_compute_mode.c_str(), "GPU") == 0 || strcmp(_compute_mode.c_str(), "gpu") == 0) {

        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): using GPU" << std::endl;

        gpu_mode = true;
        device_id = _device_id;

        Caffe::CheckDevice(device_id);

        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): using device_id = " << device_id << std::endl;

        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);

        // Optional: to check that the GPU is working properly...
        Caffe::DeviceQuery();

    } else
    {
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): using CPU" << std::endl;

        gpu_mode = false;
        device_id = -1;

        Caffe::set_mode(Caffe::CPU);
    }

    // Assign specified .caffemodel and .prototxt files
    caffemodel_file = _caffemodel_file;
    prototxt_file = _prototxt_file;

    // Network creation using the specified .prototxt
    feature_extraction_net = boost::make_shared<Net<Dtype> > (prototxt_file, caffe::TEST);
       // Network initialization using the specified .caffemodel
    feature_extraction_net->CopyTrainedLayersFrom(caffemodel_file);

    // Mean image initialization
    
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    TransformationParameter tp = memory_data_layer->layer_param().transform_param();

    if (tp.has_mean_file())
    {
        const string& mean_file = tp.mean_file();
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): loading mean file from " << mean_file << std::endl;

        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        Blob<Dtype> data_mean;
        data_mean.FromProto(blob_proto);

        mean_channels = data_mean.channels();
        mean_width = data_mean.width();
        mean_height = data_mean.height();

    } else if (tp.mean_value_size()>0)
    {

        const int b = tp.mean_value(0);
        const int g = tp.mean_value(1);
        const int r = tp.mean_value(2);

        mean_channels = tp.mean_value_size();
        mean_width = _resizeWidth;
        mean_height = _resizeHeight;

        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): B " << b << "   G " << g << "   R " << r << std::endl;
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): resizing anysotropically to " << " W: " << mean_width << " H: " << mean_height << std::endl;
    }
    else
    {
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): Error: neither mean file nor mean value in prototxt!" << std::endl;
    }

    // Initialize timing flag
    timing = _timing;
    blob_name = _blob_name;

}

template<class Dtype>
bool CaffeFeatExtractor<Dtype>::extract_singleFeat_1D(cv::Mat &image, vector<Dtype> &features, float (&times)[2])
{

    // Check input image
    if (image.empty())
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): empty imMat!" << std::endl;
        return false;
    }

    times[0] = 0.0f;
    times[1] = 0.0f;

    // Start timing
    cudaEvent_t startPrep, stopPrep, startNet, stopNet;
    if (timing)
    {
        cudaEventCreate(&startPrep);
        cudaEventCreate(&startNet);
        cudaEventCreate(&stopPrep);
        cudaEventCreate(&stopNet);
        cudaEventRecord(startPrep, NULL);
        cudaEventRecord(startNet, NULL);
    }

    // Prepare Caffe

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    // Initialize labels to zero
    int label = 0;

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1
    if (memory_data_layer->batch_size()!=1)
    {
        memory_data_layer->set_batch_size(1);
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): BATCH SIZE = " << memory_data_layer->batch_size() << std::endl;
    }

    // Image preprocessing
    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied
    if (image.rows != mean_height || image.cols != mean_height)
    {
        if (image.rows > mean_height || image.cols > mean_height)
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
        }
        else
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
        }
    }

    memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopPrep, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopPrep);

        cudaEventElapsedTime(times, startPrep, stopPrep);
    }


    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results = feature_extraction_net->Forward();

    const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_name);

    int batch_size = feature_blob->num(); // should be 1
    if (batch_size!=1)
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): Error! Retrieved more than one feature, exiting..." << std::endl;
        return -1;
    }

    int feat_dim = feature_blob->count(); // should be equal to: count/num=channels*width*height
    if (feat_dim!=feature_blob->channels())
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, height, width)" << std::endl;
    }

    features.insert(features.end(), feature_blob->mutable_cpu_data() + feature_blob->offset(0), feature_blob->mutable_cpu_data() + feature_blob->offset(0) + feat_dim);

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopNet, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopNet);

        cudaEventElapsedTime(times+1, startNet, stopNet);

    }

    return true;

}

#endif /* CAFFEFEATEXTRACTOR_H_ */
