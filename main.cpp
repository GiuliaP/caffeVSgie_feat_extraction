/* Example usage of the class CaffeExtractor */

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

// std::system includes
#include <stdio.h>
#include <stdlib.h> // getenv
#include <iostream>
#include <fstream>

#include <string>
#include <deque>
#include <algorithm>
#include <vector>
#include <memory>
#include <algorithm>

// OpenCV includes
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

// Caffe class includes
#include "CaffeFeatExtractor.hpp"

using namespace std;

#define CAFFE_ROOT_ENV				"Caffe_ROOT"

int *pArgc = NULL;
char **pArgv = NULL;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int main(int argc, char **argv)
{

	////////////////////////////////////////////////////////////////////////////////
	// CUDA Setup
	////////////////////////////////////////////////////////////////////////////////

	pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log

    }

    ////////////////////////////////////////////////////////////////////////////////
    // Caffe Initialization
    ////////////////////////////////////////////////////////////////////////////////

    // Caffe environment variable
    string caffe_ROOT = string( getenv(CAFFE_ROOT_ENV) );

    // Caffe class declaration
    CaffeFeatExtractor<float> *caffe_extractor;

    // Binary file (.caffemodel) containing the pretrained network's weights
    vector <string> pretrained_binary_proto_file;
    pretrained_binary_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/20150909-152343-ea39_Caffenet_day1_epoch_30.caffemodel");
    pretrained_binary_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/20150909-153400-d3a0_CaffeNet_day2_epoch_30.caffemodel");
    pretrained_binary_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/20150909-154827-86ad_CaffeNet_day3_epoch_30.caffemodel");
    pretrained_binary_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/20150909-183909-393c_CaffeNet_day4_epoch_30.caffemodel");

    // Text file (.prototxt) defining the network structure
    vector <string> feature_extraction_proto_file;
    feature_extraction_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/CaffeNet_day1_train_val.prototxt");
    feature_extraction_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/CaffeNet_day2_train_val.prototxt");
    feature_extraction_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/CaffeNet_day3_train_val.prototxt");
    feature_extraction_proto_file.push_back("/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/models/CaffeNet_day4_train_val.prototxt");

    // Names of layers to be extracted
    string extract_features_blob_names = "prob";
    int num_features = 1;

    // GPU or CPU mode
    string compute_mode = "GPU";
    // If compute_mode="GPU", must specify device ID
    int device_id = 0;

    bool timing = true;

    int batch_size = 50;
    int batch_size_caffe = 50;

    string image_rootdir = "/data/giulia/DATASETS/iCubWorld28_jpg/test/";
    string registry_dir = "/data/giulia/DATASETS/iCubWorld28_digit_registries/";
    //string extension = ".ppm";

    string out_rootdir = "/data/giulia/DATASETS/iCubWorld28_experiments/CaffeNet_finetuned/features/";

    for (int m=0; m<4; m++) {

    	// class instantiation

    	caffe_extractor = new CaffeFeatExtractor<float>(pretrained_binary_proto_file[m],
    	    		feature_extraction_proto_file[m],
    	    		extract_features_blob_names,
    	    		compute_mode,
    	    		device_id,
    	    		timing);

    	for (int d=0; d<4; d++)  {

    		// image source preparation

    		string image_dir = image_rootdir + "day" + patch::to_string(d+1);
    		string registry_file = registry_dir + "TEday" + patch::to_string(d+1) + ".txt";

    		vector<string> registry;
    		ifstream infile;
    		string line, label;
    		infile.open (registry_file.c_str());
    		infile >> line;
    		infile >> label;
    		while(!infile.eof())
    		{
    			registry.push_back(line);
    		    infile >> line;
    		    infile >> label;
    		}
    		infile.close();

    	    int num_images = registry.size();
    		cout << num_images << endl;

    		// batch size update

    		int last_batch = num_images%batch_size;
    		int num_mini_batches = num_images/batch_size;
    		cout << num_mini_batches << " " << last_batch << endl;


    		// feature extraction

    		string out_dir = out_rootdir + "TRday" + patch::to_string(m+1) + "TEday" + patch::to_string(d+1);

    		ofstream outfile;
    		string out_filename;

    		for (int batch_index = 0; batch_index < num_mini_batches; batch_index++) {

    			vector< vector<float> > features;
    		    vector<cv::Mat> images;

    		    for (int i=0; i<batch_size; i++) {

    		    	string image_path = image_dir + "/" + registry[batch_index*batch_size + i];

    		    	cv::Mat img = cv::imread(image_path);
    		    	images.push_back(img);
    		    }

    		    caffe_extractor->extractBatch_singleFeat_1D(images, batch_size_caffe, features);

    		    for (int i=0; i<batch_size; i++) {

    		    	string imgname = registry[batch_index*batch_size + i];
    		    	out_filename = out_dir + "/" + imgname.substr(0, imgname.size()-4) + ".txt";
    		    	//cout << out_filename << endl;

    		    	outfile.open (out_filename.c_str());
    		    	for (int j=0; j<features[i].size(); j++)
    		    		outfile << features[i][j] << endl;
    		    	outfile.close();
    		    }

    		    features.clear();
    		    images.clear();
    		}

    		// last batch

    		vector< vector<float> > features;
    		vector<cv::Mat> images;

    		for (int i=0; i<last_batch; i++) {
    			string image_path = image_dir + "/" + registry[num_mini_batches*batch_size + i];
    			cv::Mat img = cv::imread(image_path);
    		    images.push_back(img);
    		}

    		if (last_batch>0)
    			caffe_extractor->extractBatch_singleFeat_1D(images, last_batch, features);

    		for (int i=0; i<last_batch; i++) {
    			string imgname = registry[num_mini_batches*batch_size + i];
    			out_filename = out_dir + "/" + imgname.substr(0, imgname.size()-4) + ".txt";
    			//cout << out_filename << endl;

    			outfile.open (out_filename.c_str());
    			for (int j=0; j<features[i].size(); j++)
    				outfile << features[i][j] << endl;
    			outfile.close();
    		}

    		features.clear();
    		images.clear();

    		cout<< m << d << endl;
    	}

    	// delete class

    	delete caffe_extractor;

    }

    // CUDA cleanup

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cout << "done!" << endl;

    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
