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
    string pretrained_binary_proto_file;
    //pretrained_binary_proto_file = "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    pretrained_binary_proto_file = "/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

    cout << "Using pretrained network: " << pretrained_binary_proto_file << endl;

    // Text file (.prototxt) defining the network structure
    //string feature_extraction_proto_file = "/home/giulia/REPOS/caffe_batch/imagenet_val.prototxt";
    string feature_extraction_proto_file = "/data/giulia/REPOS/caffe_batch_finetuning/imagenet_val_new.prototxt";

    cout << "Using network defined in: " << feature_extraction_proto_file << endl;

    // Names of layers to be extracted
    string extract_features_blob_names = "fc6,fc7,prob";
    int num_features = 3;

    // GPU or CPU mode
    string compute_mode = "GPU";
    // If compute_mode="GPU", must specify device ID
    int device_id = 0;

    bool timing = true;

    // Caffe class instantiation
    caffe_extractor = NULL;
    caffe_extractor = new CaffeFeatExtractor<float>(pretrained_binary_proto_file,
        		   feature_extraction_proto_file,
        		   extract_features_blob_names,
        		   compute_mode,
        		   device_id,
        		   timing);

    ////////////////////////////////////////////////////////////////////////////////
    // Registry preparation
    ////////////////////////////////////////////////////////////////////////////////

    string root_dir = "/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_centroid256_disp_finaltree";

    string registry_file = "/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/flower.txt";

    string extension = ".jpg";

    vector<string> registry;

    ifstream infile;
    string line;
    infile.open (registry_file.c_str());
    getline(infile,line);
    cout << line << endl;
    while(!infile.eof())
    {
    	registry.push_back(line);
    	getline(infile,line);
    	//cout << line << endl;
    }
    infile.close();


    int num_images = registry.size();

    cout << num_images << endl;

    ////////////////////////////////////////////////////////////////////////////////
    // Feature extraction
    ////////////////////////////////////////////////////////////////////////////////

    int batch_size = 512;
    int batch_size_caffe = 512;

    if (num_images%batch_size!=0)
    {
    	batch_size = 1;
    	cout << "WARNING main: image number is not multiple of batch size, setting it to 1 (low performance)" << endl;
    }
    int num_mini_batches = num_images/batch_size;

    //vector< Blob<float>* > features;
    vector< vector<float> > features;
    //vector<float> features;

    string out_dir1 = "/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_centroid256_disp_finaltree_experiments/test_offtheshelfnets/scores/caffenet/fc6";
    string out_dir2 = "/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_centroid256_disp_finaltree_experiments/test_offtheshelfnets/scores/caffenet/fc6";
    string out_dir3 = "/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_centroid256_disp_finaltree_experiments/test_offtheshelfnets/scores/caffenet/fc6";

    ofstream outfile1, outfile2, outfile3;
    string out_filename1, out_filename2, out_filename3;

    for (int batch_index = 0; batch_index < num_mini_batches; batch_index++)
    {

    	vector<cv::Mat> images;
    	for (int i=0; i<batch_size; i++)
    	{
    	    string image_path = root_dir + "/" + registry[batch_index*batch_size + i] + extension;
    	    cv::Mat img = cv::imread(image_path);

    	    /*
    	    cv::namedWindow( "image", cv::WINDOW_AUTOSIZE);
    	    cv::imshow( "image", images[i] );
    	    cv::waitKey(0);
    	    */

    	    /*
    	    const int img_height = img.rows;
    	    const int img_width = img.cols;
    	    const int crop_size = 227;

    	    int h_off = 0;
    	    int w_off = 0;
    	    h_off = (img_height - crop_size) / 2;
    	    w_off = (img_width - crop_size) / 2;

    	    cout << h_off << " " << w_off << endl;
    	    */

    	    //cvtColor(img, img, CV_BGR2RGB);
    	    images.push_back(img);
    	}

    	caffe_extractor->extractBatch_multipleFeat_1D(images, batch_size_caffe, features);

    	/*for (int i=0; i<images.size(); i++)
    	{
    		features.push_back(vector<float>());
    		caffe_extractor->extract_singleFeat_1D(images[i], features[i]);
    	}*/

    	for (int i=0; i<batch_size; i++)
    	{
    		out_filename1 = out_dir1 + "/" + registry[batch_index*batch_size + i] + ".txt";
    		out_filename2 = out_dir2 + "/" + registry[batch_index*batch_size + i] + ".txt";
    		out_filename3 = out_dir3 + "/" + registry[batch_index*batch_size + i] + ".txt";

    		outfile1.open (out_filename1.c_str());
    		outfile2.open (out_filename2.c_str());
    		outfile3.open (out_filename3.c_str());

    		for (int j=0; j<features[i].size(); j++)
    		{
    			outfile1 << features[i][j] << endl;
    		}
    		for (int j=0; j<features[i+batch_size].size(); j++)
    		{
    			outfile2 << features[i+batch_size][j] << endl;
    		}
    		for (int j=0; j<features[i+2*batch_size].size(); j++)
    		{
    			outfile3 << features[i+2*batch_size][j] << endl;
    			//cout <<  features[i+2*batch_size][j] << " ";
    		}
    		//cout << endl;

    		outfile1.close();
    		outfile2.close();
    		outfile3.close();
    	}


    	features.clear();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // CUDA Cleanup
    ////////////////////////////////////////////////////////////////////////////////

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    exit(EXIT_SUCCESS);
}
