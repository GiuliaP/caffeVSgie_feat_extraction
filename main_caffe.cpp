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
	// Initialization
	////////////////////////////////////////////////////////////////////////////////

	CaffeFeatExtractor<float> *caffe_extractor;

	// .caffemodel containing the pretrained network's weights
	vector <string> caffemodel_file;
	caffemodel_file.push_back("/usr/local/src/robot/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");
    caffemodel_file.push_back("/usr/local/src/robot/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel");

	// .prototxt defining the network structure
	vector <string> prototxt_file_caffe;
    prototxt_file_caffe.push_back("/usr/local/src/robot/caffeVSgie_feat_extraction/networks/bvlc_reference_caffenet_val_cutfc6.prototxt");
    prototxt_file_caffe.push_back("/usr/local/src/robot/caffeVSgie_feat_extraction/networks/bvlc_googlenet_val_cutpool5.prototxt");
	
    // mean info
    vector<string> binaryproto_meanfile;
    binaryproto_meanfile.push_back("/usr/local/src/robot/caffe/data/ilsvrc12/imagenet_mean.binaryproto");
    binaryproto_meanfile.push_back("");
    vector<float> meanB;
    meanB.push_back(-1);
    meanB.push_back(104);
    vector<float> meanG;
    meanG.push_back(-1);
    meanG.push_back(117);
    vector<float> meanR;
    meanR.push_back(-1);
    meanR.push_back(123);
 
	// Image dir
	string dset_dir = "images2";
    string image_dir = "/usr/local/src/robot/caffeVSgie_feat_extraction/" + dset_dir;

	// Registries
    string registry_file = "/usr/local/src/robot/caffeVSgie_feat_extraction/registries/images2.txt";

	// Output dirs
	vector <string> out_dir_caffe;
    out_dir_caffe.push_back("/home/ubuntu/giulia/GIEvsCaffe/Caffe/caffenet/images2");
    out_dir_caffe.push_back("/home/ubuntu/giulia/GIEvsCaffe/Caffe/googlenet/images2");

	// Names of layers to be extracted
	vector<string> blob_names_caffe;
    blob_names_caffe.push_back("fc6");
    blob_names_caffe.push_back("pool5/7x7_s1");
	int num_features = 1;

	// GPU or CPU mode
	string compute_mode = "GPU";
	// If compute_mode="GPU", must specify device ID
	int device_id = 0;

	bool timing = true;

	int batch_size = 1;
	int batch_size_caffe = 1;

	for (int m=0; m<caffemodel_file.size(); m++) {

		// declare classes

	    caffe_extractor = new CaffeFeatExtractor<float>(caffemodel_file[m],
				prototxt_file_caffe[m], 256, 256,
				blob_names_caffe[m],
				compute_mode,
				device_id,
				timing);

		// read registry
        
        cout << "here" << endl;

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
		cout << endl << num_images << endl;

		// feature extraction

		ofstream outfile_caffe;
		string out_filename_caffe;

		for (int i=0; i<num_images; i++) {

				string image_path = image_dir + "/" + registry[i];

				cv::Mat img = cv::imread(image_path);
				float times_caffe[2];
				std::vector<float> codingVec_caffe;
				caffe_extractor->extract_singleFeat_1D(img, codingVec_caffe, times_caffe);
		
				std::cout << "Caffe " << times_caffe[0] << ": PREP " << times_caffe[1] << ": NET" << std::endl;
				
				out_filename_caffe = out_dir_caffe[m] + "/" + registry[i].substr(0, registry[i].size()-4) + ".txt";
				
				outfile_caffe.open (out_filename_caffe.c_str());
				if (outfile_caffe.is_open())
				{
				    for (int j=0; j<codingVec_caffe.size(); j++)
				        outfile_caffe << codingVec_caffe[j] << endl;
				    outfile_caffe.close();
				} else
				{
				    std::cerr<< "File not written: " << out_filename_caffe << std::endl;
				}
		}

		// clean classes

		delete caffe_extractor;

	}

	// CUDA cleanup

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cout << endl <<  "done!" << endl;

	cudaDeviceReset();

	exit(EXIT_SUCCESS);
}
