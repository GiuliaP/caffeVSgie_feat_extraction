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

// Boost includes, to manage output directories
#include <boost/filesystem.hpp>

// Caffe class includes
#include "CaffeFeatExtractor.hpp"
#include "GIEFeatExtractor.hpp"

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

	CaffeFeatExtractor<double> *caffe_extractor;
	GIEFeatExtractor *gie_extractor;

	// .caffemodel containing the pretrained network's weights
	vector <string> caffemodel_file;
	caffemodel_file.push_back("/usr/local/src/robot/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");
	caffemodel_file.push_back("/usr/local/src/robot/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel");

	// .prototxt defining the network structure
	vector <string> prototxt_file_caffe;
	vector <string> prototxt_file_gie;
	prototxt_file_caffe.push_back("/home/icub/giulia/REPOS/caffe_feat_extraction/networks/blvc_reference_caffenet_val_cutfc6.prototxt");
	prototxt_file_gie.push_back("/usr/local/src/robot/GIE/models/bvlc_reference_caffenet/deploy.prototxt");
	prototxt_file_caffe.push_back("/home/icub/giulia/REPOS/caffe_feat_extraction/networks/blvc_googlenet_val_cutpool5.prototxt");
	prototxt_file_gie.push_back("/usr/local/src/robot/GIE/models/bvlc_googlenet/deploy.prototxt");

	// Image dir
	string dset_dir = "iCWU_cc256";
	string image_dir = "/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir;

	// Registries
	vector <string> registry_files;
	registry_files.push_back("/home/icub/giulia/REPOS/caffe_feat_extraction/registries/prova.txt");

	// Output dirs
	vector <string> out_dir_caffe;
	vector <string> out_dir_gie;
	out_dir_gie.push_back("/data/giulia/GIEvsCaffe/GIE/caffenet");
    out_dir_gie.push_back("/data/giulia/GIEvsCaffe/GIE/googlenet");
	out_dir_caffe.push_back("/data/giulia/GIEvsCaffe/Caffe/caffenet");
	out_dir_caffe.push_back("/data/giulia/GIEvsCaffe/Caffe/googlenet");

	// Names of layers to be extracted
	string blob_names_caffe = "fc6";
	string blob_names_gie = "pool5/7x7_s1";
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

	    caffe_extractor = new CaffeFeatExtractor<float>(caffemodel_file_caffe[m],
				prototxt_file_caffe[m],
				blob_names_caffe,
				compute_mode,
				device_id,
				timing);

		gie_extractor = new GIEFeatExtractor(caffemodel_file_gie[m],
		                prototxt_file_gie[m], 256, 256,
		                blob_names_gie,
		                compute_mode,
		                device_id,
		                timing);

		// read registry

		vector<string> registry;
		ifstream infile;
		string line, label;
		infile.open (registry_files[m].c_str());
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

		ofstream outfile_caffe, outfile_gie;
		string out_filename_caffe, out_filename_gie;

		for (int i=0; i<num_images; i++) {

				string image_path = image_dir + "/" + registry[i];

				cv::Mat img = cv::imread(image_path);
				int times_caffe[2];
				int times_gie[2];
				std::vector<float> codingVec_caffe;
				std::vector<float> codingVec_gie;
				caffe_extractor->extract_singleFeat_1D(img, codingVec_caffe, times_caffe);
				gie_extractor->extract_singleFeat_1D(img, codingVec_gie, times_gie);

				std::cout << "Caffe " << times_caffe[0] << ": PREP " << times_caffe[1] << ": NET" << std::endl;
				std::cout << "GIE " << times_gie[0] << ": PREP " << times_gie[1] << ": NET" << std::endl;

				out_filename_caffe = out_dir_caffe[m] + "/" + registry[i].substr(0, registry[i].size()-4) + ".txt";
				out_filename_gie = out_dir_gie[m] + "/" + registry[i].substr(0, registry[i].size()-4) + ".txt";

				string dirpath_caffe = out_filename_caffe.substr(0, out_filename_caffe.size()-4-8-1);
				string dirpath_gie = out_filename_gie.substr(0, out_filename_gie.size()-4-8-1);
				boost::filesystem::path dir_caffe(dirpath_caffe.c_str());
				boost::filesystem::path dir_gie(dirpath_gie.c_str());

				if (!boost::filesystem::exists(dir_caffe))
				{
					boost::filesystem::create_directories(dir_caffe);
					std::cerr<< "Directory Created: " << dirpath_caffe <<std::endl;
				}

				if (!boost::filesystem::exists(dir_gie))
				{
				    boost::filesystem::create_directories(dir_gie);
				    std::cerr<< "Directory Created: " << dirpath_gie <<std::endl;
				}

				outfile_caffe.open (out_filename_caffe.c_str());
				if (outfile_caffe.is_open())
				{
				    for (int j=0; j<codingVec_caffe[i].size(); j++)
				        outfile << codingVec_caffe[i][j] << endl;
				    outfile_caffe.close();
				} else
				{
				    std::cerr<< "File not written: " << out_filename_caffe << std::endl;
				}
				outfile_gie.open (out_filename_gie.c_str());
				if (outfile_gie.is_open())
				{
				    for (int j=0; j<codingVec_gie[i].size(); j++)
				        outfile << codingVec_gie[i][j] << endl;
				    outfile_gie.close();
				} else
				    std::cerr<< "File not written: " << out_filename_gie << std::endl;
		}

		// clean classes

		delete caffe_extractor;
		delete gie_extractor;

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
