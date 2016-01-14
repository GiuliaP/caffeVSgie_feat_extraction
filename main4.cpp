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
	// Caffe Initialization
	////////////////////////////////////////////////////////////////////////////////

	// Caffe class declaration
	CaffeFeatExtractor<float> *caffe_extractor;

	// Binary file (.caffemodel) containing the pretrained network's weights
	vector <string> pretrained_binary_proto_file;

	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-195349-3da3/snapshot_iter_2103.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-195228-94d7/snapshot_iter_4236.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-180429-efa3/snapshot_iter_1416.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-195112-83df/snapshot_iter_4188.caffemodel");

	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-194944-fdde/snapshot_iter_4248.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-194705-08c0/snapshot_iter_1062.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-194553-b0f2/snapshot_iter_4248.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160111-112149-f7c1/snapshot_iter_4248.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-194123-26cf/snapshot_iter_354.caffemodel");

	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-193332-4245/snapshot_iter_592.caffemodel");
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20160110-190237-f8f4/snapshot_iter_148.caffemodel");

	// Text file (.prototxt) defining the network structure
	vector <string> feature_extraction_proto_file;

	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_11.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_10.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_9.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_8.prototxt");

	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_7.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_6.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_5.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_4.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_3.prototxt");

	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_2.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet_15_inst_fr_transf/bvlc_reference_caffenet_memlayer_1.prototxt");

	// Image dir
	string dset_dir = "iCubWorldUltimate_centroid256_disp_finaltree";
	string image_dir = "/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir;

	// Output dirs
	vector <string> out_dir;

	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_1_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_3_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_2_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_4_cam_1_day_1");

	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_1-2-3-4-5_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_2-3-4-5_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_2-4-5_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_4-5_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_5_cam_1_day_1");

	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORinst/train_1-2-3-4-5-6-7_tr_5_cam_1_day_1");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/caffenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORinst/train_1-2-3_tr_5_cam_1_day_1");

	// Names of layers to be extracted
	string extract_features_blob_names = "prob";
	int num_features = 1;

	// GPU or CPU mode
	string compute_mode = "GPU";
	// If compute_mode="GPU", must specify device ID
	int device_id = 0;

	bool timing = true;

	int batch_size = 512;
	int batch_size_caffe = 512;

	vector <string> registry_files;

	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");

	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");

	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_5_cam_1_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_5_cam_1_day_1-2_Y.txt");

	for (int m=0; m<pretrained_binary_proto_file.size(); m++) {

		// class instantiation

		caffe_extractor = new CaffeFeatExtractor<float>(pretrained_binary_proto_file[m],
				feature_extraction_proto_file[m],
				extract_features_blob_names,
				compute_mode,
				device_id,
				timing);

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

		// batch size update

		int last_batch = num_images%batch_size;
		int num_mini_batches = num_images/batch_size;
		cout << num_mini_batches << " " << last_batch << endl;

		// feature extraction

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
				out_filename = out_dir[m] + "/" + imgname.substr(0, imgname.size()-4) + ".txt";

				string dirpath = out_filename.substr(0, out_filename.size()-4-8-1);
				boost::filesystem::path dir(dirpath.c_str());
				if (!boost::filesystem::exists(dir))
				{
					boost::filesystem::create_directories(dir);
					std::cerr<< "Directory Created: " << dirpath <<std::endl;
				}

				outfile.open (out_filename.c_str());
				if (outfile.is_open())
				{
					for (int j=0; j<features[i].size(); j++)
						outfile << features[i][j] << endl;
					outfile.close();
				} else
					std::cerr<< "File not written: " << out_filename << std::endl;
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
			out_filename = out_dir[m] + "/" + imgname.substr(0, imgname.size()-4) + ".txt";

			string dirpath = out_filename.substr(0, out_filename.size()-4-8-1);
			boost::filesystem::path dir(dirpath.c_str());
			if (!boost::filesystem::exists(dir))
			{
				boost::filesystem::create_directories(dir);
				std::cerr<< "Directory Created: " << dirpath << std::endl;
			}

			outfile.open (out_filename.c_str());
			if (outfile.is_open())
			{
				for (int j=0; j<features[i].size(); j++)
					outfile << features[i][j] << endl;
				outfile.close();
			} else
				std::cerr<< "File not written: " << out_filename << std::endl;

		}

		features.clear();
		images.clear();

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
