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

	//pretrained_binary_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/VGG/VGG_ILSVRC_19_layers.caffemodel");

	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185941-ee17/snapshot_iter_460.caffemodel"); // cat2 exp1
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185623-a924/snapshot_iter_352.caffemodel"); // cat2 exp2
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185848-27c7/snapshot_iter_236.caffemodel"); // cat2 exp3
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185758-3c53/snapshot_iter_116.caffemodel"); // cat2 exp4
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185525-014e/snapshot_iter_56.caffemodel"); // cat2 exp5

	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185354-fa78/snapshot_iter_1088.caffemodel"); // cat5 exp1
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185300-e530/snapshot_iter_824.caffemodel"); // cat5 exp2
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185209-da6f/snapshot_iter_548.caffemodel"); // cat5 exp3
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185105-1a2b/snapshot_iter_276.caffemodel"); // cat5 exp4
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-185001-4745/snapshot_iter_132.caffemodel"); // cat5 exp5

	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-171513-1fd5/snapshot_iter_2136.caffemodel"); // cat10 exp1
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-182440-71c0/snapshot_iter_1616.caffemodel"); // cat10 exp2
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-175804-74d9/snapshot_iter_1088.caffemodel"); // cat10 exp3
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-181503-82c9/snapshot_iter_544.caffemodel"); // cat10 exp4
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-170206-6485/snapshot_iter_272.caffemodel"); // cat10 exp5

	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-125252-956d/snapshot_iter_3320.caffemodel"); // cat15 exp1
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-134315-4df8/snapshot_iter_2516.caffemodel"); // cat15 exp2
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-140714-fe6e/snapshot_iter_1684.caffemodel"); // cat15 exp3
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-153335-42c0/snapshot_iter_844.caffemodel"); // cat15 exp4
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151210-162943-62c8/snapshot_iter_424.caffemodel"); // cat15 exp5

	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151223-170423-d0a4/snapshot_iter_2808.caffemodel"); // cat2 exp1
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151223-170526-9e8e/snapshot_iter_332.caffemodel"); // cat2 exp5
	//pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151223-171144-c556/snapshot_iter_19992.caffemodel"); // cat15 exp1
	pretrained_binary_proto_file.push_back("/data/giulia/digits-jobs/20151223-170707-2c09/snapshot_iter_2540.caffemodel"); // cat15 exp5


	// Text file (.prototxt) defining the network structure
	vector <string> feature_extraction_proto_file;

	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_batch_finetuning/VGG_ILSVRC_19_layers_train_val_imagedatalayer.prototxt");

	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_2class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_2class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_2class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_2class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_2class.prototxt");

	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_5class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_5class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_5class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_5class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_5class.prototxt");

	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_10class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_10class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_10class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_10class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_10class.prototxt");

	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_15class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_15class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_15class.prototxt");
	//feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_15class.prototxt");
	feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_googlenet_2-5-10-15/bvlc_googlenet_memlayer_15class.prototxt");

	// Image dir
	string dset_dir = "iCubWorldUltimate_centroid384_disp_finaltree";
	string image_dir = "/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir;

	// Output dirs
	vector <string> out_dir;
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/test_offtheshelfnets/scores/vgg");

	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_2/9-13/train_1-2-3-4-6-7-8-9_tr_1-2-3-4-5_cam_1_day_1_val_5_tr_1-2-3-4-5_cam_1_day_1_test_10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_2/9-13/train_1-2-3-4-6-7_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_2/9-13/train_1-2-3-4_tr_2_cam_1_day_1_val_5-6-7_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_2/9-13/train_1-2_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_2/9-13/train_1_tr_1-2-3-4-5_cam_1_day_1_val_5_tr_1-2-3-4-5_cam_1_day_1_test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");

	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_5/8-9-13-14-15/train_1-2-3-4-6-7-8-9_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_5/8-9-13-14-15/train_1-2-3-4-6-7_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_5/8-9-13-14-15/train_1-2-3-4_tr_2_cam_1_day_1_val_5-6-7_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_5/8-9-13-14-15/train_1-2_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_5/8-9-13-14-15/train_1_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");

	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/train_1-2-3-4-6-7-8-9_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/train_1-2-3-4-6-7_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/train_1-2-3-4_tr_2_cam_1_day_1_val_5-6-7_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/train_1-2_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/train_1_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");

	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-6-7-8-9_tr_1-2-3-4-5_cam_1_day_1_val_5_tr_1-2-3-4-5_cam_1_day_1_test_10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-6-7_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4_tr_2_cam_1_day_1_val_5-6-7_tr_2_cam_1_day_1_test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	//out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2_tr_2_cam_1_day_1_val_5_tr_2_cam_1_day_1_test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");
	out_dir.push_back("/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir + "_experiments/tuning/scores/googlenet/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1_tr_1-2-3-4-5_cam_1_day_1_val_5_tr_1-2-3-4-5_cam_1_day_1_test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2");


	// Names of layers to be extracted
	string extract_features_blob_names = "prob";
	int num_features = 1;

	// GPU or CPU mode
	string compute_mode = "GPU";
	// If compute_mode="GPU", must specify device ID
	int device_id = 0;

	bool timing = true;

	int batch_size = 200;
	int batch_size_caffe = 100;

	vector <string> registry_files;

	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/flower.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/glass.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/hairclip.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/mouse.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/remote.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/mug.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/pencilcase.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/perfume.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/book.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/hairbrush.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/cellphone.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/ringbinder.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/soapdispenser.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/sunglasses.txt");
	//    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/wallet.txt");


	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/test_10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");

	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/test_10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");

	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/test_10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");

	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	//registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_2-3-4-6-7-8-9-10_tr_1-2-3-4-5_cam_1-2_day_1-2_Y.txt");

/*
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/val_5-6-7_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_2/9-13/val_5_tr_2_cam_1_day_1_Y.txt");

	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/val_5-6-7_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_5/8-9-13-14-15/val_5_tr_2_cam_1_day_1_Y.txt");

	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/val_5-6-7_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_10/3-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");

	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/val_5-6-7_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
	registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/val_5_tr_2_cam_1_day_1_Y.txt");
*/

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
