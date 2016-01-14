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
    // Caffe Initialization
    ////////////////////////////////////////////////////////////////////////////////

    // Caffe class declaration
    CaffeFeatExtractor<float> *caffe_extractor;

    // Binary file (.caffemodel) containing the pretrained network's weights
    vector <string> pretrained_binary_proto_file;
    pretrained_binary_proto_file.push_back("/data/giulia/REPOS/caffe_data_and_models/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");

    // Text file (.prototxt) defining the network structure
    vector <string> feature_extraction_proto_file;
    feature_extraction_proto_file.push_back("/data/giulia/REPOS/caffe_batch_finetuning/imagenet_val_new.prototxt");

    // Image dir
    string dset_dir = "iCubWorldUltimate_centroid256_disp_finaltree";
    string image_dir = "/data/giulia/ICUBWORLD_ULTIMATE/" + dset_dir;

    // Output dirs
    vector <string> out_dir;

    out_dir.push_back("/media/icub/MyPassport/" + dset_dir + "_experiments/test_offtheshelfnets/scores/caffenet/fc6");
    out_dir.push_back("/media/icub/MyPassport/" + dset_dir + "_experiments/test_offtheshelfnets/scores/caffenet/fc7");
    out_dir.push_back("/media/icub/MyPassport/" + dset_dir + "_experiments/test_offtheshelfnets/scores/caffenet/fc8");

    // Names of layers to be extracted
    string extract_features_blob_names = "fc6,fc7,fc8";
    int num_features = 3;

    // GPU or CPU mode
    string compute_mode = "GPU";
    // If compute_mode="GPU", must specify device ID
    int device_id = 0;

    int batch_size = 512;
    int batch_size_caffe = 512;

    vector <string> registry_files;

    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/test_8-9-10_tr_1-2-3-4-5_cam_1_day_1-2_Y.txt");

    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_1_cam_1_day_1_Y.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_2_cam_1_day_1_Y.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_3_cam_1_day_1_Y.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/train_1-2-3-4-5-6-7_tr_4_cam_1_day_1_Y.txt");

    //registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_1-2-3-4-5_cam_1_day_1_Y.txt");
    //registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_2-3-4-5_cam_1_day_1_Y.txt");
    //registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_2-4-5_cam_1_day_1_Y.txt");
    //registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_4-5_cam_1_day_1_Y.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORtransf/train_1-2-3-4-5-6-7_tr_5_cam_1_day_1_Y.txt");

    //registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORinst/train_1-2-3-4-5-6-7_tr_5_cam_1_day_1_Y.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/categorization/Ncat_15/2-3-4-5-6-7-8-9-11-12-13-14-15-19-20/frameORinst/train_1-2-3_tr_5_cam_1_day_1_Y.txt");


    /*
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/flower.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/glass.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/hairclip.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/mouse.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/remote.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/mug.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/pencilcase.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/perfume.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/book.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/hairbrush.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/cellphone.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/ringbinder.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/soapdispenser.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/sunglasses.txt");
    registry_files.push_back("/data/giulia/ICUBWORLD_ULTIMATE/iCubWorldUltimate_registries/full_registries/wallet.txt");
    */

    for (int m=0; m<pretrained_binary_proto_file.size(); m++) {

    	// class instantiation

    	caffe_extractor = new CaffeFeatExtractor<float>(pretrained_binary_proto_file[m],
    			feature_extraction_proto_file[m],
    			extract_features_blob_names,
    			compute_mode,
    			device_id);

    	for (int r=0; r<registry_files.size(); r++) {


    		vector<string> registry;
    		ifstream infile;
    		string line, label;
    		infile.open (registry_files[r].c_str());
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

    		for (int batch_index = 0; batch_index < num_mini_batches; batch_index++) {

    			vector<cv::Mat> images;
    			for (int i=0; i<batch_size; i++) {
    				string image_path = image_dir + "/" + registry[batch_index*batch_size + i];
    				cv::Mat img = cv::imread(image_path);
    				images.push_back(img);
    			}

    			vector< vector<float> > features;
    			caffe_extractor->extractBatch_multipleFeat_1D(images, batch_size_caffe, features);

    			for (int i=0; i<batch_size; i++) {

    				string imgname = registry[batch_index*batch_size + i];

    				for (int ff=0; ff<num_features; ff++)
    				{
    					string out_filename = out_dir[ff] + "/" + imgname.substr(0, imgname.size()-4) + ".txt";
    					string dirpath = out_filename.substr(0, out_filename.size()-4-8-1);

    					boost::filesystem::path dir(dirpath.c_str());
    					if (!boost::filesystem::exists(dir))
    					{
    						boost::filesystem::create_directories(dir);
    						std::cerr<< "Directory Created: " << dirpath <<std::endl;
    					}
    					ofstream outfile;
    					outfile.open (out_filename.c_str());
    					if (outfile.is_open())
    					{
    						for (int j=0; j<features[i+ff*batch_size].size(); j++)
    							outfile << features[i+ff*batch_size][j] << endl;
    						outfile.close();
    					} else
    						std::cerr<< "File not written: " << out_filename << std::endl;
    				}
    			}
    			features.clear();
    			images.clear();
    		}

    		// last batch

    		if (last_batch>0)
    		{
    			vector< vector<float> > features;
    			vector<cv::Mat> images;

    			for (int i=0; i<last_batch; i++) {
    				string image_path = image_dir + "/" + registry[num_mini_batches*batch_size + i];
    				cv::Mat img = cv::imread(image_path);
    				images.push_back(img);
    			}

    			caffe_extractor->extractBatch_multipleFeat_1D(images, last_batch, features);

    			for (int i=0; i<last_batch; i++) {

    				string imgname = registry[num_mini_batches*batch_size + i];

    				for (int ff=0; ff<num_features; ff++)
    				{
    					string out_filename = out_dir[ff] + "/" + imgname.substr(0, imgname.size()-4) + ".txt";
    					string dirpath = out_filename.substr(0, out_filename.size()-4-8-1);

    					boost::filesystem::path dir(dirpath.c_str());
    					if (!boost::filesystem::exists(dir))
    					{
    						boost::filesystem::create_directories(dir);
    						std::cerr<< "Directory Created: " << dirpath <<std::endl;
    					}

    					ofstream outfile;
    					outfile.open (out_filename.c_str());
    					if (outfile.is_open())
    					{
    						for (int j=0; j<features[i+ff*last_batch].size(); j++)
    							outfile << features[i+ff*last_batch][j] << endl;
    						outfile.close();
    					} else
    						std::cerr<< "File not written: " << out_filename << std::endl;

    				}
    			}

    			features.clear();
    			images.clear();

    		}

    		cout<< r << endl;
    	}

    	// delete class

    	delete caffe_extractor;

    }
}
