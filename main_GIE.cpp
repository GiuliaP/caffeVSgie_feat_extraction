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
#include "GIEFeatExtractor.h"

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
	// Initialization
	////////////////////////////////////////////////////////////////////////////////

	GIEFeatExtractor *gie_extractor;

	// .caffemodel containing the pretrained network's weights
	vector <string> caffemodel_file;
	caffemodel_file.push_back("/usr/local/src/robot/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");
	caffemodel_file.push_back("/usr/local/src/robot/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel");

	// .prototxt defining the network structure
	vector <string> prototxt_file_gie;
	prototxt_file_gie.push_back("/usr/local/src/robot/GIE/models/bvlc_reference_caffenet/deploy.prototxt");
	prototxt_file_gie.push_back("/usr/local/src/robot/GIE/models/bvlc_googlenet/deploy.prototxt");

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
	string image_dir = "/home/ubuntu/giulia/REPOS/caffe_feat_extraction/" + dset_dir;

	// Registries
	string registry_file = "/home/ubuntu/giulia/REPOS/caffe_feat_extraction/registries/images2.txt";

	// Output dirs
	vector <string> out_dir_gie;
	out_dir_gie.push_back("/home/ubuntu/giulia/GIEvsCaffe/GIE/caffenet/images2");
    out_dir_gie.push_back("/home/ubuntu/giulia/GIEvsCaffe/GIE/googlenet/images2");
	
	// Names of layers to be extracted
	vector<string> blob_names_gie;
    blob_names_gie.push_back("fc6");
    blob_names_gie.push_back("pool5/7x7_s1");

	bool timing = true;

	for (int m=0; m<caffemodel_file.size(); m++) {

		// declare classes

		gie_extractor = new GIEFeatExtractor(caffemodel_file[m],
                        binaryproto_meanfile[m], meanR[m], meanG[m], meanB[m], 
		                prototxt_file_gie[m], 256, 256,
		                blob_names_gie[m],
		                timing);

		// read registry

		vector<string> registry;
		ifstream infile;
		string line, label;
		infile.open (registry_file.c_str());
		infile >> line;
		infile >> label;
        cout << "here" << endl;
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

		ofstream outfile_gie;
		string out_filename_gie;

		for (int i=0; i<num_images; i++) {

				string image_path = image_dir + "/" + registry[i];

				cv::Mat img = cv::imread(image_path);
				float times_gie[2];
				std::vector<float> codingVec_gie;
				gie_extractor->extract_singleFeat_1D(img, codingVec_gie, times_gie);

				std::cout << "GIE " << times_gie[0] << ": PREP " << times_gie[1] << ": NET" << std::endl;

				out_filename_gie = out_dir_gie[m] + "/" + registry[i].substr(0, registry[i].size()-4) + ".txt";

				outfile_gie.open (out_filename_gie.c_str());
				if (outfile_gie.is_open())
				{
				    for (int j=0; j<codingVec_gie.size(); j++)
				        outfile_gie << codingVec_gie[j] << endl;
				    outfile_gie.close();
				} else
                {
				    std::cerr<< "File not written: " << out_filename_gie << std::endl;
                }
		}

		// clean classes

		delete gie_extractor;

	}

}
