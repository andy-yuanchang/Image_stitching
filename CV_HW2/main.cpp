#include"Photo_Stitching.h"

int main() {
	vector<string> image_name;
	/*image_name.push_back("logo/puzzle1.bmp");
	image_name.push_back("logo/puzzle2.bmp");
	image_name.push_back("logo/puzzle3.bmp");
	image_name.push_back("logo/puzzle4.bmp");
	image_name.push_back("logo/target.bmp");
	image_name.push_back("logo/sample.bmp");*/
	image_name.push_back("table/puzzle1.bmp");
	image_name.push_back("table/puzzle2.bmp");
	image_name.push_back("table/puzzle3.bmp");
	image_name.push_back("table/puzzle4.bmp");
	image_name.push_back("table/puzzle5.bmp");
	image_name.push_back("table/puzzle6.bmp");
	image_name.push_back("table/puzzle7.bmp");
	image_name.push_back("table/target.bmp");
	image_name.push_back("table/sample.bmp");
	PHOTO_STITCHING photo(image_name);
	photo.sift_FeatureDetection();
	photo.doKNN();
	photo.start();
	photo.doProjectingAgain();
	//photo.show();
	//photo.set_Root();
	//photo.feature_matching();
	//photo.show_nice_match();
	//photo.find_connected_pic();
	//photo.start_stitching();
	//photo.show_kTree();

	waitKey();
	
	return EXIT_SUCCESS;
}