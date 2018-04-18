#pragma once   
#ifndef _PHOTO_STITCHING_H_
#define _PHOTO_STITCHING_H_
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>
#include <algorithm>
#include <functional>
#define KNN_K 2
#define K_DIMENSION 20
#define NUM_HOMOGRAPHY 1000
#define POINT_SELECTED 4
#define INLIERS_THRESHOLD 0.8f
#define MAX_LOOP_NUM INLIERS_THRESHOLD / 0.05f
#define DIST_THRESHOLD 5

using namespace std;
using namespace cv;

struct POINT{
	int index;
	vector<int> dim;
	inline float distance(const POINT &point)const{
		float dis = 0;
		for (int i = 0; i < K_DIMENSION; i++){
			dis += pow(point.dim[i] - dim[i], 2);
		}
		return dis;
	}
	inline bool operator==(const POINT &point){
		for (int i = 0; i < K_DIMENSION; i++){
			if (point.dim[i] != dim[i])
				return false;
		}
		return true;
	}
	/*inline bool operator<(const POINT &point)const{
		return dim[0] < point.dim[0];
	}*/
};

class NODE{
public:
	int left;
	int right;
	int parent;
	int depth = 0;
	POINT _point;
	vector<pair<int, int>>match_feature; // record 2 nearest neighbor match feature from every image's every feature  
	bool nice_match = false;
};

class PHOTO_STITCHING{
public:
	PHOTO_STITCHING();
	PHOTO_STITCHING(vector<string>&);
	void sift_FeatureDetection();
	void show();
	void show_kTree();
	void show_nice_match();
	void set_Root();
	void create_KDimTree(vector<NODE>&, int, int, int);
	void feature_matching();
	void doKNN();
	void _doRANSAC(int, int);
	void doImageWarping();
	void doProjectingAgain();
	void doRANSAC(int, int);
	void image_warping(Mat&);
	void start_stitching();
	void start();
	void find_connected_pic();
	float Weight(Mat&, float, float);
	pair<int, int> search_kDimTree(vector<NODE>, NODE);
	Vec3f interpolated_color(Mat&, float, float);
	Mat find_homography(Point2f[POINT_SELECTED * 2]);
private:
	struct COMPARE{
		bool operator()(const NODE &first, const NODE &second){
			if (first._point.dim[first.depth] != second._point.dim[second.depth])
				return first._point.dim[first.depth] < second._point.dim[second.depth];
			for (int i = 0; i < K_DIMENSION; i++){
				if (first._point.dim[i] != second._point.dim[i])
					return first._point.dim[i] < second._point.dim[i];
			}
		}
	}Compare;
	// SIFT feature detector and feature extractor
	SiftFeatureDetector mDetector;
	SiftDescriptorExtractor mExtractor;
	// key points of each image
	vector<vector<KeyPoint>> mKeypoints;
	// pixel value of each image
	vector<Mat> mImageBuffer;
	// k-d tree of each image
	vector<vector<NODE>> mKDTree; // vector<NODE> stored k-d tree of each image
	vector<Mat> mDescriptor;
	// features of image mark by circle
	vector<Mat> mFeature;
	vector<Mat> mHomography;
	vector<int> mConnectedPicture;
	// best match feature of each picture
	vector<vector<int>> mBestMatchFeature;
	Mat matrixU;
	Mat matrixU_transpose;
	Mat mul_result;
	Mat mEigenvalue;
	Mat mEigenvector;
	Mat mSource_matrix;
	Mat mDst_matrix;
	Mat best_homo_matrix;
	Mat mCompare_Result;
	Mat mCompare_Final_Result;
	// result image
	Mat mResult;
	Mat mFinal_Result;
	Mat mFinal_Result_descriptor;
	Mat mFinal_Result_feature;
	vector<KeyPoint> mFinal_Result_keypoints;
	vector<vector<pair<int, int>>> mKNN;
	vector<vector<int>> mKNN_good_match;
	vector<vector<Point2f>> mSource;
	vector<vector<Point2f>> mDst;
	vector<Point2f> mFinal_Result_source;
	vector<Point2f> mFinal_Result_dst;
};
#endif