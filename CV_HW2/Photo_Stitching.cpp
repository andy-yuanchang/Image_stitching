#include"Photo_Stitching.h"

PHOTO_STITCHING::PHOTO_STITCHING(){
	
}

PHOTO_STITCHING::PHOTO_STITCHING(vector<string>& name){
	Mat tmpImg;
	for (int i = 0; i < name.size(); i++){
		tmpImg = imread(name[i], IMREAD_COLOR);
		mImageBuffer.push_back(tmpImg);
	}
	mKNN.resize(mImageBuffer.size() - 1);
	mKNN_good_match.resize(mImageBuffer.size() - 1);
	mSource.resize(mImageBuffer.size() - 1);
	mDst.resize(mImageBuffer.size() - 1);
	matrixU = Mat(8, 9, CV_32FC1);
	matrixU_transpose = Mat(9, 8, CV_32FC1);
	mul_result = Mat(9, 9, CV_32FC1);
	mEigenvalue = Mat(9, 1, CV_32FC1);
	mEigenvector = Mat(9, 9, CV_32FC1);
	mSource_matrix = Mat(3, 1, CV_32FC1);
	mDst_matrix = Mat(3, 1, CV_32FC1);
	mBestMatchFeature.resize(mImageBuffer.size());
}

void PHOTO_STITCHING::sift_FeatureDetection(){
	for (int i = 0; i < mImageBuffer.size(); i++){
		time_t startTime = time(NULL);

		Mat descriptor;
		Mat feature;
		vector<KeyPoint> keyPoints;
		mDetector.detect(mImageBuffer[i], keyPoints);
		mKeypoints.push_back(keyPoints);
		cout << "Keypoints'" + to_string(i) + " number = " << mKeypoints[i].size() << endl;
		mExtractor.compute(mImageBuffer[i], mKeypoints[i], descriptor);
		mDescriptor.push_back(descriptor);
		cout << "Descriptor's" + to_string(i) + " size = " << mDescriptor[i].size() << endl;
		drawKeypoints(mImageBuffer[i], mKeypoints[i], feature, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		mFeature.push_back(feature);

		time_t endTime = time(NULL);
		cout << "time: " << endTime - startTime << " s" << endl;
		cout << endl;
	}
}

void PHOTO_STITCHING::doKNN(){
	for (int i = 0; i < mImageBuffer.size() - 1; i++){
		cout << "do KNN between picture" + to_string(i) + " and picture" + to_string(mImageBuffer.size() - 1) << endl;
		cout << endl;
		for (int j = 0; j < mDescriptor[i].rows; j++){

			float* difference = new float[mDescriptor[mImageBuffer.size() - 1].rows];
			for (int k = 0; k < mDescriptor[mImageBuffer.size() - 1].rows; k++){
				float diff = 0.0f;
				for (int l = 0; l < mDescriptor[mImageBuffer.size() - 1].cols; l++){
					float feature1 = mDescriptor[i].at<float>(j, l);
					float featrue2 = mDescriptor[mImageBuffer.size() - 1].at<float>(k, l);
					diff += pow(feature1 - featrue2, 2);
				}
				difference[k] = diff;
			}

			bool* choose_flg = new bool[mDescriptor[mImageBuffer.size() - 1].rows];
			for (int k = 0; k < mDescriptor[mImageBuffer.size() - 1].rows; k++){
				choose_flg[k] = false;
			}

			pair<int, int> K_nearest;
			for (int k = 0; k < KNN_K; k++){
				float min_diff = 10000.0f;
				int min_index = 0;
				for (int l = 0; l < mDescriptor[mImageBuffer.size() - 1].rows; l++){
					if (choose_flg[l] == false){
						if (difference[l] < min_diff){
							min_diff = difference[l];
							min_index = l;
						}
					}
				}
				if (k == 0){
					K_nearest.first = min_index;
					choose_flg[min_index] = true;
				}
				else{
					K_nearest.second = min_index;
				}
			}
			if (difference[K_nearest.first] < 0.69f * difference[K_nearest.second]){
				mSource[i].push_back(mKeypoints[i][j].pt);
				mDst[i].push_back(mKeypoints[mImageBuffer.size() - 1][K_nearest.first].pt);
			}
			delete[] difference;
			delete[] choose_flg;
		}
	}
}

void PHOTO_STITCHING::_doRANSAC(int target, int image){
	cout << "do RANSAC between image" + to_string(target) + " and image" + to_string(image) << "..." << endl;
	Mat matrix_homography = Mat(3, 3, CV_32FC1);
	Mat final_homography = Mat(3, 3, CV_32FC1);
	int max_inliers = 0;
	int num_points = mSource[image].size();

	for (int find_num = 0; find_num < NUM_HOMOGRAPHY; find_num++){
		Point2f random_four_points[POINT_SELECTED * 2];
		int point_index[POINT_SELECTED];
		
		for (int i = 0; i < POINT_SELECTED; i++){
			point_index[i] = rand() % num_points;
		}
		
		int pt_index = 0;
		for (int i = 0; i < POINT_SELECTED * 2; i++){
			if (i % 2 == 0)
				random_four_points[i] = mSource[image][point_index[pt_index]];
			if (i % 2 == 1){
				random_four_points[i] = mDst[image][point_index[pt_index]];
				pt_index++;
			}
		}

		matrix_homography = find_homography(random_four_points);
		int num_inliers = 0;
		
		Mat X(3, 1, CV_32FC1);
		for (int i = 0; i < num_points; i++){
			X.at<float>(0, 0) = mSource[image][i].x;
			X.at<float>(1, 0) = mSource[image][i].y;
			X.at<float>(2, 0) = 1.0f;
			X = matrix_homography * X;
			X.at<float>(0, 0) /= X.at<float>(2, 0);
			X.at<float>(1, 0) /= X.at<float>(2, 0);
			float x_dis = X.at<float>(0, 0) - mDst[image][i].x;
			float y_dis = X.at<float>(1, 0) - mDst[image][i].y;
			
			float dis = sqrt(pow(x_dis, 2) + pow(y_dis, 2));
			if (dis < DIST_THRESHOLD)
				num_inliers++;
		}

		if (num_inliers > max_inliers){
			max_inliers = num_inliers;
			final_homography = matrix_homography;
		}
		if (max_inliers > num_points * INLIERS_THRESHOLD)
			break;
	}

	mHomography.push_back(final_homography.clone());
	cout << "done." << endl;
	cout << endl;
}

Mat PHOTO_STITCHING::find_homography(Point2f rand_points[POINT_SELECTED * 2]){
	Mat matrix_homography(3, 3, CV_32FC1);
	/*for (int i = 0; i < POINT_SELECTED * 2; i++){
		cout << rand_points[i].x << " " << rand_points[i].y << endl;
	}*/

	float array_u[] = {
		rand_points[0].x, rand_points[0].y, 1.0f, 0.0f, 0.0f, 0.0f,
		-(rand_points[1].x * rand_points[0].x), -(rand_points[1].x * rand_points[0].y), -rand_points[1].x,

		0.0f, 0.0f, 0.0f, rand_points[0].x, rand_points[0].y, 1.0f,
		-(rand_points[1].y * rand_points[0].x), -(rand_points[1].y * rand_points[0].y), -rand_points[1].y,

		rand_points[2].x, rand_points[2].y, 1.0f, 0.0f, 0.0f, 0.0f,
		-(rand_points[3].x * rand_points[2].x), -(rand_points[3].x * rand_points[2].y), -rand_points[3].x,

		0.0f, 0.0f, 0.0f, rand_points[2].x, rand_points[2].y, 1.0f,
		-(rand_points[3].y * rand_points[2].x), -(rand_points[3].y * rand_points[2].y), -rand_points[3].y,

		rand_points[4].x, rand_points[4].y, 1.0f, 0.0f, 0.0f, 0.0f,
		-(rand_points[5].x * rand_points[4].x), -(rand_points[5].x * rand_points[4].y), -rand_points[5].x,

		0.0f, 0.0f, 0.0f, rand_points[4].x, rand_points[4].y, 1.0f,
		-(rand_points[5].y * rand_points[4].x), -(rand_points[5].y * rand_points[4].y), -rand_points[5].y,

		rand_points[6].x, rand_points[6].y, 1.0f, 0.0f, 0.0f, 0.0f,
		-(rand_points[7].x * rand_points[6].x), -(rand_points[7].x * rand_points[6].y), -rand_points[7].x,

		0.0f, 0.0f, 0.0f, rand_points[6].x, rand_points[6].y, 1.0f,
		-(rand_points[7].y * rand_points[6].x), -(rand_points[7].y * rand_points[6].y), -rand_points[7].y };

	Mat matrix_U(8, 9, CV_32FC1, &array_u);
	Mat matrix_U_transpose(9, 8, CV_32FC1);
	Mat multiply_result(9, 9, CV_32FC1);
	Mat Eigenvalue(9, 1, CV_32FC1);
	Mat Eigenvector(9, 9, CV_32FC1);
	
	matrix_U_transpose = matrix_U.t();
	multiply_result = matrix_U_transpose * matrix_U;
	cv::eigen(multiply_result, true, Eigenvalue, Eigenvector);

	for (int i = 0; i < matrix_homography.rows; i++){
		for (int j = 0; j < matrix_homography.cols; j++){
			matrix_homography.at<float>(i, j) = Eigenvector.at<float>(8, i * 3 + j);
		}
	}

	return matrix_homography.clone();
}

void PHOTO_STITCHING::doImageWarping(){
	cout << "start image warping..." << endl;
	Mat X(3, 1, CV_32FC1);
	mResult = Mat(mImageBuffer[mImageBuffer.size() - 1].rows, mImageBuffer[mImageBuffer.size() - 1].cols, CV_8UC3);
	for (int i = 0; i < mResult.rows; i++){
		for (int j = 0; j < mResult.cols; j++){
			mResult.at<Vec3b>(i, j).val[0] = 0;
			mResult.at<Vec3b>(i, j).val[1] = 0;
			mResult.at<Vec3b>(i, j).val[2] = 0;
		}
	}

	mCompare_Result = Mat(mImageBuffer[mImageBuffer.size() - 1].rows, mImageBuffer[mImageBuffer.size() - 1].cols, CV_8UC3);
	for (int i = 0; i < mCompare_Result.rows; i++){
		for (int j = 0; j < mCompare_Result.cols; j++){
			mCompare_Result.at<Vec3b>(i, j).val[0] = 0;
			mCompare_Result.at<Vec3b>(i, j).val[1] = 0;
			mCompare_Result.at<Vec3b>(i, j).val[2] = 0;
		}
	}
	/*
	result with directly corresponding to original image color
	*/
	for (int image = 0; image < mImageBuffer.size() - 1; image++){
		for (int i = 0; i < mImageBuffer[image].rows; i++){
			for (int j = 0; j < mImageBuffer[image].cols; j++){
				if (mImageBuffer[image].at<Vec3b>(i, j).val[0] == 0 && mImageBuffer[image].at<Vec3b>(i, j).val[1] == 0 && mImageBuffer[image].at<Vec3b>(i, j).val[2] == 0)
					continue;
				X.at<float>(0, 0) = j;
				X.at<float>(1, 0) = i;
				X.at<float>(2, 0) = 1.0f;
				mDst_matrix = mHomography[image] * X;
				mDst_matrix.at<float>(0, 0) /= mDst_matrix.at<float>(2, 0);
				mDst_matrix.at<float>(1, 0) /= mDst_matrix.at<float>(2, 0);
				if (floor(mDst_matrix.at<float>(0, 0)) >= 0 && floor(mDst_matrix.at<float>(1, 0)) >= 0 && floor(mDst_matrix.at<float>(0, 0)) < mCompare_Result.cols && floor(mDst_matrix.at<float>(1, 0)) < mCompare_Result.rows)
					mCompare_Result.at<Vec3b>(floor(mDst_matrix.at<float>(1, 0)), floor(mDst_matrix.at<float>(0, 0))) = mImageBuffer[image].at<Vec3b>(i, j);
			}
		}
	}
	/*
	result with linear blending, find every pixel color in the stitching image
	*/
	vector<float> weight;
	weight.resize(mImageBuffer.size() - 1);
	vector<Vec3f> color;
	color.resize(mImageBuffer.size() - 1);
	float sum = 0.0f;
	for (int i = 0; i < mResult.rows; i++){
		for (int j = 0; j < mResult.cols; j++){
			sum = 0.0f;
			for (int k = 0; k < mImageBuffer.size() - 1; k++){
				X.at<float>(0, 0) = float(j);
				X.at<float>(1, 0) = float(i);
				X.at<float>(2, 0) = 1.0f;
				X = mHomography[k].inv() * X;
				X.at<float>(0, 0) /= X.at<float>(2, 0);
				X.at<float>(1, 0) /= X.at<float>(2, 0);
				color[k] = interpolated_color(mImageBuffer[k], X.at<float>(0, 0), X.at<float>(1, 0));
				weight[k] = Weight(mImageBuffer[k], X.at<float>(0, 0), X.at<float>(1, 0));
				sum += weight[k];
			}
			Vec3f c = Vec3f(0, 0, 0);
			for (int k = 0; k < mImageBuffer.size() - 1; k++){
				c += weight[k] * color[k];
			}
			if (sum > 0)
				c /= sum;
			if (c.val[0] == 0 && c.val[1] == 0 && c.val[2] == 0)
				continue;
			mResult.at<Vec3b>(i, j) = Vec3b(c);
		}
	}
	X.release();

	cout << "done!" << endl;
}

void PHOTO_STITCHING::start(){
	srand(time(NULL));
	for (int i = 0; i < mImageBuffer.size() - 1; i++){
		_doRANSAC(mImageBuffer.size() - 1, i);
	}

	doImageWarping();
	
	imshow("image_stitching_compare", mCompare_Result);
	imwrite("unblending_result.bmp", mCompare_Result);
	imshow("image_stitching", mResult);
	imwrite("blending_result.bmp", mResult);
}
/*
project the stitching image to the target again
*/
void PHOTO_STITCHING::doProjectingAgain(){
	time_t startTime = time(NULL);

	//GaussianBlur(mResult, mResult, Size(5, 5), 0, 0);
	mDetector.detect(mResult, mFinal_Result_keypoints);
	mKeypoints.push_back(mFinal_Result_keypoints);
	cout << "Keypoints' result number = " << mFinal_Result_keypoints.size() << endl;
	mExtractor.compute(mResult, mFinal_Result_keypoints, mFinal_Result_descriptor);
	cout << "Descriptor's result size = " << mFinal_Result_descriptor.size() << endl;
	drawKeypoints(mResult, mFinal_Result_keypoints, mFinal_Result_feature, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("feature", mFinal_Result_feature);
	imshow("feature_", mFeature[mImageBuffer.size() - 2]);

	time_t endTime = time(NULL);
	cout << "time: " << endTime - startTime << " s" << endl;
	cout << endl;

	cout << "do KNN between picture result and picture target" << endl;
	cout << endl;

	int target_index = mImageBuffer.size() - 2;
	for (int j = 0; j < mFinal_Result_descriptor.rows; j++){

		float* difference = new float[mDescriptor[target_index].rows];
		for (int k = 0; k < mDescriptor[target_index].rows; k++){
			float diff = 0.0f;
			for (int l = 0; l < mDescriptor[target_index].cols; l++){
				float feature1 = mFinal_Result_descriptor.at<float>(j, l);
				float featrue2 = mDescriptor[target_index].at<float>(k, l);
				diff += pow(feature1 - featrue2, 2);
			}
			difference[k] = diff;
		}

		bool* choose_flg = new bool[mDescriptor[target_index].rows];
		for (int k = 0; k < mDescriptor[target_index].rows; k++){
			choose_flg[k] = false;
		}

		pair<int, int> K_nearest;
		for (int k = 0; k < KNN_K; k++){
			float min_diff = 10000.0f;
			int min_index = 0;
			for (int l = 0; l < mDescriptor[target_index].rows; l++){
				if (choose_flg[l] == false){
					if (difference[l] < min_diff){
						min_diff = difference[l];
						min_index = l;
					}
				}
			}
			if (k == 0){
				K_nearest.first = min_index;
				choose_flg[min_index] = true;
			}
			else{
				K_nearest.second = min_index;
			}
		}
		if (difference[K_nearest.first] < 0.69f * difference[K_nearest.second]){
			mFinal_Result_source.push_back(mFinal_Result_keypoints[j].pt);
			mFinal_Result_dst.push_back(mKeypoints[target_index][K_nearest.first].pt);
		}
		delete[] difference;
		delete[] choose_flg;
	}

	cout << "do RANSAC between result image and target image..." << endl;
	cout << endl;
	Mat matrix_homography = Mat(3, 3, CV_32FC1);
	Mat final_homography = Mat(3, 3, CV_32FC1);
	int max_inliers = 0;
	int num_points = mFinal_Result_source.size();

	for (int find_num = 0; find_num < NUM_HOMOGRAPHY; find_num++){
		Point2f random_four_points[POINT_SELECTED * 2];
		int point_index[POINT_SELECTED];

		for (int i = 0; i < POINT_SELECTED; i++){
			point_index[i] = rand() % num_points;
		}

		int pt_index = 0;
		for (int i = 0; i < POINT_SELECTED * 2; i++){
			if (i % 2 == 0)
				random_four_points[i] = mFinal_Result_source[point_index[pt_index]];
			if (i % 2 == 1){
				random_four_points[i] = mFinal_Result_dst[point_index[pt_index]];
				pt_index++;
			}
		}

		matrix_homography = find_homography(random_four_points);
		int num_inliers = 0;

		Mat X(3, 1, CV_32FC1);
		for (int i = 0; i < num_points; i++){
			X.at<float>(0, 0) = mFinal_Result_source[i].x;
			X.at<float>(1, 0) = mFinal_Result_source[i].y;
			X.at<float>(2, 0) = 1.0f;
			X = matrix_homography * X;
			X.at<float>(0, 0) /= X.at<float>(2, 0);
			X.at<float>(1, 0) /= X.at<float>(2, 0);
			float x_dis = X.at<float>(0, 0) - mFinal_Result_dst[i].x;
			float y_dis = X.at<float>(1, 0) - mFinal_Result_dst[i].y;

			float dis = sqrt(pow(x_dis, 2) + pow(y_dis, 2));
			if (dis < DIST_THRESHOLD)
				num_inliers++;
		}

		if (num_inliers > max_inliers){
			max_inliers = num_inliers;
			final_homography = matrix_homography;
		}
		if (max_inliers > num_points * INLIERS_THRESHOLD)
			break;
	}

	cout << "done." << endl;
	cout << endl;

	mFinal_Result = mImageBuffer[mImageBuffer.size() - 2].clone();

	Mat X(3, 1, CV_32FC1);
	vector<float> weight;
	weight.resize(1);
	vector<Vec3f> color;
	color.resize(1);

	for (int i = 0; i < mResult.rows; i++){
		for (int j = 0; j < mResult.cols; j++){
			if (mResult.at<Vec3b>(i, j).val[0] == 0 && mResult.at<Vec3b>(i, j).val[1] == 0 && mResult.at<Vec3b>(i, j).val[2] == 0)
				continue;
			X.at<float>(0, 0) = j;
			X.at<float>(1, 0) = i;
			X.at<float>(2, 0) = 1.0f;
			mDst_matrix = final_homography * X;
			mDst_matrix.at<float>(0, 0) /= mDst_matrix.at<float>(2, 0);
			mDst_matrix.at<float>(1, 0) /= mDst_matrix.at<float>(2, 0);
			if (floor(mDst_matrix.at<float>(0, 0)) >= 0 && floor(mDst_matrix.at<float>(1, 0)) >= 0 && floor(mDst_matrix.at<float>(0, 0)) < mFinal_Result.cols && floor(mDst_matrix.at<float>(1, 0)) < mFinal_Result.rows){
				Vec3b temp = mResult.at<Vec3b>(i, j) + Vec3b(50, 50, 50);
				if (temp.val[0] > 255)
					temp.val[0] = 255;
				if (temp.val[1] > 255)
					temp.val[1] = 255;
				if (temp.val[2] > 255)
					temp.val[2] = 255;
				mFinal_Result.at<Vec3b>(floor(mDst_matrix.at<float>(1, 0)), floor(mDst_matrix.at<float>(0, 0))) = temp;
			}
		}
	}
	imshow("final_image_stitching", mFinal_Result);
	imwrite("target_blending_result.bmp", mFinal_Result);

	/*
	----------------------------------------------no blending result--------------------------------------------------------
	*/
	time_t _startTime = time(NULL);

	mDetector.detect(mCompare_Result, mFinal_Result_keypoints);
	mKeypoints.push_back(mFinal_Result_keypoints);
	cout << "Keypoints' result number = " << mFinal_Result_keypoints.size() << endl;
	mExtractor.compute(mCompare_Result, mFinal_Result_keypoints, mFinal_Result_descriptor);
	cout << "Descriptor's result size = " << mFinal_Result_descriptor.size() << endl;
	drawKeypoints(mCompare_Result, mFinal_Result_keypoints, mFinal_Result_feature, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("feature_", mFinal_Result_feature);

	time_t _endTime = time(NULL);
	cout << "time: " << _endTime - _startTime << " s" << endl;
	cout << endl;

	cout << "do KNN between picture result and picture target" << endl;
	cout << endl;

	int target_index_ = mImageBuffer.size() - 2;
	for (int j = 0; j < mFinal_Result_descriptor.rows; j++){

		float* _difference = new float[mDescriptor[target_index_].rows];
		for (int k = 0; k < mDescriptor[target_index_].rows; k++){
			float diff = 0.0f;
			for (int l = 0; l < mDescriptor[target_index_].cols; l++){
				float feature1 = mFinal_Result_descriptor.at<float>(j, l);
				float featrue2 = mDescriptor[target_index_].at<float>(k, l);
				diff += pow(feature1 - featrue2, 2);
			}
			_difference[k] = diff;
		}

		bool* choose_flg = new bool[mDescriptor[target_index_].rows];
		for (int k = 0; k < mDescriptor[target_index_].rows; k++){
			choose_flg[k] = false;
		}

		pair<int, int> K_nearest;
		for (int k = 0; k < KNN_K; k++){
			float min_diff = 10000.0f;
			int min_index = 0;
			for (int l = 0; l < mDescriptor[target_index_].rows; l++){
				if (choose_flg[l] == false){
					if (_difference[l] < min_diff){
						min_diff = _difference[l];
						min_index = l;
					}
				}
			}
			if (k == 0){
				K_nearest.first = min_index;
				choose_flg[min_index] = true;
			}
			else{
				K_nearest.second = min_index;
			}
		}
		if (_difference[K_nearest.first] < 0.69f * _difference[K_nearest.second]){
			mFinal_Result_source.push_back(mFinal_Result_keypoints[j].pt);
			mFinal_Result_dst.push_back(mKeypoints[target_index_][K_nearest.first].pt);
		}
		delete[] _difference;
		delete[] choose_flg;
	}

	cout << "do RANSAC between result image and target image..." << endl;
	cout << endl;
	Mat matrix_homography_ = Mat(3, 3, CV_32FC1);
	Mat final_homography_ = Mat(3, 3, CV_32FC1);
	int max_inliers_ = 0;
	int num_points_ = mFinal_Result_source.size();

	for (int find_num = 0; find_num < NUM_HOMOGRAPHY; find_num++){
		Point2f random_four_points[POINT_SELECTED * 2];
		int point_index[POINT_SELECTED];

		for (int i = 0; i < POINT_SELECTED; i++){
			point_index[i] = rand() % num_points_;
		}

		int pt_index = 0;
		for (int i = 0; i < POINT_SELECTED * 2; i++){
			if (i % 2 == 0)
				random_four_points[i] = mFinal_Result_source[point_index[pt_index]];
			if (i % 2 == 1){
				random_four_points[i] = mFinal_Result_dst[point_index[pt_index]];
				pt_index++;
			}
		}

		matrix_homography_ = find_homography(random_four_points);
		int num_inliers = 0;

		Mat X(3, 1, CV_32FC1);
		for (int i = 0; i < num_points_; i++){
			X.at<float>(0, 0) = mFinal_Result_source[i].x;
			X.at<float>(1, 0) = mFinal_Result_source[i].y;
			X.at<float>(2, 0) = 1.0f;
			X = matrix_homography_ * X;
			X.at<float>(0, 0) /= X.at<float>(2, 0);
			X.at<float>(1, 0) /= X.at<float>(2, 0);
			float x_dis = X.at<float>(0, 0) - mFinal_Result_dst[i].x;
			float y_dis = X.at<float>(1, 0) - mFinal_Result_dst[i].y;

			float dis = sqrt(pow(x_dis, 2) + pow(y_dis, 2));
			if (dis < DIST_THRESHOLD)
				num_inliers++;
		}

		if (num_inliers > max_inliers_){
			max_inliers_ = num_inliers;
			final_homography_ = matrix_homography_;
		}
		if (max_inliers_ > num_points_ * INLIERS_THRESHOLD)
			break;
	}

	cout << "done." << endl;
	cout << endl;

	mCompare_Final_Result = mImageBuffer[mImageBuffer.size() - 2].clone();

	float _sum = 0.0f;
	for (int i = 0; i < mCompare_Final_Result.rows; i++){
		for (int j = 0; j < mCompare_Final_Result.cols; j++){
			_sum = 0.0f;
			for (int k = 0; k < 1; k++){
				X.at<float>(0, 0) = float(j);
				X.at<float>(1, 0) = float(i);
				X.at<float>(2, 0) = 1.0f;
				X = final_homography_.inv() * X;
				X.at<float>(0, 0) /= X.at<float>(2, 0);
				X.at<float>(1, 0) /= X.at<float>(2, 0);
				color[k] = interpolated_color(mCompare_Result, X.at<float>(0, 0), X.at<float>(1, 0));
				weight[k] = Weight(mCompare_Result, X.at<float>(0, 0), X.at<float>(1, 0));
				_sum += weight[k];
			}
			Vec3f c = Vec3f(0, 0, 0);
			c += weight[0] * color[0];
			if (_sum > 0)
				c /= _sum;
			/*if (c.val[0] == 0 && c.val[1] == 0 && c.val[2] == 0)
			continue;*/
			c += Vec3f(20, 20, 20);
			if (c.val[0] > 255)
				c.val[0] = 255;
			if (c.val[1] > 255)
				c.val[1] = 255;
			if (c.val[1] > 255)
				c.val[1] = 255;
			mCompare_Final_Result.at<Vec3b>(i, j) = Vec3b(c);
		}
	}
	//GaussianBlur(mCompare_Final_Result, mCompare_Final_Result, Size(3, 3), 0, 0);
	imshow("final_image_stitching_compare", mCompare_Final_Result);
	imwrite("target_unblending_result.bmp", mCompare_Final_Result);
}

void PHOTO_STITCHING::set_Root(){
	mKDTree.resize(mImageBuffer.size());
	for (int i = 0; i < mKDTree.size(); i++){ // number of trees
		for (int j = 0; j < mDescriptor[i].rows; j++){ // number of keypoints
			NODE tmp;
			for (int k = 0; k < mDescriptor[i].cols; k++){ // each key point has a descriptor, and each descriptor has 128-d 
				tmp._point.dim.push_back(mDescriptor[i].at<float>(j, k));
			}
			mKDTree[i].push_back(tmp);
			mKDTree[i][j]._point.index = j;
		}
		/*nth_element(mKDTree[i].first.begin()
			, mKDTree[i].first.begin() + mKDTree[i].first.size() / 2
			, mKDTree[i].first.end(), mKDTree[i].second);
		cout << "The median of " << to_string(i) << " is " 
			<< mKDTree[i].first[mKDTree[i].first.size() / 2]._point.dim[0] << endl;*/

		create_KDimTree(mKDTree[i], 0, mKDTree[i].size() - 1, 0);
	}

}

void PHOTO_STITCHING::create_KDimTree(vector<NODE>& tree, int low, int high, int depth){
	int mid = (high + low) / 2;
	tree[mid].parent = mid;
	tree[mid].depth = depth;
	nth_element(tree.begin() + low, tree.begin() + mid, tree.begin() + high, Compare);

	int left_high = mid - 1;
	int left_mid = (low + left_high) / 2; //left child's median position
	int right_low = mid + 1;
	int right_mid = (right_low + high) / 2; //right child's median position	
	//cout << "kmtree" << endl;
	if (mid > low){
		//cout << "left" << endl;
		create_KDimTree(tree, low, left_high, (depth + 1) % K_DIMENSION);
		tree[left_mid].parent = mid;
		tree[mid].left = left_mid;
	}
	else{
		tree[mid].left = -1;
	}
	if (mid < high){
		//cout << "right" << endl;
		create_KDimTree(tree, right_low, high, (depth + 1) % K_DIMENSION);
		tree[right_mid].parent = mid;
		tree[mid].right = right_mid;

	}
	else{ 
		tree[mid].right = -1; 
	}
}
/*
tree is the kd-tree, and we want to search the two nearest neighbors to target.
it will return the type pair<int, int> stored the two nearest neighbors
*/
pair<int, int> PHOTO_STITCHING::search_kDimTree(vector<NODE> tree, NODE target){
	vector<NODE> mStack; // stored the path target visited 
	NODE kd_point;
	if (tree.size() % 2 == 1)
		kd_point = tree[tree.size() / 2];
	else
		kd_point = tree[tree.size() / 2 - 1];
	POINT nearest = kd_point._point; // initialize the nearest neighbor
	float max_dis = 0.0f;

	while (kd_point.left != -1 || kd_point.right != -1){ // break from the while loop if we visit the leaf
		mStack.push_back(kd_point);
		if (nearest.distance(target._point) > kd_point._point.distance(target._point)){
			nearest = kd_point._point; // update the nearest neighbor
			max_dis = kd_point._point.distance(target._point); // update the distance from nearest neighbor to the target
		}
		int dimension = kd_point.depth;
		if (target._point.dim[dimension] <= kd_point._point.dim[dimension]){
			if (kd_point.left == -1)
				break;
			NODE left = tree[kd_point.left];
			kd_point = left;
		}
		else{
			if (kd_point.right == -1)
				break;
			NODE right = tree[kd_point.right];
			kd_point = right;
		}
		//cout << "left: " << kd_point.left << ", right: " << kd_point.right << endl;
	}
	nearest = mStack[mStack.size() - 1]._point;
	max_dis = nearest.distance(target._point);

	float min_dis = 9999.0f;
	pair<int, int> nearestNeighbors;
	while (mStack.size() != 0)
	{
		NODE back_node = mStack[mStack.size() - 1];
		int dimension = back_node.depth; // the spilt direction
		/*
		determine the subtree we should visit whether or not
		*/
		if (abs(target._point.dim[dimension] - back_node._point.dim[dimension]) < max_dis && (back_node.right != -1 && back_node.left != -1)){
			if (target._point.dim[dimension] <= back_node._point.dim[dimension]){
				if (back_node.right != -1 ){
					NODE right = tree[back_node.right];
					kd_point = right; // if target is located in left subtree, then we should visit the right subtree
					mStack.pop_back();
				}
				
			}
			else{
				if (back_node.left != -1){
					NODE left = tree[back_node.left];
					kd_point = left; // if target is located in right subtree, then we should visit the left subtree
					mStack.pop_back();
				}
				
			}
			mStack.push_back(kd_point);
		}
		else{
			mStack.pop_back();
		}
		if (nearest.distance(target._point) > kd_point._point.distance(target._point)){
			nearestNeighbors.first = kd_point._point.index;
			nearestNeighbors.second = nearest.index;
			nearest = kd_point._point;
			min_dis = kd_point._point.distance(target._point);
		}
	}
	// best matching error must lower 0.6 times to second matching error, e1 - NN / e2 - NN < 0.6
	/*if ((target._point.distance(tree[nearestNeighbors.first]._point) - KNN_K) / (target._point.distance(tree[nearestNeighbors.second]._point) - KNN_K) > 0.6f){
		nearestNeighbors.first = -1;
		nearestNeighbors.second = -1;
	}*/
	return nearestNeighbors;
}

void PHOTO_STITCHING::feature_matching(){
	for (int i = 0; i < mKDTree.size() - 1; i++){ 
		// the i-th kd-tree
		time_t start_time = time(NULL);
		cout << "tree[" + to_string(i) + "] start find feature matching..." << endl;
		for (int j = 0; j < mKDTree[i].size(); j++){
			// the j-th keypoint of i-th kd-tree
			mKDTree[i][j].match_feature.push_back(search_kDimTree(mKDTree[mKDTree.size() - 1], mKDTree[i][j]));
			if ((mKDTree[i][j]._point.distance(mKDTree[mKDTree.size() - 1][mKDTree[i][j].match_feature[0].first]._point) - KNN_K)
				/ (mKDTree[i][j]._point.distance(mKDTree[mKDTree.size() - 1][mKDTree[i][j].match_feature[0].second]._point) - KNN_K) < 0.69f){
				mKDTree[i][j].nice_match = true;
			}
			/*for (int k = 0; k < mKDTree.size(); k++){
				//the k-th kd-tree, we want to find the match feature between i-th and k-th 
				if (i == k){
					mKDTree[i][j].match_feature.push_back(pair<int, int>(-1, -1)); // we can ignore the node if it's (-1,-1)
				}
				else{
					mKDTree[i][j].match_feature.push_back(
						search_kDimTree(mKDTree[k], mKDTree[i][j])
						);
				}
			}*/
		}
		time_t end_time = time(NULL);
		cout << "time: " << end_time - start_time << " s." << endl;
		cout << endl;
	}
}
/*
find the picture j connected to the picture i
*/
void PHOTO_STITCHING::find_connected_pic(){
	cout << "start find connected picture..." << endl;
	for (int i = 0; i < mKDTree.size(); i++){
		cout << "find connected picture to the picture" + to_string(i) << endl;
		int maxnum_inlier = 0;
		int maxnum_inlier_index;
		for (int j = 0; j < mKDTree.size(); j++){
			int num_inlier = 0;
			for (int k = 0; k < mKDTree[i].size(); k++){
				if (i == j)
					continue;
				int first_index = mKDTree[i][k].match_feature[j].first;
				int second_index = mKDTree[i][k].match_feature[j].second;
				if (first_index == -1)
					continue;
				if ((mKDTree[i][k]._point.distance(mKDTree[j][first_index]._point) - KNN_K) / (mKDTree[i][k]._point.distance(mKDTree[j][second_index]._point) - KNN_K) < 0.6f){
					num_inlier++;
				}
			}
			if (num_inlier > maxnum_inlier){
				maxnum_inlier = num_inlier;
				maxnum_inlier_index = j;
			}
		}
		mConnectedPicture.push_back(maxnum_inlier_index);
		for (int j = 0; j < mKDTree[i].size(); j++){
			int first_index = mKDTree[i][j].match_feature[maxnum_inlier_index].first;
			int second_index = mKDTree[i][j].match_feature[maxnum_inlier_index].second;
			if ((mKDTree[i][j]._point.distance(mKDTree[maxnum_inlier_index][first_index]._point) - KNN_K) / (mKDTree[i][j]._point.distance(mKDTree[maxnum_inlier_index][second_index]._point) - KNN_K) < 0.6f){
				mBestMatchFeature[i].push_back(j);
			}
		}
		cout << "picture " + to_string(maxnum_inlier_index) << endl;
		cout << endl;
	}
	cout << "done." << endl;
}

void PHOTO_STITCHING::doRANSAC(int target, int image){
	cout << "do RANSAC between image" + to_string(target) + " and image" + to_string(image) << "..." << endl;
	float inliers_threshold = INLIERS_THRESHOLD;
	Mat matrix_homography;
	Mat best_homography;
	int max_inliers = 0;
	int best_inliers = -1;
	for (int i = 0; i < MAX_LOOP_NUM; i++){
		int num_inliers = 0;
		int current_findedNum = 0;
		for (int j = 0; j < NUM_HOMOGRAPHY; j++){
			vector<Point2f> source_points;
			int tmp_pointIndex[POINT_SELECTED];
			for (int k = 0; k < POINT_SELECTED; k++){
				tmp_pointIndex[k] = (rand() % mKeypoints[image].size());
				while (mKDTree[image][tmp_pointIndex[k]].nice_match == false){
					tmp_pointIndex[k] = (rand() % mKeypoints[image].size());
				}
				//cout << tmp_pointIndex[k] << endl;
				source_points.push_back(mKeypoints[image].at(tmp_pointIndex[k]).pt);
			}
			int num_sets = KNN_K * KNN_K * KNN_K * KNN_K;

			for (int k = 0; k < num_sets; k++){
				num_inliers = 0;
				vector<Point2f> dst_points;
				int index_k;
				int index = k;
				for (int l = 0; l < POINT_SELECTED; l++){
					index_k = index % KNN_K;
					int nearest_index;
					if (index_k == 0)
						nearest_index = mKDTree[image][tmp_pointIndex[l]].match_feature[0].first;
					else
						nearest_index = mKDTree[image][tmp_pointIndex[l]].match_feature[0].second;
					dst_points.push_back(mKeypoints[target][nearest_index].pt);
					index = index / KNN_K;
				}
				/*
				do projective mapping.
				construct the homography matrix.
				*/
				float array_u[] = { 
					source_points[0].x, source_points[0].y, 1.0f, 0.0f, 0.0f, 0.0f,
					-(dst_points[0].x*source_points[0].x), -(dst_points[0].x*source_points[0].y), -dst_points[0].x,

					0.0f, 0.0f, 0.0f, source_points[0].x, source_points[0].y, 1.0f,
					-(dst_points[0].y*source_points[0].x), -(dst_points[0].y*source_points[0].y), -dst_points[0].y,

					source_points[1].x, source_points[1].y, 1.0f, 0.0f, 0.0f, 0.0f,
					-(dst_points[1].x*source_points[1].x), -(dst_points[1].x*source_points[1].y), -dst_points[1].x,

					0.0f, 0.0f, 0.0f, source_points[1].x, source_points[1].y, 1.0f,
					-(dst_points[1].y*source_points[1].x), -(dst_points[1].y*source_points[1].y), -dst_points[1].y,

					source_points[2].x, source_points[2].y, 1.0f, 0.0f, 0.0f, 0.0f,
					-(dst_points[2].x*source_points[2].x), -(dst_points[2].x*source_points[2].y), -dst_points[2].x,

					0.0f, 0.0f, 0.0f, source_points[2].x, source_points[2].y, 1.0f,
					-(dst_points[2].y*source_points[2].x), -(dst_points[2].y*source_points[2].y), -dst_points[2].y,

					source_points[3].x, source_points[3].y, 1.0f, 0.0f, 0.0f, 0.0f,
					-(dst_points[3].x*source_points[3].x), -(dst_points[3].x*source_points[3].y), -dst_points[3].x,

					0.0f, 0.0f, 0.0f, source_points[3].x, source_points[3].y, 1.0f,
					-(dst_points[3].y*source_points[3].x), -(dst_points[3].y*source_points[3].y), -dst_points[3].y };

				matrixU = Mat(8, 9, CV_32FC1, &array_u);
				matrixU_transpose = matrixU.t();
				mul_result = matrixU_transpose * matrixU;
				cv::eigen(mul_result, true, mEigenvalue, mEigenvector); 
				for (int l = 0; l < matrix_homography.rows; l++){
					for (int m = 0; m < matrix_homography.cols; m++){
						matrix_homography.at<float>(l, m) = mEigenvector.at<float>(l * 3 + m, 8);
						//cout << matrix_homography.at<float>(l, m) << " ";
					}
					//cout << endl;
				}

				for (int l = 0; l < mKeypoints[image].size(); l++){
					float array_source[] = { mKeypoints[image][l].pt.x, mKeypoints[image][l].pt.y, 1 };
					mSource_matrix = Mat(3, 1, CV_32FC1, &array_source);
					mDst_matrix = matrix_homography * mSource_matrix;
					//cout << "source: " << mSource_matrix.at<float>(0, 0) << " " << mSource_matrix.at<float>(1, 0) << " " << mSource_matrix.at<float>(2, 0) << endl;
					//cout << "dst: " << mDst_matrix.at<float>(0, 0) << " " << mDst_matrix.at<float>(1, 0) << " " << mDst_matrix.at<float>(2, 0) << endl;
					float x = mDst_matrix.at<float>(0, 0) / mDst_matrix.at<float>(2, 0);
					float y = mDst_matrix.at<float>(1, 0) / mDst_matrix.at<float>(2, 0);
					mDst_matrix.at<float>(0, 0) = x;
					mDst_matrix.at<float>(1, 0) = y;

					float min_dis = 10000.0f;
					for (int m = 0; m < KNN_K; m++){
						float x_dis;
						float y_dis;
						float dis = 0.0f;
						int nearest_index_;
						if (m % KNN_K == 0)
							nearest_index_ = mKDTree[image][l].match_feature[0].first;
						else
							nearest_index_ = mKDTree[image][l].match_feature[0].second;
						x_dis = mDst_matrix.at<float>(0, 0) - mKeypoints[target][nearest_index_].pt.x;
						y_dis = mDst_matrix.at<float>(1, 0) - mKeypoints[target][nearest_index_].pt.y;
						dis = sqrt(pow(x_dis, 2) + pow(y_dis, 2));
						if (dis < min_dis)
							min_dis = dis;
					}
					//cout << min_dis << endl;
					if (min_dis < DIST_THRESHOLD)
						num_inliers++;
					//dst_points.clear();
					if ((float)num_inliers / mKeypoints[image].size() > inliers_threshold){
						break;
					}
				}
				//source_points.clear();
				//cout << (float)num_inliers / mKeypoints[target].size() << endl;
				if ((float)num_inliers / mKeypoints[image].size() > inliers_threshold){
					current_findedNum = j;
					break;
				}
				else if ((float)num_inliers / mKeypoints[image].size() < inliers_threshold && (j == NUM_HOMOGRAPHY - 1)){
					cout << "Can't find good match when INLIERS_THRESHOLD = " << inliers_threshold << endl;
					inliers_threshold -= 0.05f;
					if (inliers_threshold == 0)
						break;
					cout << "Reduce the INLIERS_THRESHOLD and try to find again..." << endl;
				}
			}
			if (inliers_threshold == 0){
				cout << "Image" + to_string(image) + " and Image" + to_string(target) + " can't not connect..." << endl;
				break;
			}
			else if ((float)num_inliers / mKeypoints[image].size() > inliers_threshold){
				cout << "Loop: " << i << endl;
				cout << "Inliers_threshold: " << inliers_threshold << endl;
				cout << "Find " << current_findedNum << " times." << endl;
				cout << "Number of inliers: " << num_inliers << endl;
				cout << "Total points: " << mKeypoints[image].size() << endl;
				if (num_inliers > best_inliers)
					best_homo_matrix = matrix_homography;
				break;
			}
		}
	}
	
	mHomography.push_back(best_homo_matrix);
	cout << "done!" << endl;
}

void PHOTO_STITCHING::image_warping(Mat& target){

	//cout << "start image" + to_string(image) + " warping" << endl;
	cout << "start image warping..." << endl;

	float x_min = 0;
	float y_min = 0;
	float x_max = mImageBuffer[0].cols - 1;
	float y_max = mImageBuffer[0].rows - 1;

	Mat X(3, 1, CV_32FC1);
	for (int i = 1; i < mImageBuffer.size() - 1; i++){
		X.at<float>(0, 0) = float(0);
		X.at<float>(1, 0) = float(0);
		X.at<float>(2, 0) = 1.0f;
		X = mHomography[i] * X;
		X.at<float>(0, 0) /= X.at<float>(2, 0);
		X.at<float>(1, 0) /= X.at<float>(2, 0);
		x_min = min(x_min, X.at<float>(0, 0));
		y_min = min(y_min, X.at<float>(1, 0));
		x_max = max(x_max, X.at<float>(0, 0));
		y_max = max(y_max, X.at<float>(1, 0));

		X.at<float>(0, 0) = float(0);
		X.at<float>(1, 0) = float(mImageBuffer[i].rows - 1);
		X.at<float>(2, 0) = 1.0f;
		X = mHomography[i] * X;
		X.at<float>(0, 0) /= X.at<float>(2, 0);
		X.at<float>(1, 0) /= X.at<float>(2, 0);
		x_min = min(x_min, X.at<float>(0, 0));
		y_min = min(y_min, X.at<float>(1, 0));
		x_max = max(x_max, X.at<float>(0, 0));
		y_max = max(y_max, X.at<float>(1, 0));

		X.at<float>(0, 0) = float(mImageBuffer[i].cols - 1);
		X.at<float>(1, 0) = float(0);
		X.at<float>(2, 0) = 1.0f;
		X = mHomography[i] * X;
		X.at<float>(0, 0) /= X.at<float>(2, 0);
		X.at<float>(1, 0) /= X.at<float>(2, 0);
		x_min = min(x_min, X.at<float>(0, 0));
		y_min = min(y_min, X.at<float>(1, 0));
		x_max = max(x_max, X.at<float>(0, 0));
		y_max = max(y_max, X.at<float>(1, 0));

		X.at<float>(0, 0) = float(mImageBuffer[i].cols - 1);
		X.at<float>(1, 0) = float(mImageBuffer[i].rows - 1);
		X.at<float>(2, 0) = 1.0f;
		X = mHomography[i] * X;
		X.at<float>(0, 0) /= X.at<float>(2, 0);
		X.at<float>(1, 0) /= X.at<float>(2, 0);
		x_min = min(x_min, X.at<float>(0, 0));
		y_min = min(y_min, X.at<float>(1, 0));
		x_max = max(x_max, X.at<float>(0, 0));
		y_max = max(y_max, X.at<float>(1, 0));
	}

	int x_low, x_high;
	int y_low, y_high;
  	x_low = int(floor(x_min));
	x_high = int(ceil(x_max));
	y_low = int(floor(y_min));
	y_high = int(ceil(y_max));

	//mResult = Mat(x_high - x_low + 1, y_high - y_low + 1, CV_8UC3);
	mResult = Mat(target.rows, target.cols, CV_8UC3);
	for (int i = 0; i < mResult.rows; i++){
		for (int j = 0; j < mResult.cols; j++){
			mResult.at<Vec3b>(i, j).val[0] = 0;
			mResult.at<Vec3b>(i, j).val[1] = 0;
			mResult.at<Vec3b>(i, j).val[2] = 0;
		}
	}

	//cout << result_height << " " << result_width << endl;

	/*for (int i = 0; i < target.rows; i++){
		for (int j = 0; j < target.cols; j++){
			mResult.at<Vec3b>(i, j) = target.at<Vec3b>(i, j);
		}
	}*/



	/*for (int i = 0; i < mImageBuffer[0].rows; i++){
		for (int j = 0; j < mImageBuffer[0].cols; j++){
			X.at<float>(0, 0) = i;
			X.at<float>(1, 0) = j;
			X.at<float>(2, 0) = 1.0f;
			X = mHomography[0] * X;
			X.at<float>(0, 0) /= X.at<float>(2, 0);
			X.at<float>(1, 0) /= X.at<float>(2, 0);
			mResult.at<Vec3b>(X.at<float>(1, 0), X.at<float>(0, 0)) = mImageBuffer[0].at<Vec3b>(i, j);
		}
	}*/
	//cout << mDst_matrix.at<float>(0, 0) << " " << x_min << endl;
	//cout << mDst_matrix.at<float>(1, 0) << " " << y_min << endl;
	/*for (int i = 0; i < mImageBuffer[image].rows; i++){
		for (int j = 0; j < mImageBuffer[image].cols; j++){
			mSource_matrix.at<float>(0, 0) = i;
			mSource_matrix.at<float>(1, 0) = j;
			mSource_matrix.at<float>(2, 0) = 1;
			cout << "source: " << mSource_matrix.at<float>(0, 0) << " " << mSource_matrix.at<float>(1, 0) << " " << mSource_matrix.at<float>(2, 0) << endl;
			mDst_matrix = mHomography[image] * mSource_matrix;
			cout << mDst_matrix.at<float>(0, 0) << " " << mDst_matrix.at<float>(1, 0) << " " << mDst_matrix.at<float>(2, 0) << endl;
			for (int l = 0; l < matrix_homography.rows; l++){
				for (int m = 0; m < matrix_homography.cols; m++){
					cout << matrix_homography.at<float>(l, m) << " ";
				}
				cout << endl;
			}
			if (mDst_matrix.at<float>(2, 0) == 0){
				mDst_matrix.at<float>(0, 0) = 0;
				mDst_matrix.at<float>(1, 0) = 0;
			}
			else{
				mDst_matrix.at<float>(0, 0) = mDst_matrix.at<float>(0, 0) / mDst_matrix.at<float>(2, 0);
				mDst_matrix.at<float>(1, 0) = mDst_matrix.at<float>(1, 0) / mDst_matrix.at<float>(2, 0);
			}
			cout << mDst_matrix.at<float>(0, 0) << " " << mDst_matrix.at<float>(1, 0) << " " << mDst_matrix.at<float>(2, 0) << endl;
			//cout << (int)mDst_matrix.at<float>(0, 0) << " " << x_min << endl;
			//cout << (int)mDst_matrix.at<float>(1, 0) << " " << y_min << endl;
			//cout << mDst_matrix.at<float>(0, 0) << " " << x_min << endl;
			//cout << mDst_matrix.at<float>(1, 0) << " " << y_min << endl;
			//cout << target.at<Vec3b>(i, j) << endl;
			mResult.at<Vec3b>((int)mDst_matrix.at<float>(0, 0) - (int)x_min, (int)mDst_matrix.at<float>(1, 0) - (int)y_min) = mImageBuffer[image].at<Vec3b>(i, j);
			//cout << (int)mDst_matrix.at<float>(0, 0) << " " << (int)mDst_matrix.at<float>(1, 0) << endl;
		}
	}*/
	for (int i = 0; i < mHomography.size(); i++){
		for (int x = 0; x < mHomography[i].rows; x++){
			for (int y = 0; y < mHomography[i].cols; y++){
				cout << mHomography[i].at<float>(x, y) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	vector<float> weight;
	weight.resize(mImageBuffer.size() - 1);
	vector<Vec3f> color;
	color.resize(mImageBuffer.size() - 1);
	float sum = 0.0f;
	for (int i = 0; i < mResult.rows; i++){
		for (int j = 0; j < mResult.cols; j++){
			sum = 0.0f;
			for (int k = 0; k < mImageBuffer.size() - 1; k++){
				X.at<float>(0, 0) = float(i) + float(y_low);
				X.at<float>(1, 0) = float(j) + float(x_low);
				X.at<float>(2, 0) = 1.0f;
				X = mHomography[k].inv() * X;
				X.at<float>(0, 0) /= X.at<float>(2, 0);
				X.at<float>(1, 0) /= X.at<float>(2, 0);
				color[k] = interpolated_color(mImageBuffer[k], X.at<float>(0, 0), X.at<float>(1, 0));
				weight[k] = Weight(mImageBuffer[k], X.at<float>(0, 0), X.at<float>(1, 0));
				sum += weight[k];
			}
			Vec3f c = Vec3f(0, 0, 0);
			for (int k = 0; k < mImageBuffer.size() - 1; k++){
				c += weight[k] * color[k];
			}
			if (sum > 0)
				c /= sum;
			if (c.val[0] == 0 && c.val[1] == 0 && c.val[2] == 0)
				continue;
			mResult.at<Vec3b>(i, j) = Vec3b(c);
		}
	}
	X.release();

	cout << "done!" << endl;
}

float PHOTO_STITCHING::Weight(Mat& pic, float x, float y){
	if (x >= 0 && x <= float(pic.cols - 1) && y >= 0 && y <= float(pic.rows - 1)){
		float dx, dy;
		dx = 2 * fabs(x - float(pic.cols - 1) / 2) / float(pic.cols + 1);
		dy = 2 * fabs(y - float(pic.rows - 1) / 2) / float(pic.rows + 1);
		return (1 - max(dx, dy));
	}
	return 0.0f;
}

Vec3f PHOTO_STITCHING::interpolated_color(Mat& pic, float x, float y){
	if (x >= 0 && x <= float(pic.cols - 1) && y >= 0 && y <= float(pic.rows - 1)){
		float x1, x2;
		float y1, y2;
		x1 = floor(x);
		x2 = ceil(x);
		y1 = floor(y);
		y2 = ceil(y);

		if (x1 == x){
			if (y1 == y){
				return Vec3f(pic.at<Vec3b>(int(y), int(x)));
			}
			else{
				return (Vec3f(pic.at<Vec3b>(int(y1), int(x))) * (y2 - y) + Vec3f(pic.at<Vec3b>(int(y2), int(x))) * (y - y1));
			}
		}
		else{
			if (y1 == y){
				return (Vec3f(pic.at<Vec3b>(int(y), int(x1))) * (x2 - x) + Vec3f(pic.at<Vec3b>(int(y), int(x2))) * (x - x1));
			}
			else{
				return (x2 - x) * (Vec3f(pic.at<Vec3b>(int(y1), int(x1))) * (y2 - y) + Vec3f(pic.at<Vec3b>(int(y2), int(x1))) * (y - y1))
					 + (x - x1) * (Vec3f(pic.at<Vec3b>(int(y1), int(x2))) * (y2 - y) + Vec3f(pic.at<Vec3b>(int(y2), int(x2))) * (y - y1));
			}
		}
	}
	return Vec3f(0, 0, 0);
}

void PHOTO_STITCHING::start_stitching(){
	srand(NULL);
	for (int i = 0; i < mImageBuffer.size() - 1; i++){
		doRANSAC(mImageBuffer.size() - 1, i);
	}
	/*for (int i = 0; i < mImageBuffer.size() - 1; i++){
		image_warping(mImageBuffer[mImageBuffer.size() - 1], i);
	}*/
	image_warping(mImageBuffer[mImageBuffer.size() - 1]);
	imshow("image_stitching", mResult);
}

void PHOTO_STITCHING::show(){
	for (int i = 0; i < mImageBuffer.size(); i++){
		imshow("feature_point" + to_string(i), mFeature[i]);
	}
}

void PHOTO_STITCHING::show_kTree(){
	for (int i = 0; i < mKDTree.size(); i++){
		cout << "tree" + to_string(i) + "-----------------------" << endl;
		for (int j = 0; j < mKDTree[i].size(); j++){
			cout << "[" << j << "] parent: " << mKDTree[i][j].parent << ", left: " << mKDTree[i][j].left << ", right: " << mKDTree[i][j].right << endl;
		}
	}
}

void PHOTO_STITCHING::show_nice_match(){
	for (int i = 0; i < mKDTree.size() - 1; i++){
		cout << "nice match between picture" + to_string(i) + " and picture" + to_string(mKDTree.size() - 1) + "..." << endl;
		vector<KeyPoint> temp;
		Mat result;
		for (int j = 0; j < mKDTree[i].size(); j++){
			if (mKDTree[i][j].nice_match == true){
				cout << "node" + to_string(j) + ": nice!" << endl;
				temp.push_back(mKeypoints[i][j]);
			}
			/*else{
				cout << "node" + to_string(j) + ": bad!" << endl;
			}*/
		}
		drawKeypoints(mImageBuffer[i], temp, result, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("feature point" + to_string(i), result);
		cout << endl;
		//system("pause");
	}
}