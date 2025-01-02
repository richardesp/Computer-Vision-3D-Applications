#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <fstream>
#include <vector>

struct StereoParams {
    cv::Mat mtxL, distL, mtxR, distR;
    cv::Mat Rot, Trns, Emat, Fmat;
};

void loadStereoParams(const std::string& filename, StereoParams& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open calibration file: " + filename);
    }
    fs["LEFT_K"] >> params.mtxL;
    fs["LEFT_D"] >> params.distL;
    fs["RIGHT_K"] >> params.mtxR;
    fs["RIGHT_D"] >> params.distR;
    fs["R"] >> params.Rot;
    fs["T"] >> params.Trns;
    fs.release();
}

void rectifyStereoImages(const StereoParams& params, cv::Mat& left, cv::Mat& right, cv::Mat& Q) {
    cv::Mat rectL, rectR, projMatL, projMatR;
    cv::Mat mapL1, mapL2, mapR1, mapR2;
    cv::stereoRectify(params.mtxL, params.distL, params.mtxR, params.distR, left.size(),
                      params.Rot, params.Trns, rectL, rectR, projMatL, projMatR, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);
    cv::initUndistortRectifyMap(params.mtxL, params.distL, rectL, projMatL, left.size(), CV_16SC2, mapL1, mapL2);
    cv::initUndistortRectifyMap(params.mtxR, params.distR, rectR, projMatR, right.size(), CV_16SC2, mapR1, mapR2);
    
    cv::remap(left, left, mapL1, mapL2, cv::INTER_LINEAR);
    cv::remap(right, right, mapR1, mapR2, cv::INTER_LINEAR);
}

void writeToOBJ(const std::string& path, const std::vector<cv::Point3f>& points) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + path);
    }
    for (const auto& p : points) {
        file << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }
    file.close();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./stereo_sparse stereo_image.jpg calibration.yml out.obj" << std::endl;
        return EXIT_FAILURE;
    }

    std::string stereoImagePath = argv[1];
    std::string calibrationFilePath = argv[2];
    std::string outputPath = argv[3];

    cv::Mat stereoImage = cv::imread(stereoImagePath, cv::IMREAD_GRAYSCALE);
    if (stereoImage.empty()) {
        std::cerr << "Error reading stereo image: " << stereoImagePath << std::endl;
        return EXIT_FAILURE;
    }

    int halfWidth = stereoImage.cols / 2;
    cv::Mat left = stereoImage(cv::Rect(0, 0, halfWidth, stereoImage.rows));
    cv::Mat right = stereoImage(cv::Rect(halfWidth, 0, halfWidth, stereoImage.rows));

    StereoParams params;
    try {
        loadStereoParams(calibrationFilePath, params);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat Q;
    rectifyStereoImages(params, left, right, Q);

    // Detect keypoints and compute descriptors using AKAZE
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4f, 8);
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    detector->detectAndCompute(left, cv::Mat(), keypointsLeft, descriptorsLeft);
    detector->detectAndCompute(right, cv::Mat(), keypointsRight, descriptorsRight);

    // Match descriptors using BruteForce-Hamming
    std::vector<cv::DMatch> matches;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptorsLeft, descriptorsRight, matches, cv::Mat());

    // Filter matches based on horizontal alignment
    std::vector<cv::DMatch> filteredMatches;
    for (const auto& match : matches) {
        cv::Point2f ptLeft = keypointsLeft[match.queryIdx].pt;
        cv::Point2f ptRight = keypointsRight[match.trainIdx].pt;
        if (std::abs(ptLeft.y - ptRight.y) < 5.0) { // Allow small vertical difference
            filteredMatches.push_back(match);
        }
    }

    // Draw matches before and after filtering
    cv::Mat matchesImage, filteredMatchesImage;
    cv::drawMatches(left, keypointsLeft, right, keypointsRight, matches, matchesImage);
    cv::drawMatches(left, keypointsLeft, right, keypointsRight, filteredMatches, filteredMatchesImage);

    cv::imshow("Matches Before Filtering", matchesImage);
    cv::imshow("Matches After Filtering", filteredMatchesImage);
    cv::waitKey(0);

    // Triangulate matches
    std::vector<cv::Point3f> points;
    for (const auto& match : filteredMatches) {
        cv::Point2f ptLeft = keypointsLeft[match.queryIdx].pt;
        cv::Point2f ptRight = keypointsRight[match.trainIdx].pt;

        float disparity = ptLeft.x - ptRight.x;
        if (disparity > 0) {
            float Z = (params.mtxL.at<double>(0, 0) * cv::norm(params.Trns)) / disparity;
            float X = (ptLeft.x - params.mtxL.at<double>(0, 2)) * Z / params.mtxL.at<double>(0, 0);
            float Y = (ptLeft.y - params.mtxL.at<double>(1, 2)) * Z / params.mtxL.at<double>(0, 0);
            points.emplace_back(X, Y, Z);
        }
    }

    try {
        writeToOBJ(outputPath, points);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Sparse 3D points computed and saved to OBJ file: " << outputPath << std::endl;
    return EXIT_SUCCESS;
}
