#include <opencv2/opencv.hpp>
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

void rectifyStereoImages(const StereoParams& params, cv::Mat& left, cv::Mat& right) {
    cv::Mat rectL, rectR, projMatL, projMatR, Q;
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
        std::cerr << "Usage: ./stereo_disparity stereo_image.jpg calibration.yml out.obj" << std::endl;
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

    rectifyStereoImages(params, left, right);

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(192, 25); // Increased disparities and block size
    stereoBM->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    stereoBM->setPreFilterCap(31);
    stereoBM->setTextureThreshold(20);
    stereoBM->setUniquenessRatio(15);
    stereoBM->setSpeckleWindowSize(100);
    stereoBM->setSpeckleRange(32);
    cv::Mat disparity16S, disparity32F;
    stereoBM->compute(left, right, disparity16S);

    disparity16S.convertTo(disparity32F, CV_32F, 1.0 / 16.0);

    cv::imshow("Disparity", disparity32F);
    cv::waitKey(0);

    std::vector<cv::Point3f> points;
    double baseline = cv::norm(params.Trns);
    double focalLength = params.mtxL.at<double>(0, 0);
    double cx = params.mtxL.at<double>(0, 2);
    double cy = params.mtxL.at<double>(1, 2);

    std::cout << baseline << std::endl;
    std::cout << focalLength << std::endl;
    std::cout << cx << std::endl;
    std::cout << cy << std::endl;

    for (int y = 0; y < disparity32F.rows; ++y) {
        for (int x = 0; x < disparity32F.cols; ++x) {

            // Disparity between the left image and the right image
            float d = disparity32F.at<float>(y, x);
            if (d > 10.0) { // Valid disparity
                float Z = (baseline * focalLength) / d;
                float X = (x - cx) * Z / focalLength;
                float Y = (y - cy) * Z / focalLength;
                points.emplace_back(X, Y, Z);
            }
        }
    }

    try {
        writeToOBJ(outputPath, points);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Disparity map computed and saved to OBJ file: " << outputPath << std::endl;
    return EXIT_SUCCESS;
}
