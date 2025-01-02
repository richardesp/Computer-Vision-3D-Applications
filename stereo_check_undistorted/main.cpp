#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

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
    fs["E"] >> params.Emat;
    fs["F"] >> params.Fmat;
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

void drawHorizontalLine(cv::Mat& img, int y) {
    cv::line(img, cv::Point(0, y), cv::Point(img.cols, y), cv::Scalar(0, 0, 255), 1);
}

void onMouse(int event, int x, int y, int, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        cv::Mat* images = static_cast<cv::Mat*>(userdata);
        cv::Mat leftDisplay, rightDisplay;

        images[0].copyTo(leftDisplay);
        images[1].copyTo(rightDisplay);

        drawHorizontalLine(leftDisplay, y);
        drawHorizontalLine(rightDisplay, y);

        cv::Mat combined;
        cv::hconcat(leftDisplay, rightDisplay, combined);
        cv::imshow("Original Stereo Images", combined);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./stereo_checkundistorted stereo_image.jpg stereocalibrationfile.yml" << std::endl;
        return EXIT_FAILURE;
    }

    std::string stereoImagePath = argv[1];
    std::string calibrationFilePath = argv[2];

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

    cv::Mat rectLeft = left.clone();
    cv::Mat rectRight = right.clone();
    rectifyStereoImages(params, rectLeft, rectRight);

    cv::Mat originalCombined, rectifiedCombined;
    cv::hconcat(left, right, originalCombined);
    cv::hconcat(rectLeft, rectRight, rectifiedCombined);

    cv::namedWindow("Original Stereo Images");
    cv::setMouseCallback("Original Stereo Images", onMouse, new cv::Mat[2]{left, right});
    cv::imshow("Original Stereo Images", originalCombined);

    cv::namedWindow("Rectified Stereo Images");
    cv::setMouseCallback("Rectified Stereo Images", onMouse, new cv::Mat[2]{rectLeft, rectRight});
    cv::imshow("Rectified Stereo Images", rectifiedCombined);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}
