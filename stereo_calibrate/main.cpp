#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

const cv::String keys =
    "{help h ?    |      | Show help message}"
    "{@images      |      | Directory containing stereo images}"
    "{@output      |<none>| Path to the output .yml file}";

void readImages(const std::string& dirPath, std::vector<std::string>& imagePaths) {
    std::string adjustedDir = dirPath;
    if (!adjustedDir.empty() && adjustedDir.back() != '/') {
        adjustedDir += "/";
    }
    cv::glob(adjustedDir + "*.jpg", imagePaths);
    for (const auto& path : imagePaths) {
        std::cout << "Found image: " << path << std::endl;
    }
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Stereo Calibration Program");

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    cv::String imageDir = parser.get<cv::String>("@images");
    cv::String outputPath = parser.get<cv::String>("@output");

    if (!parser.check()) {
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (imageDir.empty() || outputPath.empty()) {
        std::cerr << "Error: Image directory or output file not provided." << std::endl;
        return EXIT_FAILURE;
    }

    const cv::Size CheckerBoardSize = {7, 5};
    const double SquareSize = 0.02875;
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60, 1e-6);

    std::vector<std::string> imagePaths;
    readImages(imageDir, imagePaths);

    if (imagePaths.empty()) {
        std::cerr << "Error: No images found in the directory " << imageDir << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CheckerBoardSize.height; ++i) {
        for (int j = 0; j < CheckerBoardSize.width; ++j) {
            objp.emplace_back(j * SquareSize, i * SquareSize, 0);
        }
    }

    cv::Size imageSize;
    bool imageSizeSet = false;

    for (const auto& imagePath : imagePaths) {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error reading image: " << imagePath << std::endl;
            continue;
        }

        // Split the image into left and right halves
        int halfWidth = img.cols / 2;
        cv::Mat imgLeft = img(cv::Rect(0, 0, halfWidth, img.rows));
        cv::Mat imgRight = img(cv::Rect(halfWidth, 0, halfWidth, img.rows));

        if (!imageSizeSet) {
            imageSize = imgLeft.size();
            imageSizeSet = true;
        }

        std::vector<cv::Point2f> cornersLeft, cornersRight;
        bool foundLeft = cv::findChessboardCorners(imgLeft, CheckerBoardSize, cornersLeft, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool foundRight = cv::findChessboardCorners(imgRight, CheckerBoardSize, cornersRight, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (foundLeft && foundRight) {
            cv::cornerSubPix(imgLeft, cornersLeft, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::cornerSubPix(imgRight, cornersRight, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            imagePointsLeft.push_back(cornersLeft);
            imagePointsRight.push_back(cornersRight);
            objectPoints.push_back(objp);
        } else {
            std::cerr << "Chessboard corners not found in image: " << imagePath << std::endl;
        }
    }

    if (objectPoints.empty()) {
        std::cerr << "Error: No valid chessboard patterns were found." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat cameraMatrixLeft = cv::initCameraMatrix2D(objectPoints, imagePointsLeft, imageSize, 0);
    cv::Mat cameraMatrixRight = cv::initCameraMatrix2D(objectPoints, imagePointsRight, imageSize, 0);
    cv::Mat distCoeffsLeft = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat distCoeffsRight = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat R, T, E, F;

    cv::stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight,
                        cameraMatrixLeft, distCoeffsLeft,
                        cameraMatrixRight, distCoeffsRight,
                        imageSize,
                        R, T, E, F,
                        cv::CALIB_USE_INTRINSIC_GUESS,
                        criteria);

    cv::FileStorage fs(outputPath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Error opening file for writing: " << outputPath << std::endl;
        return EXIT_FAILURE;
    }

    fs << "LEFT_K" << cameraMatrixLeft;
    fs << "LEFT_D" << distCoeffsLeft;
    fs << "RIGHT_K" << cameraMatrixRight;
    fs << "RIGHT_D" << distCoeffsRight;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;

    fs.release();

    std::cout << "Stereo calibration completed successfully. Results saved to " << outputPath << std::endl;
    return 0;
}
