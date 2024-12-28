#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

const cv::String keys =
    "{help h ?    |      | Show help message}"
    "{calibrate c |      | Perform camera calibration and save intrinsics.yml}"
    "{@size        |      | Size of the axis to be drawn}"
    "{@intrinsics  |      | Path to the intrinsics.yml file (optional)}"
    "{@video       |<none>| Path to the video file}";

void drawAxis(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec, float axisSize)
{
    std::vector<cv::Point3f> axisPoints = {
        {0, 0, 0},
        {axisSize * 2, 0, 0},
        {0, axisSize * 2, 0},
        {0, 0, -axisSize * 2}};

    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 2); // X-axis in red
    cv::line(image, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 2); // Y-axis in green
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 2); // Z-axis in blue
}

void draw3DModels(cv::Mat &image, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, const cv::Mat &rvec, const cv::Mat &tvec, const cv::Size &boardSize, float squareSize) {
    // Iterate through each square of the chessboard
    for (int i = 0; i < boardSize.height - 1; i++) {
        for (int j = 0; j < boardSize.width - 1; j++) {
            // Only draw cubes on black squares
            if ((i + j) % 2 == 0) {
                // Calculate the position of the square
                std::vector<cv::Point3f> cubePoints = {
                    {j * squareSize, i * squareSize, 0}, 
                    {(j + 1) * squareSize, i * squareSize, 0},
                    {(j + 1) * squareSize, (i + 1) * squareSize, 0},
                    {j * squareSize, (i + 1) * squareSize, 0},
                    {j * squareSize, i * squareSize, -squareSize},
                    {(j + 1) * squareSize, i * squareSize, -squareSize},
                    {(j + 1) * squareSize, (i + 1) * squareSize, -squareSize},
                    {j * squareSize, (i + 1) * squareSize, -squareSize}
                };

                // Project the 3D points to 2D
                std::vector<cv::Point2f> projectedPoints;
                cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

                // Draw the edges of the cube
                std::vector<std::pair<int, int>> edges = {
                    {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
                    {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
                    {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Vertical edges
                };

                for (const auto &edge : edges) {
                    cv::line(image, projectedPoints[edge.first], projectedPoints[edge.second],
                             cv::Scalar(255, 0, 0), 2); // Blue color for edges
                }
            }
        }
    }
}

void calibrateCameraFromImages(const cv::Size &boardSize, float squareSize, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, const std::string &outputFilename)
{
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    std::vector<cv::String> imageFiles;
    cv::glob("../calibration/*.jpg", imageFiles);

    std::cout << "Board size received: " << boardSize << std::endl;

    for (const auto &file : imageFiles)
    {
        cv::Mat image = cv::imread(file, cv::IMREAD_COLOR); // Read in color to allow preprocessing
        if (image.empty())
        {
            std::cerr << "Error: Could not read image " << file << std::endl;
            continue;
        }

        // Detect chessboard corners
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, boardSize, corners);

        if (found)
        {

            // Convert to grayscale for corner detection
            cv::Mat grayImage;
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            grayImage.convertTo(grayImage, CV_8UC3);

            // Enhance corner detection with subpixel refinement
            cv::cornerSubPix(grayImage, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

            // Visual feedback for debugging
            cv::drawChessboardCorners(image, boardSize, corners, found);
            cv::imshow("Chessboard Detection", image);
            cv::waitKey(200); // Pause to visually inspect

            // Prepare object points for 3D calibration
            std::vector<cv::Point3f> obj;
            for (int i = 0; i < boardSize.height; i++)
            {
                for (int j = 0; j < boardSize.width; j++)
                {
                    obj.emplace_back(j * squareSize, i * squareSize, 0);
                }
            }
            objectPoints.push_back(obj);
            imagePoints.push_back(corners);
        }
        else
        {
            std::cerr << "Warning: Chessboard not found in image: " << file << std::endl;
        }
    }

    if (objectPoints.empty() || imagePoints.empty())
    {
        std::cerr << "Error: Could not find chessboard corners in any images." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Perform camera calibration
    cv::calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, cv::noArray(), cv::noArray());

    // Save the camera parameters to a file
    cv::FileStorage fs(outputFilename, cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs.release();

    std::cout << "Camera calibration complete. Parameters saved to " << outputFilename << std::endl;
    cv::destroyWindow("Chessboard Detection"); // Close the debugging window
}

int main(int argc, char **argv)
{
    // Parse command-line arguments
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Camera Calibration and Augmented Reality Example");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if (parser.has("calibrate"))
    {
        cv::Mat cameraMatrix, distCoeffs;
        calibrateCameraFromImages(cv::Size(9 - 1, 6 - 1), 1.0f, cameraMatrix, distCoeffs, "../intrinsics.yml");
        return 0;
    }

    float axisSize = parser.get<float>("@size");
    cv::String intrinsicsPath = parser.get<cv::String>("@intrinsics");
    cv::String videoPath = parser.get<cv::String>("@video");

    if (!parser.check())
    {
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (videoPath.empty())
    {
        std::cerr << "Error: Video file not provided." << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat cameraMatrix, distCoeffs;

    if (!intrinsicsPath.empty())
    {

        // Load camera parameters
        cv::FileStorage fs(intrinsicsPath, cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            std::cerr << "Error: Could not open intrinsics file " << intrinsicsPath << std::endl;
            return EXIT_FAILURE;
        }

        fs["camera_matrix"] >> cameraMatrix;
        fs["distortion_coefficients"] >> distCoeffs;
    }
    else
    {
        // Calibrate camera directly if intrinsics file is not provided
        std::cout << "No intrinsics file provided. Use --calibrate arg previously" << std::endl;
        return EXIT_SUCCESS;
    }

    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file " << videoPath << std::endl;
        return EXIT_FAILURE;
    }

    cv::Size boardSize(9 - 1, 6 - 1); // Chessboard dimensions
    float squareSize = axisSize;      // Size of a square in the calibration pattern

    std::vector<cv::Point3f> objectPoints;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            objectPoints.emplace_back(j * squareSize, i * squareSize, 0);
        }
    }

    cv::Mat frame;
    while (true)
    {
        bool success = cap.read(frame);
        if (!success)
        {
            std::cout << "End of video or error reading frame." << std::endl;
            break;
        }

        std::vector<cv::Point2f> imagePoints;
        bool found = cv::findChessboardCorners(frame, boardSize, imagePoints);

        if (found)
        {
            cv::Mat grayFrame;
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(grayFrame, imagePoints, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

            drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, axisSize);
            draw3DModels(frame, cameraMatrix, distCoeffs, rvec, tvec, boardSize, squareSize);

        }

        cv::imshow("Augmented Reality", frame);

        char key = static_cast<char>(cv::waitKey(30));
        if (key == 27)
        { // ESC key
            std::cout << "Exiting..." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
