#include <iostream>
#include <exception>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "common_code.hpp"

// Include header for min/max location functions
void fsiv_find_min_max_loc_2(cv::Mat const& input,
    std::vector<double>& min_v, std::vector<double>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc);

const char * keys =
    "{help h usage ? |      | print this message}"
    "{w              |20    | Wait time (milliseconds) between frames.}"
    "{v              |      | The input is a video file.}"
    "{c              |      | The input is a camera index.}"
    "{@input         |<none>| Input <filename|int>}"
    ;

int main(int argc, char* const* argv)
{
    int retCode = EXIT_SUCCESS;

    try {
        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Show the extremes values and their locations.");
        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

        bool is_video = parser.has("v");
        bool is_camera = parser.has("c");
        int wait = parser.get<int>("w");
        cv::String input = parser.get<cv::String>("@input");

        if (!parser.check())
        {
            parser.printErrors();
            return 0;
        }

        cv::VideoCapture cap;
        if (is_camera)
        {
            int camera_index = std::stoi(input);
            cap.open(camera_index);
        }
        else if (is_video)
        {
            cap.open(input);
        }
        else
        {
            cap.open(input);
        }

        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open the input." << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat frame;
        while (cap.read(frame))
        {
            // Vectors to store min/max values and their locations
            std::vector<double> min_v, max_v;
            std::vector<cv::Point> min_loc, max_loc;

            // Find min and max values for each channel
            fsiv_find_min_max_loc_2(frame, min_v, max_v, min_loc, max_loc);

            // Annotate frame with min/max information
            for (size_t i = 0; i < min_v.size(); ++i)
            {
                // Display minimum values and locations
                cv::circle(frame, min_loc[i], 5, cv::Scalar(255, 0, 0), -1);
                cv::putText(frame, "Min: " + std::to_string(min_v[i]),
                            min_loc[i] + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 0, 0), 1);

                // Display maximum values and locations
                cv::circle(frame, max_loc[i], 5, cv::Scalar(0, 255, 0), -1);
                cv::putText(frame, "Max: " + std::to_string(max_v[i]),
                            max_loc[i] + cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 0), 1);
            }

            // Show the annotated frame
            cv::imshow("Extremes", frame);

            // Break the loop if 'q' is pressed
            if (cv::waitKey(wait) == 'q')
                break;
        }

        // For single image processing, just display it once
        if (!is_video && !is_camera)
        {
            cap >> frame;
            if (!frame.empty())
            {
                std::vector<double> min_v, max_v;
                std::vector<cv::Point> min_loc, max_loc;
                fsiv_find_min_max_loc_2(frame, min_v, max_v, min_loc, max_loc);

                // Annotate the image similarly
                for (size_t i = 0; i < min_v.size(); ++i)
                {
                    cv::circle(frame, min_loc[i], 5, cv::Scalar(255, 0, 0), -1);
                    cv::putText(frame, "Min: " + std::to_string(min_v[i]),
                                min_loc[i] + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 0, 0), 1);

                    cv::circle(frame, max_loc[i], 5, cv::Scalar(0, 255, 0), -1);
                    cv::putText(frame, "Max: " + std::to_string(max_v[i]),
                                max_loc[i] + cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(0, 255, 0), 1);
                }

                cv::imshow("Extremes", frame);
                cv::waitKey(0);  // Wait indefinitely for image
            }
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        retCode = EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception!" << std::endl;
        retCode = EXIT_FAILURE;
    }
    return retCode;
}