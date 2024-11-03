
#include "common_code.hpp"

void fsiv_find_min_max_loc_1(cv::Mat const& input,
    std::vector<cv::uint8_t>& min_v, std::vector<cv::uint8_t>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc)
{
    CV_Assert(input.depth() == CV_8U);

    // Split the input image into its channels
    std::vector<cv::Mat> channels;
    cv::split(input, channels);

    // Initialize the min and max vectors
    int num_channels = input.channels();
    min_v.resize(num_channels);
    max_v.resize(num_channels);
    min_loc.resize(num_channels);
    max_loc.resize(num_channels);

    // Loop through each channel
    for (int c = 0; c < num_channels; ++c)
    {
        // Initialize min and max values and their locations for the channel
        uint8_t min_val = 255, max_val = 0;
        cv::Point min_position, max_position;

        // Scan through rows and columns
        for (int row = 0; row < channels[c].rows; ++row)
        {
            for (int col = 0; col < channels[c].cols; ++col)
            {
                uint8_t pixel_val = channels[c].at<uint8_t>(row, col);

                // Update min value and location
                if (pixel_val < min_val)
                {
                    min_val = pixel_val;
                    min_position = cv::Point(col, row);
                }

                // Update max value and location
                if (pixel_val > max_val)
                {
                    max_val = pixel_val;
                    max_position = cv::Point(col, row);
                }
            }
        }

        // Store the results for this channel
        min_v[c] = min_val;
        max_v[c] = max_val;
        min_loc[c] = min_position;
        max_loc[c] = max_position;
    }

    CV_Assert(input.channels() == min_v.size());
    CV_Assert(input.channels() == max_v.size());
    CV_Assert(input.channels() == min_loc.size());
    CV_Assert(input.channels() == max_loc.size());
}


void fsiv_find_min_max_loc_2(cv::Mat const& input,
    std::vector<double>& min_v, std::vector<double>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc)
{
    // Split the input image into its channels
    std::vector<cv::Mat> channels;
    cv::split(input, channels);

    // Initialize the output vectors
    int num_channels = input.channels();
    min_v.resize(num_channels);
    max_v.resize(num_channels);
    min_loc.resize(num_channels);
    max_loc.resize(num_channels);

    // Loop through each channel
    for (int c = 0; c < num_channels; ++c)
    {
        // Use cv::minMaxLoc to find the minimum and maximum values and their locations
        double min_val, max_val;
        cv::Point min_position, max_position;

        cv::minMaxLoc(channels[c], &min_val, &max_val, &min_position, &max_position);

        // Store the results for this channel
        min_v[c] = min_val;
        max_v[c] = max_val;
        min_loc[c] = min_position;
        max_loc[c] = max_position;
    }

    CV_Assert(input.channels() == min_v.size());
    CV_Assert(input.channels() == max_v.size());
    CV_Assert(input.channels() == min_loc.size());
    CV_Assert(input.channels() == max_loc.size());
}

