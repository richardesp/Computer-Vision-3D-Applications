#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_code.hpp"

void fsiv_compute_derivate(cv::Mat const &img, cv::Mat &dx, cv::Mat &dy, int g_r,
                           int s_ap)
{
    CV_Assert(img.type() == CV_8UC1);
    // TODO
    // Remember: if g_r > 0 apply a previous Gaussian Blur operation with kernel size 2*g_r+1.
    // Hint: use Sobel operator to compute derivate.

    // Apply Gaussian blur if g_r > 0
    if (g_r > 0)
    {
        int kernel_size = 2 * g_r + 1; // Calculate kernel size
        cv::GaussianBlur(img, img, cv::Size(kernel_size, kernel_size), 0);
    }

    cv::Sobel(img, dx, CV_32F, 1, 0, s_ap); // Derivative in x-direction
    cv::Sobel(img, dy, CV_32F, 0, 1, s_ap); // Derivative in y-direction

    //
    CV_Assert(dx.size() == img.size());
    CV_Assert(dy.size() == dx.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);
}

void fsiv_compute_gradient_magnitude(cv::Mat const &dx, cv::Mat const &dy,
                                     cv::Mat &gradient)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);

    // TODO
    // Hint: use cv::magnitude.
    cv::magnitude(dx, dy, gradient);
    //

    CV_Assert(gradient.size() == dx.size());
    CV_Assert(gradient.type() == CV_32FC1);
}

void fsiv_compute_gradient_histogram(cv::Mat const &gradient, int n_bins, cv::Mat &hist, float &max_gradient)
{
    // TODO
    // Hint: use cv::minMaxLoc to get the gradient range {0, max_gradient}

    // Find the maximum gradient value
    double min_val, max_val;
    cv::minMaxLoc(gradient, &min_val, &max_val);
    max_gradient = static_cast<float>(max_val);

    // Ensure the maximum gradient is positive
    CV_Assert(max_gradient > 0.0);

    // Initialize the histogram with n_bins rows and 1 column
    hist = cv::Mat::zeros(n_bins, 1, CV_32F);

    // Compute the bin width
    float bin_width = max_gradient / n_bins;

    // Populate the histogram
    for (int y = 0; y < gradient.rows; ++y)
    {
        for (int x = 0; x < gradient.cols; ++x)
        {
            float value = gradient.at<float>(y, x);

            // Compute bin index
            int bin_idx = static_cast<int>(value / bin_width);

            // Ensure edge values fall into the correct bin
            if (value == max_gradient)
            {
                bin_idx = n_bins - 1;
            }
            else
            {
                bin_idx = std::min(bin_idx, n_bins - 1);
            }

            hist.at<float>(bin_idx) += 1.0f;
        }
    }

    hist.at<float>(hist.rows - 1) -= 1.0f;

    //
    CV_Assert(max_gradient > 0.0);
    CV_Assert(hist.rows == n_bins);
}

int fsiv_compute_histogram_percentile(cv::Mat const &hist, float percentile)
{
    CV_Assert(percentile >= 0.0 && percentile <= 1.0);
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    int idx = -1;
    // TODO
    // Hint: use cv::sum to compute the histogram area.
    // Remember: The percentile p is the first i that sum{h[0], h[1], ..., h[i]} >= p

    // Compute the total area of the histogram
    float total_area = cv::sum(hist)[0];
    CV_Assert(total_area > 0.0); // Ensure the histogram is non-empty

    // Compute the cumulative sum to find the percentile
    float cumulative_sum = 0.0f;

    for (int i = 0; i < hist.rows; ++i)
    {
        cumulative_sum += hist.at<float>(i);

        // Check if the cumulative sum meets or exceeds the target percentile
        if (cumulative_sum / total_area >= percentile)
        {
            idx = i;
            break;
        }
    }

    // Handle the case where percentile == 1.0
    if (percentile == 1.0)
    {
        idx = hist.rows - 1; // Assign the last index explicitly
    }

    //
    CV_Assert(idx >= 0 && idx < hist.rows);
    CV_Assert(idx == 0 || cv::sum(hist(cv::Range(0, idx), cv::Range::all()))[0] / cv::sum(hist)[0] < percentile);
    CV_Assert(cv::sum(hist(cv::Range(0, idx + 1), cv::Range::all()))[0] / cv::sum(hist)[0] >= percentile);
    return idx;
}

float fsiv_histogram_idx_to_value(int idx, int n_bins, float max_value,
                                  float min_value)
{
    CV_Assert(idx >= 0);
    CV_Assert(idx < n_bins);
    float value = 0.0;
    // TODO
    // Remember: Map integer range [0, n_bins) into float
    // range [min_value, max_value)

    value = min_value + (idx * (max_value - min_value)) / n_bins;

    //
    CV_Assert(value >= min_value);
    CV_Assert(value < max_value);
    return value;
}

void fsiv_percentile_edge_detector(cv::Mat const &gradient, cv::Mat &edges,
                                   float th, int n_bins)
{
    CV_Assert(gradient.type() == CV_32FC1);

    // TODO
    // Remember: user other fsiv_xxx to compute histogram and percentiles.
    // Remember: map histogram range {0, ..., n_bins} to the gradient range
    // {0.0, ..., max_grad}
    // Hint: use "operator >=" to threshold the gradient magnitude image.

    cv::Mat hist;
    float max_gradient = 0.0f;

    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);

    int threshold_idx = fsiv_compute_histogram_percentile(hist, th);

    float threshold_value = fsiv_histogram_idx_to_value(threshold_idx, n_bins, max_gradient, 0.0f);

    edges = (gradient >= threshold_value);

    edges.convertTo(edges, CV_8UC1, 255.0);

    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_otsu_edge_detector(cv::Mat const &gradient, cv::Mat &edges)
{
    CV_Assert(gradient.type() == CV_32FC1);

    // TODO
    // Hint: normalize input gradient into rango [0, 255] to use
    // cv::threshold properly.
    //

    cv::Mat normalized_gradient;
    double min_val, max_val;
    cv::minMaxLoc(gradient, &min_val, &max_val);                       // Find the min and max values
    gradient.convertTo(normalized_gradient, CV_8UC1, 255.0 / max_val); // Scale to [0, 255]

    double otsu_threshold = cv::threshold(normalized_gradient, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_canny_edge_detector(cv::Mat const &dx, cv::Mat const &dy, cv::Mat &edges,
                              float th1, float th2, int n_bins)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(th1 < th2);

    // TODO
    // Hint: convert the intput derivatives to CV_16C1 to be used with canny.
    // Remember: th1 and th2 are given as percentiles so you must transform to
    //           gradient range to be used in canny method.
    // Remember: we compute gradients with L2_NORM so we must indicate this in
    //           the canny method too.

    cv::Mat gradient_magnitude;
    cv::magnitude(dx, dy, gradient_magnitude); // Compute L2 norm of the gradient

    cv::Mat hist;
    float max_gradient = 0.0f;
    fsiv_compute_gradient_histogram(gradient_magnitude, n_bins, hist, max_gradient);

    int th1_idx = fsiv_compute_histogram_percentile(hist, th1);
    int th2_idx = fsiv_compute_histogram_percentile(hist, th2);

    float th1_value = fsiv_histogram_idx_to_value(th1_idx, n_bins, max_gradient, 0.0f);
    float th2_value = fsiv_histogram_idx_to_value(th2_idx, n_bins, max_gradient, 0.0f);

    cv::Mat dx_16s, dy_16s;
    dx.convertTo(dx_16s, CV_16SC1);
    dy.convertTo(dy_16s, CV_16SC1);

    cv::Canny(dx_16s, dy_16s, edges, th1_value, th2_value, true); // Use L2 norm

    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == dx.size());
}

void fsiv_compute_ground_truth_image(cv::Mat const &consensus_img,
                                     float min_consensus, cv::Mat &gt)
{
    //! TODO
    // Hint: use cv::normalize to normalize consensus_img into range (0, 100)
    // Hint: use "operator >=" to threshold the consensus image.

    cv::Mat normalized_consensus;
    cv::normalize(consensus_img, normalized_consensus, 0, 100, cv::NORM_MINMAX, CV_32F);

    // Threshold the normalized consensus image
    gt = (normalized_consensus >= min_consensus);

    // Convert the binary image to 8-bit format
    gt.convertTo(gt, CV_8UC1, 255.0);

    //
    CV_Assert(consensus_img.size() == gt.size());
    CV_Assert(gt.type() == CV_8UC1);
}

void fsiv_compute_confusion_matrix(cv::Mat const &gt, cv::Mat const &pred, cv::Mat &cm)
{
    CV_Assert(gt.type() == CV_8UC1);
    CV_Assert(pred.type() == CV_8UC1);
    CV_Assert(gt.size() == pred.size());

    // TODO
    // Remember: a edge detector confusion matrix is a 2x2 matrix where the
    // rows are ground truth {Positive: "is edge", Negative: "is not edge"} and
    // the columns are the predictions labels {"is edge", "is not edge"}
    // A pixel value means edge if it is <> 0, else is a "not edge" pixel.

    cm = cv::Mat::zeros(2, 2, CV_32FC1);

    // Iterate through the ground truth and prediction images
    for (int y = 0; y < gt.rows; ++y)
    {
        for (int x = 0; x < gt.cols; ++x)
        {
            // Determine ground truth and predicted labels
            bool is_edge_gt = (gt.at<uchar>(y, x) != 0);     // Ground truth edge
            bool is_edge_pred = (pred.at<uchar>(y, x) != 0); // Predicted edge

            // Update confusion matrix
            if (is_edge_gt && is_edge_pred)
                cm.at<float>(0, 0) += 1.0f; // True Positive (TP)
            else if (is_edge_gt && !is_edge_pred)
                cm.at<float>(0, 1) += 1.0f; // False Negative (FN)
            else if (!is_edge_gt && is_edge_pred)
                cm.at<float>(1, 0) += 1.0f; // False Positive (FP)
            else
                cm.at<float>(1, 1) += 1.0f; // True Negative (TN)
        }
    }

    //
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cv::abs(cv::sum(cm)[0] - (gt.rows * gt.cols)) < 1.0e-6);
}

float fsiv_compute_sensitivity(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float sensitivity = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix

    float TP = cm.at<float>(0, 0);
    float FN = cm.at<float>(0, 1);

    // Sensitivity = TP / (TP + FN)
    if ((TP + FN) > 0.0f) // Avoid division by zero
    {
        return TP / (TP + FN);
    }
    else
    {
        return 0.0f; // Sensitivity is undefined if no positive ground truths
    }
}

float fsiv_compute_precision(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float precision = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix

    float TP = cm.at<float>(0, 0);
    float FP = cm.at<float>(1, 0);

    // Precision = TP / (TP + FP)
    if ((TP + FP) > 0.0f) // Avoid division by zero
    {
        return TP / (TP + FP);
    }
    else
    {
        return 0.0f; // Precision is undefined if no predicted positives
    }
}

float fsiv_compute_F1_score(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float F1 = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix

    float precision = fsiv_compute_precision(cm);
    float recall = fsiv_compute_sensitivity(cm);

    // Compute F1 Score
    if (precision + recall > 0.0f) // Avoid division by zero
    {
        return 2.0f * (precision * recall) / (precision + recall);
    }
    else
    {
        return 0.0f; // F1 score is undefined if both precision and recall are zero
    }

    //
    return F1;
}
