// uses OpenCV ver.4.5
// Opens two images and compares them taking into account relative shift between them
// Use several trackbars to dynamically select regions of interest, template regions 
// (to find match), thresholds, minimum defect area etc.

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

const std::string filename_first = "reference.tiff"; // reference image
const std::string filename_second = "compare.tiff"; // image to be compared with reference

// global points and coordinates to be updated dynamically using trackbars
cv::Point ref_TopLeft(20, 100), ref_BotRight(1200, 900);       // margins for reference image
cv::Point templ_TopLeft(1085, 100), templ_BotRight(1185, 400); // margins for template region
int ref_corner_x = 20, templ_corner_x = 1085; // same but as separate coordinates and dimensions 
int ref_corner_y = 100, templ_corner_y = 100;
int ref_width = 1200, templ_width = 100;
int ref_height = 900, templ_height = 300;

// reference crop rectangle (should have size x,y offset to accommodate shift during matching);  
cv::Rect ref_crop_rect(20, 100, 1200, 900); 
// template crop rectangle (should have prominent image feature, should be long along y-axis since y-shift is bigger than x-shift )
cv::Rect templ_crop_rect(1085, 100, 100, 300); 

int threshold_value = 50;        // minimum pixel difference value for thresholding
int max_threshold_value = 255;   // pixel value set after thresholding
int errode_dilate_seed = 0;       
int threshold_defect_area = 300;
cv::RNG rng(12345);              // random generator for color contours visualization

cv::Mat image_first, image_second, copy1, copy2;


void show(const std::string& name, const cv::Mat& img, int xSize = 500, int ySize = 400, int xOffset = 0, int yOffset = 0) {
    cv::namedWindow(name, 0);
    cv::resizeWindow(name, xSize, ySize);
    cv::moveWindow(name, xOffset, yOffset);
    cv::imshow(name, img);
}


void doWork(int, void*) {
    
    // update regions
    ref_TopLeft.x = ref_corner_x; ref_TopLeft.y = ref_corner_y;
    ref_BotRight.x = ref_TopLeft.x + ref_width; ref_BotRight.y = ref_TopLeft.y + ref_height;
    
    // reference rectangle
    cv::rectangle(image_first, ref_TopLeft, ref_BotRight, cv::Scalar(255, 0, 0), 2, cv::LINE_8); 
    templ_TopLeft.x = templ_corner_x; templ_TopLeft.y = templ_corner_y;
    templ_BotRight.x = templ_TopLeft.x + templ_width; templ_BotRight.y = templ_TopLeft.y + templ_height;
    
    // template rectangle
    cv::rectangle(image_first, templ_TopLeft, templ_BotRight, cv::Scalar(255, 0, 0), 8, cv::LINE_8); 
    show(filename_first, image_first);
    
    // restore image after drawing selection rectangles
    copy1.copyTo(image_first); 
    
    // update template crop rectangle
    templ_crop_rect.x = templ_corner_x; templ_crop_rect.y = templ_corner_y;
    templ_crop_rect.width = templ_width; templ_crop_rect.height = templ_height;
    
    // Make template for searching match according to the templ_crop_rect
    cv::Mat image_template = image_first(templ_crop_rect);
    cv::Mat image_match_result = cv::Mat::zeros(image_second.size(), image_second.type());
    
    // find match and best match position
    cv::matchTemplate(image_second, image_template, image_match_result, cv::TM_CCORR_NORMED);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc, template_delta_point;
    cv::minMaxLoc(image_match_result, &minVal, &maxVal, &minLoc, &maxLoc);
    template_delta_point = cv::Point(templ_crop_rect.x, templ_crop_rect.y) - maxLoc;
    std::cout << "MAXIMUM MATCH LOCATION = " << maxLoc << ";  DELTA = " << template_delta_point << std::endl;
    
    // update reference crop rectangle
    ref_crop_rect.x = ref_corner_x; ref_crop_rect.y = ref_corner_y;
    ref_crop_rect.width = ref_width; ref_crop_rect.height = ref_height;
    
    // crop image to match according to the match found
    const cv::Rect image_crop_rect(ref_crop_rect.x - template_delta_point.x, ref_crop_rect.y - template_delta_point.y,
        ref_crop_rect.width, ref_crop_rect.height);
    std::cout << "Image_crop_rect = " << image_crop_rect << std::endl;
    
    // draw found matching rectangles for/on the second image 
    cv::rectangle(image_second, ref_TopLeft - template_delta_point, ref_BotRight - template_delta_point, cv::Scalar(255, 0, 0), 2, cv::LINE_8); // found reference rectangle
    cv::rectangle(image_second, maxLoc, cv::Point(maxLoc.x + templ_width, maxLoc.y + templ_height), cv::Scalar(255, 0, 0), 8, cv::LINE_8); // found template rectangle
    show(filename_second, image_second, 500, 400, 550, 0);
    
    // restore image after drawing matching rectangles
    copy2.copyTo(image_second); 

    // crop two images to the same size for comparison and according to the matchTemplate found result
    cv::Mat img1_croped = image_first(ref_crop_rect);
    cv::Mat img2_croped = image_second(image_crop_rect);
    
    // Find absolute difference between two images
    cv::Mat image_diff = cv::Mat::zeros(img1_croped.size(), img1_croped.type());
    cv::absdiff(img1_croped, img2_croped, image_diff);
    
    // copy for restoration between iterations
    cv::Mat image_diff_copy; image_diff.copyTo(image_diff_copy); 
    
    // make THRESHOLDING in order to compensate for exposure differences
    cv::threshold(image_diff, image_diff, threshold_value, max_threshold_value, 0); // 0 = Binary Type Thresholding
    
    // Erode - dilate operations
    int er_dil_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(er_dil_type, cv::Size(2 * errode_dilate_seed + 1, 2 * errode_dilate_seed + 1), cv::Point(errode_dilate_seed, errode_dilate_seed));
    erode(image_diff, image_diff, element);
    dilate(image_diff, image_diff, element);
    
    // find count of nonzero pixels in difference image
    int nonZeroCount = cv::countNonZero(image_diff);
    std::cout << "Discrepancy Pixels Count = " << nonZeroCount << std::endl;

    // Find continuous contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image_diff, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    // Draw colorful borders around detected contours GREATER THAN A PARTICULAR MINIMUM SIZE
    cv::Mat image_diff_contours = image_diff.clone();
    cv::merge(std::vector<cv::Mat>{image_diff_contours, image_diff_contours, image_diff_contours}, image_diff_contours);
    
    int contours_cnt = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar random_color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        if (cv::contourArea(contours[i]) > threshold_defect_area) {
            ++contours_cnt;
            cv::drawContours(image_diff_contours, contours, (int)i, random_color, 6, cv::LINE_8, hierarchy, 0);
        }
    }
    std::cout << "Count of regions with area > threshold_defect_area = " << contours_cnt << std::endl;
    show("DIFFERENCE", image_diff_contours, 500, 400, 550, 200);
    
    // restore difference image after displaying
    image_diff_copy.copyTo(image_diff); 
}

int main()
{   // Load two input images to be compared
    image_first = cv::imread(filename_first, cv::IMREAD_GRAYSCALE); //or cv::IMREAD_COLOR
    if (image_first.empty()) std::cout << "Couldn't load " << filename_first << std::endl;
    image_second = cv::imread(filename_second, cv::IMREAD_GRAYSCALE); //or cv::IMREAD_COLOR
    if (image_second.empty()) std::cout << "Couldn't load " << filename_second << std::endl;
    
    // copy of first image to restore after drawing selection rectangles
    image_first.copyTo(copy1);
    // copy of second image to restore after drawing matching rectangles
    image_second.copyTo(copy2); 
    
    // first(reference) image window and track-bars for region-of-search selection
    cv::namedWindow(filename_first, 0); 
    cv::createTrackbar("r_x_TL ", filename_first, &ref_corner_x, 1400, doWork);
    cv::createTrackbar("r_y_TL ", filename_first, &ref_corner_y, 1000, doWork);
    cv::createTrackbar("r_x_BR ", filename_first, &ref_width, 1400, doWork);
    cv::createTrackbar("r_y_BR ", filename_first, &ref_height, 1000, doWork);
    
    // second image window and track-bars for template position and size selection
    cv::namedWindow(filename_second, 0);
    cv::createTrackbar("t_x_TL ", filename_second, &templ_corner_x, 1400, doWork);
    cv::createTrackbar("t_y_TL ", filename_second, &templ_corner_y, 1000, doWork);
    cv::createTrackbar("t_x_BR ", filename_second, &templ_width, 1400, doWork);
    cv::createTrackbar("t_y_BR ", filename_second, &templ_height, 1000, doWork);
    
    // difference image window and trac-kbars for thresholding, erode/dilate processing and minimum detectable defect area
    cv::namedWindow("DIFFERENCE", 0); 
    cv::createTrackbar("thrshld ", "DIFFERENCE", &threshold_value, 255, doWork);
    cv::createTrackbar("err/dil ", "DIFFERENCE", &errode_dilate_seed, 50, doWork);
    cv::createTrackbar("min.area ", "DIFFERENCE", &threshold_defect_area, 2000, doWork);
    doWork(0, 0);

    int c = cv::waitKey();
    if (c == 27) return 0;

    return 0;
}