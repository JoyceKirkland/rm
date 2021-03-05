/*
 * @Author: joyce
 * @Date: 2021-01-21 12:07:54
 * @LastEditTime: 2021-01-21 17:20:50
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#ifndef realsense.h


#define realsense.h

#include "rm-master/configure.h"
#include"serialport.h"
#endif 
//using namespace cv;
//using namespace std;
//using namespace rs2;

class mineral
{
private:
    Mat frame;
    RotatedRect rect;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    rs2::frame color_frame;
    rs2::frame depth_frame;
    rs2::frame depth_frame_1;

public:
    mineral();
    ~mineral();
    bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);
    rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
    float get_depth_scale(device dev);
    void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale);
    RotatedRect find_rect(Mat frame);
    void get_frame();
    void test();
    

};
