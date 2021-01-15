#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>

using namespace cv;
using namespace std;
using namespace rs2;
Mat hsv;
Mat ele;
Mat mask;
Mat dst;
Mat inrange;
Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));

int h_w;
int w_h;

//获取深度像素对应长度单位（米）的换算比例
float get_depth_scale(device dev)
{
    for (sensor& sensor : dev.query_sensors()) //检查设备的传感器
    {
        if (depth_sensor dpt = sensor.as<depth_sensor>()) //检查传感器是否为深度传感器
        {
            return dpt.get_depth_scale();
        }
    }
    throw runtime_error("Device does not have a depth sensor");
}
//深度图对齐到彩色图函数
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
    //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
    //We prioritize color streams to make the view look better.
    //If color is not available, we take another stream that (other than depth)
    rs2_stream align_to = RS2_STREAM_ANY;
    bool depth_stream_found = false;
    bool color_stream_found = false;
    for (rs2::stream_profile sp : streams)
    {
        rs2_stream profile_stream = sp.stream_type();
        if (profile_stream != RS2_STREAM_DEPTH)
        {
            if (!color_stream_found)         //Prefer color
                align_to = profile_stream;

            if (profile_stream == RS2_STREAM_COLOR)
            {
                color_stream_found = true;
            }
        }
        else
        {
            depth_stream_found = true;
        }
    }

    if(!depth_stream_found)
        throw std::runtime_error("No Depth stream available");

    if (align_to == RS2_STREAM_ANY)
        throw std::runtime_error("No stream found to align with Depth");

    return align_to;
}
void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

    int width = other_frame.get_width();
    int height = other_frame.get_height();
    int other_bpp = other_frame.get_bytes_per_pixel();

    #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 0; y < height; y++)
    {
        auto depth_pixel_index = y * width;
        for (int x = 0; x < width; x++, ++depth_pixel_index)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance <= 0.68 || pixels_distance > clipping_dist)
            {
                // Calculate the offset in other frame's buffer to current pixel
                auto offset = depth_pixel_index * other_bpp;

                // Set pixel to "background" color (0x999999)
                std::memset(&p_other_frame[offset], 0x99, other_bpp);
            }
        }
    }
}
RotatedRect find_rect(Mat frame)
{
    RotatedRect rect;
    //dst = Mat::zeros(frame.size(), CV_32FC3);
    
    cvtColor(frame,hsv,COLOR_BGR2HSV);
    inRange(hsv,Scalar(0,0,46),Scalar(180,30,200),inrange);
    imshow("mask",inrange);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    morphologyEx(inrange,ele, MORPH_OPEN, element);//形态学开运算
    //morphologyEx(ele,ele, MORPH_CLOSE, element);//形态学闭运算

    Canny(ele,mask,20,114,3);//边缘检测
    
    findContours(mask,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point());//寻找并绘制轮廓
    
    vector<Moments>mu(contours.size());
    for(int i=0;i<contours.size();i++)//最小外接矩形
    {
        drawContours(mask,contours,i,Scalar(255),2,8,hierarchy);
        rect=minAreaRect(contours[i]);
        Point2f P[4];
        rect.points(P);
        h_w=rect.size.height/rect.size.width;
        w_h=rect.size.width/rect.size.height;

        if((h_w>0.99||h_w<1.01)&&((rect.size.width*rect.size.height)>8000.f))
        {
            for(int j=0;j<=3;j++)
            {
                line(frame,P[j],P[(j+1)%4],Scalar(255,255,255),2);
            }
        }
    }
    
    return rect;
}
void measure_distance(Mat &color,Mat depth,Size range,pipeline_profile profile,RotatedRect RectRange)
{ 
    float depth_scale = get_depth_scale(profile.get_device()); //获取深度像素与现实单位比例（D435默认1毫米）
    Point center(RectRange.center.x,RectRange.center.y);                   //自定义图像中心点
    //Rect RectRange(center.x-range.width/2,center.y-range.height/2,
    //               range.width,range.height);                  //自定义计算距离的范围
    //遍历该范围
    float distance_sum=0;
    int effective_pixel=0;
    
    for(int y=RectRange.center.y-(RectRange.size.height/2);y<RectRange.center.y+(RectRange.size.height/2);y++){
        for(int x=RectRange.center.x-(RectRange.size.width/2);x<RectRange.center.x+(RectRange.size.width/2);x++){
            //如果深度图下该点像素不为0，表示有距离信息
            if(depth.at<uint16_t>(y,x)){
                distance_sum+=depth_scale*depth.at<uint16_t>(y,x);
                effective_pixel++;
            }
        }
    }
    cout<<"遍历完成，有效像素点:"<<effective_pixel<<endl;
    float effective_distance=(distance_sum/effective_pixel)*100;
    cout<<"目标距离："<<effective_distance<<" cm"<<endl;
    char distance_str[30];
    sprintf(distance_str,"the distance is:%f cm",effective_distance);
    //rectangle(color,RectRange,Scalar(0,0,255),2,8);
    putText(color,(string)distance_str,Point(color.cols*0.02,color.rows*0.05),
                FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,8);
}
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
    for (auto&& sp : prev)
    {
        //If previous profile is in current (maybe just added another)
        auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
        if (itr == std::end(current)) //If it previous stream wasn't found in current
        {
            return true;
        }
    }
    return false;
}
 
int main() try
{
    colorizer c;   // 帮助着色深度图像
    pipeline pipe;         //创建数据管道
    //config pipe_config;
    //pipe_config.enable_stream(RS2_STREAM_DEPTH,640,480,RS2_FORMAT_Z16,30);
    //pipe_config.enable_stream(RS2_STREAM_COLOR,640,480,RS2_FORMAT_BGR8,30);
    pipeline_profile profile = pipe.start(); //start()函数返回数据管道的profile
    float depth_scale = get_depth_scale(profile.get_device());
    rs2_stream align_to = find_stream_to_align(profile.get_streams());
    rs2::align align(align_to);

    float depth_clipping_distance = 0.8;

    while (1)
    {
        frameset frameset = pipe.wait_for_frames();  //堵塞程序直到新的一帧捕获
        
        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            profile = pipe.get_active_profile();
            align_to = find_stream_to_align(profile.get_streams());
            align = rs2::align(align_to);
            depth_scale = get_depth_scale(profile.get_device());
        }
        auto processed = align.process(frameset);

        rs2::video_frame other_frame = processed.first(align_to);
        rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

        if (!aligned_depth_frame || !other_frame)
        {
            continue;
        }

        remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);
        //pip_stream = pip_stream.adjust_ratio({ static_cast<float>(aligned_depth_frame.get_width()),static_cast<float>(aligned_depth_frame.get_height()) });

        //取深度图和彩色图
        frame color_frame = frameset.get_color_frame();
        frame depth_frame = frameset.get_depth_frame();
        frame depth_frame_1 = frameset.get_depth_frame().apply_filter(c);
        //获取宽高
        const int depth_w=aligned_depth_frame.as<video_frame>().get_width();
        const int depth_h=aligned_depth_frame.as<video_frame>().get_height();
        const int color_w=other_frame.as<video_frame>().get_width();
        const int color_h=other_frame.as<video_frame>().get_height();
        //const int depth_w_1=depth_frame_1.as<video_frame>().get_width();
        //const int depth_h_1=depth_frame_1.as<video_frame>().get_height();

        //创建OPENCV类型 并传入数据
        Mat depth_image(Size(depth_w,depth_h),
                        CV_16U,(void*)aligned_depth_frame.get_data(),Mat::AUTO_STEP);
        //Mat depth_image_1(Size(depth_w_1,depth_h_1),
        //                  CV_8UC3,(void*)depth_frame_1.get_data(),Mat::AUTO_STEP);
        Mat color_image(Size(color_w,color_h),
                        CV_8UC3,(void*)color_frame.get_data(),Mat::AUTO_STEP);
        cvtColor(color_image,color_image,COLOR_BGR2RGB);
        //实现深度图对齐到彩色图
        //Mat result=align_Depth2Color(depth_image,color_image,profile);
        find_rect(color_image);
        //measure_distance(color_image,result,Size(40,40),profile,find_rect(result));            //自定义窗口大小
        //显示
        imshow("depth_image",depth_image);
        imshow("color_image",color_image);
        //imshow("depth_image_1",depth_image_1);
        //imshow("result",result);
        int key = waitKey(1);
        if(char(key) == 27)break;
    }
    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}