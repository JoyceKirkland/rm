// #include <iostream>
// #include <sstream>
// #include <iostream>
// #include <fstream>
// #include <algorithm>
// #include <string>

// #include<opencv2/imgproc/imgproc.hpp>
// #include<opencv2/core/core.hpp>
// #include<opencv2/highgui/highgui.hpp>
// #include<librealsense2/rs.hpp>
// #include<librealsense2/rsutil.h>
#include"realsense.h"
#include"rm-eng_version2/configure.h"
#include"serialport.cpp"

// using namespace cv;
// using namespace std;
// using namespace rs2;
Mat hsv;
Mat ele;
Mat mask;
Mat dst;
Mat dst1; 
Mat inrange;
int ele_size=13;
int ele_size_Max=21;

Mat element = getStructuringElement(MORPH_RECT, Size(ele_size, ele_size));
Mat kernel = (Mat_<float>(3, 3) << 0,-1,0,0,4,0,0,-1,0);//目前较稳定，偶然会有些许抖动，基本不受杂质影响
//Mat kernel = (Mat_<float>(3, 3) << 1,1,1,1,-8,1,1,1,1);
//Mat kernel = (Mat_<float>(3, 3) << 0,1,0,1,-4,1,0,1,0);
//Mat kernel = (Mat_<float>(3, 3) << 0,-1,0,-1,4,-1,0,-1,0);
//Mat kernel = (Mat_<float>(3, 3) << -1,1,-1,1,8,-1,-1,1,-1);//能用，会抖
//Mat kernel = (Mat_<float>(3, 3) << -1,-8,1,1,8,-1,-1,-8,1);//目前较稳定，偶然会抖，但是框得不太准
//Mat kernel = (Mat_<float>(3, 3) << 0,-1,0,0,16,0,0,3,0);//目前较稳定，会受到一定杂质影响


float h_w;
float w_h;
int hw_min=41;//长宽比最小阈值//81
int hw_min_Max=100;//长宽比最小阈值上限值
int hw_max=61;//长宽比最大阈值//102
int hw_max_Max=200;//长宽比最大阈值上限值

int wh_min=41;//宽长比最小阈值
int wh_min_Max=100;//宽长比最小阈值上限值
int wh_max=61;//宽长比最大阈值
int wh_max_Max=200;//宽长比最大阈值上限值

//int min_video_distance=66;//背景消除最短距离
int min_video_distance_Max=150;//背景消除最短距离上限值
//int depth_clipping_distance=71;//背景消除最远距离
//int depth_clipping_distance=min_video_distance+24;//背景消除最远距离

int depth_clipping_distance_Max=200;//背景消除最远距离上限值

int canny_th1=180;//20
int canny_th1_Max=300;
int canny_th2=100;//100
int canny_th2_Max=300;

float move_x;
float move_y;
float move_xy;
float focal_depth=0.62;

mineral::mineral(){};
mineral::~mineral(){};

//获取深度像素对应长度单位（米）的换算比例
float mineral:: get_depth_scale(device dev)
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
rs2_stream mineral::find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
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

void mineral::remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale,int min_video_distance,int depth_clipping_distance)
//void mineral::remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale)
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
            auto pixels_distance = 100 * depth_scale * p_depth_frame[depth_pixel_index];

            // Check if the depth value is invalid (<=0) or greater than the threashold
            if (pixels_distance <= min_video_distance || pixels_distance > depth_clipping_distance)
            {
                // Calculate the offset in other frame's buffer to current pixel
                auto offset = depth_pixel_index * other_bpp;

                // Set pixel to "background" color (0x999999)
                std::memset(&p_other_frame[offset], 0x99, other_bpp);
            }
        }
    }
}
RotatedRect mineral::find_rect(Mat frame)    
{
    RotatedRect rect;
    RotatedRect rect1;
    //dst = Mat::zeros(frame.size(), CV_32FC3);
    
    morphologyEx(frame,ele, MORPH_OPEN, element);//形态学开运算
    morphologyEx(ele,ele, MORPH_CLOSE, element);//形态学闭运算
    cvtColor(ele,hsv,COLOR_BGRA2GRAY);
    
    //inRange(hsv,Scalar(26,43,46),Scalar(34,255,255),inrange);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // morphologyEx(inrange,ele, MORPH_OPEN, element);//形态学开运算
    dst1=hsv.clone();
    //imshow("dst1",dst1);
    Canny(dst1,mask,canny_th1,canny_th2,7);//边缘检测
    dst=mask.clone();
    
    filter2D(dst, dst, CV_8UC1, kernel);
    GaussianBlur(dst,dst,Size(7,7),3,3);
    
    
    findContours(dst,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point());//寻找并绘制轮廓
    vector<Moments>mu(contours.size());
    for(int i=0;i<contours.size();i++)//最小外接矩形
    {
        //drawContours(mask,contours,i,Scalar(255),2,8,hierarchy);
        rect=minAreaRect(contours[i]);
        Point2f P[4];
        rect.points(P);
        h_w=(rect.size.height/rect.size.width)*100;
        w_h=(rect.size.width/rect.size.height)*100;
        //cout<<"h:"<<rect.size.height<<endl;
        //cout<<"w:"<<rect.size.width<<endl;
        char _x[20],_y[20],xy_to_center[30];
        sprintf(_x,"x=%0.2f",rect.center.x);
        sprintf(_y,"y=%0.2f",rect.center.y);
        //if((h_w>0.99||h_w<1.01)&&((rect.size.width*rect.size.height)>8000.f))
        //if(((h_w>hw_min&&h_w<hw_max)&&(w_h>wh_min&&w_h<wh_max))&&(rect.size.width*rect.size.height)/100>95.f)
        if((rect.size.width*rect.size.height)/100>70.f)
        {
            rect1=rect;
            for(int j=0;j<=3;j++)
            {
                line(frame,P[j],P[(j+1)%4],Scalar(255,255,255),2);
            }
            
            putText(frame,_x,Point(rect.center.x-20,rect.center.y-20),FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,8);
            putText(frame,_y,Point(rect.center.x-20,rect.center.y-50),FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,8);

            SerialPort::RMserialWrite(rect.center.x,rect.center.y);
            return rect1;
        }
        
    }
    //imshow("mask",ele);
    
}
bool mineral::profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
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
 
void mineral::get_frame()
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
    
    //float depth_clipping_distance = 0.8*100;

    //namedWindow("调试",WINDOW_GUI_EXPANDED);
    //createTrackbar("背景消除最短距离","调试",&min_video_distance,min_video_distance_Max,NULL);
    //createTrackbar("背景消除最远距离","调试",&depth_clipping_distance,depth_clipping_distance_Max,NULL);

    //createTrackbar("ele_size","调试",&ele_size,ele_size_Max,NULL);

    //createTrackbar("长宽比最小值","调试",&hw_min,hw_min_Max,NULL);
    //createTrackbar("长宽比最大值","调试",&hw_max,hw_max_Max,NULL);

    //createTrackbar("宽长比最小值","调试",&wh_min,wh_min_Max,NULL);
    //createTrackbar("宽长比最大值","调试",&wh_max,wh_max_Max,NULL);
    //min_video_distance//depth_clipping_distance
    //createTrackbar("canny_th1","调试",&canny_th1,canny_th1_Max,NULL);
    //createTrackbar("canny_th2","调试",&canny_th2,canny_th2_Max,NULL);


    //double t = (double)cv::getTickCount();//开始计时
    for(;;)
    {
        frameset frameset = pipe.wait_for_frames();  //堵塞程序直到新的一帧捕获
        //frame src=rs2_extract_frame(frameset,)
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

        frameset.get_data();
        if (!aligned_depth_frame || !other_frame)
        {
            continue;
        }

        
        //pip_stream = pip_stream.adjust_ratio({ static_cast<float>(aligned_depth_frame.get_width()),static_cast<float>(aligned_depth_frame.get_height()) });

        //取深度图和彩色图
        color_frame = frameset.get_color_frame();
        depth_frame = frameset.get_depth_frame();
        depth_frame_1 = frameset.get_depth_frame().apply_filter(c);
        //获取宽高
        const int depth_w=aligned_depth_frame.as<video_frame>().get_width();
        const int depth_h=aligned_depth_frame.as<video_frame>().get_height();
        const int color_w=other_frame.as<video_frame>().get_width();
        const int color_h=other_frame.as<video_frame>().get_height();
        const int depth_w_1=depth_frame_1.as<video_frame>().get_width();
        const int depth_h_1=depth_frame_1.as<video_frame>().get_height();
        //cout<<"distance: "<<aligned_depth_frame.get_distance(depth_w/2,depth_h/2)<<"m"<<endl;

        int min_distance=aligned_depth_frame.get_distance(depth_w/2,depth_h/2)*100;

        int max_distance=min_distance+2;

        remove_background(other_frame, aligned_depth_frame, depth_scale,min_distance,max_distance);
        //remove_background(other_frame, aligned_depth_frame, depth_scale);

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
        //distance(find_rect(color_image),color_image,profile);
        //measure_distance(color_image,depth_image,profile,find_rect(color_image));            //自定义窗口大小
        //显示

        //imshow("depth_image",depth_image);
        imshow("调试",color_image);
        //imshow("depth_image_1",depth_image_1);
        //imshow("result",result);
        
        int key = waitKey(1);
        if(char(key) == 27)break;
        //t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();//结束计时
        //int fps = int(1.0 / t);//转换为帧率
        //cout << "FPS: " << fps<<endl;//输出帧率
    }
    //return 0;
}
void mineral::test()
{
    cout<<"testest"<<endl;
}