/*
 * @Author: joyce
 * @Date: 2021-01-19 14:12:34
 * @LastEditTime: 2021-01-20 22:03:23
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

/* Include the librealsense C header files */
#include <iostream>
#include <librealsense2/rs.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_option.h>
#include <librealsense2/h/rs_frame.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
// 111
//主机
//笔记本//????
using namespace cv;
using namespace std;
using namespace rs2;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     These parameters are reconfigurable                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define STREAM          RS2_STREAM_DEPTH  // rs2_stream is a types of data provided by RealSense device           //
#define FORMAT          RS2_FORMAT_Z16    // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH           640               // Defines the number of columns for each frame or zero for auto resolve//
#define HEIGHT          0                 // Defines the number of lines for each frame or zero for auto resolve  //
#define FPS             30                // Defines the rate of frames per second                                //
#define STREAM_INDEX    0                 // Defines the stream index, used for multiple streams of the same type //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat hsv;
Mat ele;
Mat mask;
Mat dst;
Mat dst1;
Mat inrange;
int ele_size=17;
int ele_size_Max=21;

Mat element = getStructuringElement(MORPH_RECT, Size(ele_size, ele_size));
//Mat kernel = (Mat_<float>(3, 3) << 0,-1,0,0,4,0,0,-1,0);//目前较稳定，偶然会有些许抖动，基本不受杂质影响
//Mat kernel = (Mat_<float>(3, 3) << 1,1,1,1,-8,1,1,1,1);
//Mat kernel = (Mat_<float>(3, 3) << 0,1,0,1,-4,1,0,1,0);
//Mat kernel = (Mat_<float>(3, 3) << 0,-1,0,-1,4,-1,0,-1,0);
//Mat kernel = (Mat_<float>(3, 3) << -1,1,-1,1,8,-1,-1,1,-1);//能用，会抖
Mat kernel = (Mat_<float>(3, 3) << -1,-8,1,1,8,-1,-1,-8,1);//目前较稳定，偶然会抖，但是框得不太准
//Mat kernel = (Mat_<float>(3, 3) << 0,-1,0,0,16,0,0,3,0);//目前较稳定，会受到一定杂质影响


float h_w;
float w_h;
int hw_min=81;//长宽比最小阈值//81
int hw_min_Max=100;//长宽比最小阈值上限值
int hw_max=122;//长宽比最大阈值//102
int hw_max_Max=200;//长宽比最大阈值上限值

int wh_min=81;//宽长比最小阈值
int wh_min_Max=100;//宽长比最小阈值上限值
int wh_max=122;//宽长比最大阈值
int wh_max_Max=200;//宽长比最大阈值上限值

int min_video_distance=69;//背景消除最短距离
int min_video_distance_Max=150;//背景消除最短距离上限值
int depth_clipping_distance=80;//背景消除最远距离
int depth_clipping_distance_Max=200;//背景消除最远距离上限值

int canny_th1=180;//20
int canny_th1_Max=300;
int canny_th2=100;//100
int canny_th2_Max=300;

float move_x;
float move_y;
float move_xy;
float focal_depth=0.62;
void check_error(rs2_error* e)
{
    if (e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs2_get_failed_function(e), rs2_get_failed_args(e));
        printf("    %s\n", rs2_get_error_message(e));
        exit(EXIT_FAILURE);
    }
}

RotatedRect find_rect(Mat frame)    
{
    RotatedRect rect;
    RotatedRect rect1;
    //dst = Mat::zeros(frame.size(), CV_32FC3);
    
    morphologyEx(frame,ele, MORPH_OPEN, element);//形态学开运算
    morphologyEx(ele,ele, MORPH_CLOSE, element);//形态学闭运算
    //cvtColor(ele,hsv,COLOR_BGR2HSV);
    
    inRange(ele,Scalar(0,0,46),Scalar(180,30,200),inrange);
    
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    // morphologyEx(inrange,ele, MORPH_OPEN, element);//形态学开运算
    dst1=inrange.clone();
    imshow("dst1",dst1);
    Canny(dst1,mask,canny_th1,canny_th2,7);//边缘检测
    dst=mask.clone();
    
    filter2D(dst, dst, CV_8UC1, kernel);
    GaussianBlur(dst,dst,Size(7,7),3,3);
    
    imshow("dst",dst);
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
        if(((h_w>hw_min&&h_w<hw_max)&&(w_h>wh_min&&w_h<wh_max))&&(rect.size.width*rect.size.height)/100>95.f)
        {
            rect1=rect;
            for(int j=0;j<=3;j++)
            {
                line(mask,P[j],P[(j+1)%4],Scalar(255,255,255),2);
            }
            putText(frame,_x,Point(rect.center.x-20,rect.center.y-20),FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,8);
            putText(frame,_y,Point(rect.center.x-20,rect.center.y-50),FONT_HERSHEY_PLAIN,2,Scalar(0,255,0),2,8);
            return rect1;
        }
    }
    imshow("mask",mask);
    
}
int main()
{
    rs2_error* e = 0;

    // // Create a context object. This object owns the handles to all connected realsense devices.
    // // The returned object should be released with rs2_delete_context(...)
    rs2_context* ctx = rs2_create_context(RS2_API_VERSION, &e);
    check_error(e);
    

    // /* Get a list of all the connected devices. */
    // // The returned object should be released with rs2_delete_device_list(...)
    rs2_device_list* device_list = rs2_query_devices(ctx, &e);
    check_error(e);

    int dev_count = rs2_get_device_count(device_list, &e);
    check_error(e);
    printf("There are %d connected RealSense devices.\n", dev_count);
    if (0 == dev_count)
        return EXIT_FAILURE;

    // // Get the first connected device
    // // The returned object should be released with rs2_delete_device(...)
    rs2_device* dev = rs2_create_device(device_list, 0, &e);
    check_error(e);

    //print_device_info(dev);

    // // Create a pipeline to configure, start and stop camera streaming
    // // The returned object should be released with rs2_delete_pipeline(...)
    rs2_pipeline* pipeline =  rs2_create_pipeline(ctx, &e);
    check_error(e);

    // // Create a config instance, used to specify hardware configuration
    // // The retunred object should be released with rs2_delete_config(...)
    rs2_config* config = rs2_create_config(&e);
    check_error(e);

    // // Request a specific configuration
    rs2_config_enable_stream(config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, &e);
    check_error(e);

    // Start the pipeline streaming
    // The retunred object should be released with rs2_delete_pipeline_profile(...)
    rs2_pipeline_profile* pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, &e);
    if (e)
    {
        printf("The connected device doesn't support depth streaming!\n");
        exit(EXIT_FAILURE);
    }

    for(;;)
    {

        // This call waits until a new composite_frame is available
        // composite_frame holds a set of frames. It is used to prevent frame drops
        // The returned object should be released with rs2_release_frame(...)
        rs2_frame* frames = rs2_pipeline_wait_for_frames(pipeline, RS2_DEFAULT_TIMEOUT, &e);
        check_error(e);
        frameset framess;
        //rs2_source source;
        // Returns the number of frames embedded within the composite frame
        int num_of_frames = rs2_embedded_frames_count(frames, &e);
        cout<<"i:"<<num_of_frames<<endl;
        //int num__of=framess.size;
        check_error(e);
        //rs2_frame color_src=rs2_create_colorizer(e);
        int i;
        for (i = 0; i < num_of_frames; ++i)
        {
            // The retunred object should be released with rs2_release_frame(...)
            rs2_frame* frame = rs2_extract_frame(frames, i, &e);
            //rs2_intrinsics intri = rs2_get_motion_intrinsics();
            //rs2_frame* frame1= rs2_get_video_stream_intrinsics(pipeline_profile,,e);
            check_error(e);

            // Check if the given frame can be extended to depth frame interface
            // Accept only depth frames and skip other frames
            if (0 == rs2_is_frame_extendable_to(frame, RS2_EXTENSION_DEPTH_FRAME, &e))
            {
                continue;
            }
            // Get the depth frame's dimensions
            int width = rs2_get_frame_width(frame, &e);
            check_error(e);
            int height = rs2_get_frame_height(frame, &e);
            check_error(e);

            // Query the distance from the camera to the object in the center of the image
            float dist_to_center = rs2_depth_frame_get_distance(frame, width/2, height/2, &e);
            check_error(e);

            auto r = rs2_get_frame_data(frame, &e);
            char str_distance[20];
            sprintf(str_distance,"%.3f m",dist_to_center);
            
            Mat color_image(Size(width,height),CV_16UC1,(void*)r,Mat::AUTO_STEP);
            // Print the distance
            printf("The camera is facing an object %.3f meters away.\n", dist_to_center);
            //putText(color_image,(string)str_distance,Point(width/2-40,height/2-40),
            //    FONT_HERSHEY_PLAIN,2,Scalar(255,0,255),2,70);  
            //line(color_image,Point2f(width/2,height/2-20),Point2f(width/2,height/2+20),Scalar(255,255,255),2);
            //line(color_image,Point2f(width/2-20,height/2),Point2f(width/2+20,height/2),Scalar(255,255,255),2);
            //imshow("color_frame",color_image);
            //cvtColor(color_image,mask,COLOR_GRAY2RGB);
            //morphologyEx(color_image,ele, MORPH_OPEN, element);//形态学开运算
            //morphologyEx(ele,ele, MORPH_CLOSE, element);//形态学闭运算
            //imshow("color_image",color_image);
            //imshow("ele",ele);
            //Canny(color_image,mask,20,100,7);//边缘检测
            //imshow("mask",mask);
            //find_rect(color_image);
            rs2_release_frame(frame);

        }

        rs2_release_frame(frames);
        int key = waitKey(1);
        if(char(key) == 27)break;
    }

    // Stop the pipeline streaming
    rs2_pipeline_stop(pipeline, &e);
    //check_error(e);

    // Release resources
    rs2_delete_pipeline_profile(pipeline_profile);
    rs2_delete_config(config);
    rs2_delete_pipeline(pipeline);
    rs2_delete_device(dev);
    rs2_delete_device_list(device_list);
    rs2_delete_context(ctx);

    return EXIT_SUCCESS;
}