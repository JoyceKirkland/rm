/*
 * @Author: joyce
 * @Date: 2020-12-01 20:20:23
 * @LastEditTime: 2020-12-03 13:34:36
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<string>
#include<algorithm>
#include"armor.h"
using namespace std;
using namespace cv;

armor::armor(){}

armor::~armor(){}

//Mat frame;
Mat bgr;
Mat hsv;
//红：（0，0，255）（112，255，255）
int hmin = 0;
int hmin_Max = 360;
int hmax = 360;
int hmax_Max = 360;

int smin = 46;
int smin_Max = 255;
int smax = 255;
int smax_Max = 255;

int vmin = 230;
int vmin_Max = 255;
int vmax = 255;
int vmax_Max = 255;

int h_w=25;//25
int h_w_max=100;
int w_h=4;//4
int w_h_max=10;//1*10

//筛选灯条的参数
int min_angle=62;//62//筛选灯条角度参数
int min_angle_max=90;
int max_angle=103;//103
int max_angle_max=180;


Mat dst;
Mat thre;//二值化输出图像
Mat _threshold;
Mat element;//开运算值定义
Mat mor;//开运算输出图像
Mat canny;
int thresh=10;
int maxval=200;
int thresh_max=300;
int maxval_max=300;
int my_color;
int color_thresh=0;//46
int color_thresh_max=100;
int split_reduce=11;//颜色缩减
int split_reduce_max=20;

void armor::pre()//预处理
{
    VideoCapture cap("/home/joyce/workplace/armor_test_avi/avis/armor_1.avi");
    if (!cap.isOpened())
    {
        cout<<"cap error"<<endl;
        //exit;
    }
    namedWindow("frame_control",WINDOW_AUTOSIZE);//滑动条窗口
    element=getStructuringElement(MORPH_RECT,Size(3,3));//开运算值
    for(;;)
    {
        double t = (double)cv::getTickCount();
        cap>>frame;

#if DATA_ADJUST_MODE ==1//滑动条
     createTrackbar("split_reduce", "frame_control", &split_reduce, split_reduce_max);//通道缩减

#endif

        split(frame,channel);//通道分离//red
        channel[2]=channel.at(2)/split_reduce;//red
        threshold(channel[2],thre,thresh,maxval,THRESH_BINARY);//二值化
        morphologyEx(thre,mor,MORPH_OPEN,element);//开运算
//    Canny(mor,dst,3,9,3);//fps两三百
        findContours(mor,this->contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);//寻找轮廓
    
    //return contours;
    }
}

void armor::find_rect()//筛选灯条
{
#if DATA_ADJUST_MODE ==1
    createTrackbar("min_angle", "frame_control", &min_angle, min_angle_max);//最小角度
    createTrackbar("max_angle", "frame_control", &max_angle, max_angle_max);//最大角度
    createTrackbar("h_w", "frame_control", &h_w, h_w_max);//长宽比
    createTrackbar("w_h", "frame_control", &w_h, w_h_max);//

#endif

    for(long unsigned int i=0;i<this->contours.size();i++)
    {
        if(this->contours[i].size()>6)
        {
//            drawContours(out,contours,i,Scalar(255,255,0),1,8,hierarchy);
//            rect=fitEllipse(contours[i]);
            RotatedRect box=fitEllipse(this->contours[i]);//所有的旋转矩形//椭圆拟合
            Point2f P[4];

            if(((box.size.width/box.size.height)*10<w_h||(box.size.height/box.size.width)*10>h_w)
               &&(abs(box.angle)<min_angle||abs(box.angle>max_angle)))//长宽+角度筛选
            {

                box.points(P);
                for(int j=0;j<=3;j++)
                {
                    line(frame,P[j],P[(j+1)%4],Scalar(255,0,0),2);
                }

                find_light.push_back(box);
//                cout<<"P:"<<P[i]<<endl;
//                cout<<"rect:"<<rect.size()<<endl;
            }
        }

    }
}

void armor::find_Armor_plate()//寻找装甲板
{
    float angle_diff;//角度差
//    float len_diff;//长度差比值
    float center_x;
    float average_height;//灯条平均长度
    float plate_area;
    RotatedRect left_rect,right_rect;
    char rect_i_x[20],rect_i_y[20];
    char rect_j_x[20],rect_j_y[20];


    if(find_light.empty())//数据丢失
    {
        cout<<"error"<<endl;
    }
    for(size_t i=0;i<find_light.size();i++)
    {
        for(size_t j=i+1;j<find_light.size();j++)
        {
//            sort(rect.begin(),rect.end());
            angle_diff=abs(find_light[i].angle-find_light[j].angle);//角度差
            center_x=abs(find_light[i].center.x-find_light[j].center.x);//灯条中心点距离
            average_height=abs((find_light[i].size.height+find_light[j].size.height)/2);//灯条平均长度

/*#if DATA_ADJUST_MODE ==1

    char find_light_i_x[20],find_light_i_y[20];
    char find_light_j_x[20],find_light_j_y[20];

    cout<<"center_x:"<<center_x<<endl;
    sprintf(find_light_i_x,"find_light_i_x:%f",find_light[i].center.x);
    sprintf(find_light_i_y,"find_light_i_y:%f",find_light[i].center.y);
    sprintf(find_light_j_x,"find_light_j_x:%f",find_light[j].center.x);
    sprintf(find_light_j_y,"find_light_j_y:%f",find_light[j].center.y);

    putText(frame,find_light_i_x,find_light[i].center,FONT_HERSHEY_COMPLEX,0.85,Scalar(255,255,255));
    putText(frame,find_light_i_x,find_light[i].center+Point2f(0,20),FONT_HERSHEY_COMPLEX,0.85,Scalar(255,255,255));
    putText(frame,find_light_j_x,find_light[j].center,FONT_HERSHEY_COMPLEX,0.35,Scalar(0,255,255));
    putText(frame,find_light_j_x,find_light[j].center+Point2f(0,20),FONT_HERSHEY_COMPLEX,0.35,Scalar(0,255,255));

#endif
*/

            if(angle_diff<15&&center_x>10&&center_x<300)
            {
                left_rect=find_light[i];
                right_rect=find_light[j];
                plate_area=abs(average_height*center_x);
//                cout<<"center_x:"<<center_x<<endl;
                if(plate_area>1000)
                {
//                    for(int j=0;j<=3;j++)
                    {
                        rectangle(frame,Rect(find_light[i].center.x,find_light[i].center.y-(find_light[i].size.height/2),
                                           abs(find_light[j].center.x-find_light[i].center.x),
                                           abs(average_height)),Scalar(0,0,255),2,0,0);
//                        line(out,Point2f(rect[i].center.x,rect[i].center.y),Point2f(rect[j].center.x,rect[i].center.y),Scalar(255,255,255),2);
                    }
                    find_plate.push_back(find_light);
//                    circle(out,Point2f((rect[i].center.x+rect[j].center.x)/2,(rect[i].center.y+rect[j].center.y)/2),5,Scalar(255,255,255),-1,8);
//            cout<<"len_diff:"<<len_diff<<endl;
                }
            }

        }
    }
}

void armor::all_pre()
{
    pre();
    find_rect();
    find_Armor_plate();
}


