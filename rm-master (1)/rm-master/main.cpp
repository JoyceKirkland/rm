/*
 * @Author: joyce
 * @Date: 2020-12-02 11:20:04
 * @LastEditTime: 2020-12-02 21:58:53
 * @LastEditors: Please set LastEditors
 * @Description:: 
 
#include <iostream>

#include "armor_pre/armor.h"

int main()
{
    armor one;
    one.all_pre();
    return 0;
}*/
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
using namespace cv;
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
Mat img;
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

int min_angle=62;//62
int min_angle_max=90;
int max_angle=103;//103
int max_angle_max=180;


Mat dst;
Mat thre;
Mat _threshold;
Mat element;
Mat mor;
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

vector<vector<Point>> pre(Mat frame,vector<vector<Point>> contours)//预处理
{
    threshold(frame,thre,thresh,maxval,THRESH_BINARY);
    element=getStructuringElement(MORPH_RECT,Size(3,3));
    morphologyEx(thre,mor,MORPH_OPEN,element);
//    Canny(mor,dst,3,9,3);//fps两三百
    findContours(mor,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);//fps300+
    return contours;
}

vector<RotatedRect> find_rect(vector<RotatedRect> rect,vector<vector<Point>> contours,Mat out)//筛选灯条
{
    for(int i=0;i<contours.size();i++)
    {
        if(contours[i].size()>6)
        {
//            drawContours(out,contours,i,Scalar(255,255,0),1,8,hierarchy);
//            rect=fitEllipse(contours[i]);
            RotatedRect box=fitEllipse(contours[i]);
            Point2f P[4];
//            cout<<"P:"<<P[i]<<endl;

            if(((box.size.width/box.size.height)*10<w_h||(box.size.height/box.size.width)*10>h_w)
               &&(abs(box.angle)<min_angle||abs(box.angle>max_angle)))//长宽+角度筛选
            {

                box.points(P);
                for(int j=0;j<=3;j++)
                {
                    line(out,P[j],P[(j+1)%4],Scalar(255,0,0),2);
                }

                rect.push_back(box);
//                cout<<"P:"<<P[i]<<endl;
//                cout<<"rect:"<<rect.size()<<endl;
            }
        }

    }
    return rect;

}

void find_Armor_plate(vector<RotatedRect> rect,Mat out)//寻找装甲板
{
    float angle_diff;//角度差
//    float len_diff;//长度差比值
    float center_x;
    float average_height;//灯条平均长度
    float plate_area;
    RotatedRect left_rect,right_rect;
    char rect_i_x[20],rect_i_y[20];
    char rect_j_x[20],rect_j_y[20];


    if(rect.empty())//数据丢失
    {
        cout<<"error"<<endl;
    }
    for(size_t i=0;i<rect.size();i++)
    {
        for(size_t j=i+1;j<rect.size();j++)
        {
//            sort(rect.begin(),rect.end());
            angle_diff=abs(rect[i].angle-rect[j].angle);//角度差
            center_x=abs(rect[i].center.x-rect[j].center.x);//灯条中心点距离
            average_height=abs((rect[i].size.height+rect[j].size.height)/2);//灯条平均长度
//            cout<<"center_x:"<<center_x<<endl;
//            sprintf(rect_i_x,"rect_i_x:%lf",rect[i].center.x);
//            sprintf(rect_i_y,"rect_i_y:%lf",rect[i].center.y);
//            sprintf(rect_j_x,"rect_j_x:%lf",rect[j].center.x);
//            sprintf(rect_j_y,"rect_j_y:%lf",rect[j].center.y);

//            putText(out,rect_i_x,rect[i].center,FONT_HERSHEY_COMPLEX,0.85,Scalar(255,255,255));
//            putText(out,rect_i_x,rect[i].center+Point2f(0,20),FONT_HERSHEY_COMPLEX,0.85,Scalar(255,255,255));
//            putText(out,rect_j_x,rect[j].center,FONT_HERSHEY_COMPLEX,0.35,Scalar(0,255,255));
//            putText(out,rect_j_x,rect[j].center+Point2f(0,20),FONT_HERSHEY_COMPLEX,0.35,Scalar(0,255,255));

            if(angle_diff<15&&center_x>10&&center_x<300)
            {
                left_rect=rect[i];
                right_rect=rect[j];
                plate_area=abs(average_height*center_x);
//                cout<<"center_x:"<<center_x<<endl;
                if(plate_area>1000)
                {
//                    for(int j=0;j<=3;j++)
                    {
                        rectangle(out,Rect(rect[i].center.x,rect[i].center.y-(rect[i].size.height/2),abs(rect[j].center.x-rect[i].center.x),
                                           abs(average_height)),Scalar(0,0,255),2,0,0);
//                        line(out,Point2f(rect[i].center.x,rect[i].center.y),Point2f(rect[j].center.x,rect[i].center.y),Scalar(255,255,255),2);
                    }
//                    circle(out,Point2f((rect[i].center.x+rect[j].center.x)/2,(rect[i].center.y+rect[j].center.y)/2),5,Scalar(255,255,255),-1,8);
//            cout<<"len_diff:"<<len_diff<<endl;
                }
            }

        }
    }

}

int main()
{

    VideoCapture cap("/home/joyce/workplace/armor_test_avi/avis/armor_1.avi");

//    img = imread("/home/joyce/桌面/color.jpg");
//	if (!img.data || img.channels() != 3)
//		return -1;
     namedWindow("aSd", WINDOW_GUI_EXPANDED);
//     createTrackbar("hmin", "aSd", &hmin, hmin_Max, NULL);
//     createTrackbar("hmax", "aSd", &hmax, hmax_Max, NULL);

//     createTrackbar("smin", "aSd", &smin, smin_Max, NULL);
//     createTrackbar("smax", "aSd", &smax, smax_Max, NULL);

//     createTrackbar("vmin", "aSd", &vmin, vmin_Max, NULL);
//     createTrackbar("vmax", "aSd", &vmax, vmax_Max, NULL);

//     createTrackbar("thresh", "aSd", &thresh, thresh_max, NULL);//二值化阈值
//     createTrackbar("maxval", "aSd", &maxval, maxval_max, NULL);

//     createTrackbar("maxval", "aSd", &color_thresh, color_thresh_max, NULL);
//     createTrackbar("split_reduce", "aSd", &split_reduce, split_reduce_max, NULL);//通道缩减
     createTrackbar("min_angle", "aSd", &min_angle, min_angle_max, NULL);//
     createTrackbar("max_angle", "aSd", &max_angle, max_angle_max, NULL);//

     createTrackbar("h_w", "aSd", &h_w, h_w_max, NULL);//
     createTrackbar("w_h", "aSd", &w_h, w_h_max, NULL);//

    for(;;)
    {
        cap>>img;
        double t = (double)cv::getTickCount();

//    cvtColor(img, hsv, COLOR_BGR2GRAY);
//    GaussianBlur(hsv,hsv,Size(7,7),3);
        vector<Mat>channel;
        vector<vector<Point>> contours;
        vector<RotatedRect>find_minrect;
//    imshow("hsv",hsv);
        split(img,channel);//red
        channel[2]=channel.at(2)/split_reduce;//red
////    Canny(mor,dst,3,9,3);//fps两三百//预处理备用
//    pre(channel[2],contours);//预处理
//        find_rect(find_minrect,pre(channel[2],contours),img);
        find_Armor_plate(find_rect(find_minrect,pre(channel[2],contours),img),img);
//        find_Armor_plate(find_minrect);


        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        int fps = int(1.0 / t);
        cout << "FPS: " << fps<<endl;
        imshow("img",img);
//    imshow("dst",dst);
//    imshow("mor",mor);
        waitKey(100);
}
    return 0;
}
