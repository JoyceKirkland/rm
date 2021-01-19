/*
 * @Author: joyce
 * @Date: 2020-12-01 20:38:16
 * @LastEditTime: 2020-12-02 22:17:23
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

#define DATA_ADJUST_MODE 1
/**
 * @brief:是否开启调参模式 
 * @param 1 开启
 * @param 0 关闭
 */


class armor
{
private:
    Mat frame;//输入图像
    vector<Mat>channel;//用于通道分离
    vector<vector<Point>>contours;//找灯条用的点集
    vector<RotatedRect>find_light;//找到的灯条
    vector<vector<RotatedRect>>find_plate;//找到的装甲板
    Point2f rect_center;//找到的装甲板中心点
    
public:

    armor();
    ~armor();
    void pre();//预处理
    void find_rect();//筛选灯条
    void find_Armor_plate();//寻找装甲板
    void all_pre();//所有处理


};





