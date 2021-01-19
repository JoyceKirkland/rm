/*
 * @Author: joyce
 * @Date: 2020-12-02 22:21:50
 * @LastEditTime: 2020-12-02 22:25:36
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

#include <iostream>

int main()
{
    Mat frame=imread("/home/joyce/image/1.jpg");
    imshow("frame",frame);
    waitKey(0);
    return 0;
}