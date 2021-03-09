/**
 * @file rm_link.h
 * @author GCUROBOT-Visual-Group (GCUROBOT_WOLF@163.com)
 * @brief RM 2019 步兵视觉各接口链接头文件
 * @version 1.1
 * @date 2019-05-06
 * @copyright Copyright (c) 2019 GCU Robot Lab. All Rights Reserved.
 */
#ifndef RM_LINK_H
#define RM_LINK_H

#include "configure.h"
#include "debug_control.h"
#include "camera/rm_videocapture.h"
#include "serial/serialport.h"
#include "armor/rm_armorfitted.h"
#include "detect_buff/buff_detect.h"

class RM_Vision_Init
{
public:
    RM_Vision_Init();
    ~RM_Vision_Init();
    void Run();
    bool is_exit();
    void updateControl_information(int arr[REC_BUFF_LENGTH]);

    Control_Information g_Ctrl;
    /** Camera Srart **/
    cv::VideoCapture capture;
    RM_VideoCapture cap;
    /** Camera Srart **/

    /** param initial **/
    cv::Mat src_img;
    /** param initial **/

    /** function initial **/
    RM_ArmorFitted armor;
    BuffDetector buff;
    /** function initial **/

    int th;
    int energy_refresh_count = 0;
};

#endif // RM_LINK_H
