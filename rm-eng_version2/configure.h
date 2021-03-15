/*
 * @Author: joyce
 * @Date: 2021-01-21 14:15:25
 * @LastEditTime: 2021-01-21 14:15:26
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */

/*---- RealSense D435深度相机 header files ----*/
#include <librealsense2/rs.h>
#include <librealsense2/rs.hpp>
#include <librealsense2/h/rs_pipeline.h>
#include <librealsense2/h/rs_option.h>
#include <librealsense2/h/rs_frame.h>
#include<librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_frame.hpp>

/*-------------------------------------------*/


/*---- OpenCV header files ----*/
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
/*-----------------------------*/


/*---- Others header files ----*/
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
/*------------------------------*/


/*---- Serial header files ----*/
#include <string.h>
#include <fcntl.h>   //文件控制定义
#include <termios.h> //POSIX终端控制定义
#include <unistd.h>  //UNIX标准定义
#include <errno.h>   //ERROR数字定义
#include <sys/select.h>
/*------------------------------*/

using namespace cv;
using namespace std;
using namespace rs2;