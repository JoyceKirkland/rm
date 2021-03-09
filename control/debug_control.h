#ifndef DEBUG_CONTROL_H
#define DEBUG_CONTROL_H

/*---------------------------------------------------*/
#define PB_MODEL_PATH "./py/model/CNN_model1.pb"
/**
  @brief: tensorflow模型路径
  @param: PB_MODEL_PATH pb文件路径
*/

#define CAMERA_PARAM_FILE "./camera/param/cameraParams_infantry4.xml"
/**
  @brief: 相机标定文件路径
*/
/*---------------------------------------------------*/

/*---------------------------------------------------*/
#define ISOPEN_INDUSTRY_CAPTURE 0
/**
  @brief: 是否使用工业相机
  @param: 0     使用工业相机
  @param: 1     使用普通USB相机
*/

#define USB_CAPTURE_DEFULT 0
/**
  @brief: 相机的默认值
  @note: 使用普通USB相机时，Opencv的VideoCapture接口的值
*/

#define CAMERA_EXPOSURETIME 800
#define CAMERA_RESOLUTION_COLS 1280
#define CAMERA_RESOLUTION_ROWS 800
#define CAMERA_RESOLUTION_COLS_FOV ((1280-CAMERA_RESOLUTION_COLS)*0.5)
#define CAMERA_RESOLUTION_ROWS_FOV ((1024-CAMERA_RESOLUTION_ROWS)*0.5)
/**
  @brief: 设置相机的分辨率
  @param: CAMERA_EXPOSURETIME   相机曝光时间
  @param: COLS                  为图像的宽度
  @param: ROWS                  为图像的高度
  @param: FOV                   为图像对应左上角的偏移值
  @note: 这部分相机文档中是写反的　x轴　和　y轴
         偏移值计算为 *** (相机最大分辨率 - 当前设置分辨率)/2 ***
*/
/*---------------------------------------------------*/

#define MY_COLOR 0
/**
  @brief: 选择己方阵营
  @param: 0     不限制颜色
  @param: 1     己方为红色
  @param: 2     己方为蓝色
  @param: 3     串口数据决定
  @note: 强制颜色模式
*/

#define  MY_MODE 2
/**
 * @brief :选择己方模式
 * @param:0     默认模式
 * @param:1     自瞄模式
 * @param:2     神符模式
 * @param:3     SENTRY模式
 * @param:4     BASE模式
 * @param:5     串口数据决定
 * @note: 强制运行模式
 */

/*---------------------------------------------------*/
#define IS_SERIAL_OPEN 1
/**
  @brief: 是否启用串口
  @param: 0     不启用
  @param: 1     启用
*/
#define SET_BANDRATE 115200
//#define SET_BANDRATE 1500000
/**
  @brief: 设置波特率
  @param: 115200    波特率115200
  @param: 1500000   波特率1500000
*/
#define SERIAL_COMMUNICATION_PLAN 0
/**
  @brief: 串口所发送的方案
  @param: 0         二维＋深度
  @param: 1         云台俯仰与偏航角度
*/
#define SHOW_SERIAL_INFORMATION 0
/**
  @brief: 是否打印串口数据信息
  @param: 0     不打印
  @param: 1     打印
*/
/*---------------------------------------------------*/

#define IS_NUMBER_PREDICT_OPEN 0
/**
  @brief: 是否启用 TF 识别装甲板数字
  @param: 0     不启用
  @param: 1     启用
*/

#define IS_KF_PREDICT_ARMOR_OPEN 0
/**
  @brief: 是否启用 KF 预测装甲板
  @param: 0     不启用
  @param: 1     启用
*/

#define ENABLE_ROI_RECT 0
/**
  @brief: 启用ROI_RECT截取图像
  @param: 0     不启用
  @param: 1     启用
*/

#define LOSE_CNT_MAX 4
/**
  @brief: 丢失目标上限次数
*/
/*---------------------------------------------------*/

#define SHOW_OUTPUT_IMG 1
/**
  @brief: 是否显示输出图像
  @param: 0     不显示
  @param: 1     显示
*/
#define SHOW_BIN_IMG 0
/**
  @brief: 是否显示二值图像
  @param: 0     不显示
  @param: 1     显示
*/
#define COUT_FPS 1
/**
  @brief: 是否打印帧率
  @param: 0     不打印
  @param: 1     打印
*/
#define IS_PARAM_ADJUSTMENT 0
/**
  @brief: 是否进入调参模式
  @param: 0     否
  @param: 1     是
*/
#define SHOW_DEBUG_INFORMATION 0
/**
  @brief: 是否打印调试信息
  @param: 0     不打印
  @param: 1     打印
  @note: 包括深度，角度等
*/
#define SHOW_ANGLE_INFORMATION 0
/**
  @brief: 是否打印PNP解算角度信息
  @param: 0     不打印
  @param: 1     打印
*/
#define DRAW_LIGHT 0
/**
  @brief: 是否绘制符合条件的light
  @param: 0     不绘制
  @param: 1     绘制
  @note: 蓝色
*/
#define DRAW_ARMOR_RECT 0
/**
  @brief: 是否绘制符合条件的armor
  @param: 0     不绘制
  @param: 1     绘制
  @note: 青色
*/
/*---------------------------------------------------*/

/*--------------------------能量机关-------------------------*/
//二值化阈值 35
#define THRESHOLD_BUFF 35

//model 1固定模型 0实时测距
#define MODEL 1

//buff-pre(buff_detect.cpp)
#define PRE_ANGLE 20
#define SMALL_LENTH_R 1.2
#define SMALL_PRE_ANGLE 20
#define BIG_LENTH_R 5

//buff-filter(buff_detect.cpp)
#define REVISE 0.1

//buff-model尺寸(solve_pnp.cpp)
#define BULLET_SPEED 25//子弹射速
#define BUFF_BOTTOM_H 519//buff最底装甲板距离地面高度
#define ROBOT_H 400//枪口高度    现在是330~340
#define BUFF_ROBOT_Z 7200//枪口和buff的直线距离    6915.340249311
#define OFFSET_Y_BARREL_PTZ 27.69 //枪管和云台的高度差

#define WIDTH 300
#define HEIGHT 170

//auto_control
// 能量机关自动控制项
//#define NO_FIRE   // 发现新目标射一发子弹
//#define NO_REPEAT_FIRE    // 没击打重复发
#define FIRE_CNT 30             // 越小响应越快
#define RESET_CNT 30            // 丢失目标复位计数 越小响应越快
#define REPEAT_FIRE_TIME 1000   // 重复发射时间，单位ｍｓ,可以修改，根据子弹飞行时间进行确认
#define RESET_ANGLE -10 // 1:-20 else: -10  // 复位绝对角度

/*--------------------------------------能量机关----------------------------------------*/

struct Control_Information{
    int my_color;
    int now_run_mode;
    int serial_plan;
    int armor_size;
    int my_Robot_ID;
};

#define IMG_CENTER (Point2f(CAMERA_RESOLUTION_COLS*0.5, CAMERA_RESOLUTION_ROWS*0.5))

enum color{
    ALL_COLOR,
    RED,
    BLUE,
    /**
      @brief: 描述己方颜色信息
      @param: ALL_COLOR     无颜色信息，两种颜色都识别
      @param: RED           己方为红色
      @param: BLUE          己方为蓝色
    */
};

enum run_mode{
    DEFAULT_MODE,
    SUP_SHOOT,
    ENERGY_AGENCY,
    SENTRY_MODE,
    BASE_MODE,
    /**
      @brief: 描述运行模式信息
      @param: SUP_SHOOT         自瞄模式
      @param: ENERGY_AGENCY     神符模式
      @param: SENTRY_MODE       SENTRY模式
      @param: BASE_MODE         BASE模式
    */
};

enum armor_size{
    BIG_ARMOR,
    SMALL_ARMOR,
    /**
      @brief: 描述装甲板尺寸信息
      @param: BIG_ARMOR             大装甲
      @param: BIG_ARMOR_SIZE_W      大装甲实际宽度
      @param: BIG_ARMOR_SIZE_H      大装甲实际高度
      @param: SMALL_ARMOR_SIZE_W    小装甲实际宽度
      @param: SMALL_ARMOR_SIZE_H    小装甲实际高度
      @param: LIGHT_SIZE_W          灯条实际宽度
      @param: LIGHT_SIZE_H          灯条实际高度
    */
    BIG_ARMOR_SIZE_W = 225,
    BIG_ARMOR_SIZE_H = 55,
    SMALL_ARMOR_SIZE_W = 125,
    SMALL_ARMOR_SIZE_H = 55,
    LIGHT_SIZE_W = 10,
    LIGHT_SIZE_H = 55,
};

enum communicationl_plan{
    COORDINATE,
    ANGLE,
    /**
      @brief: 描述串口通讯方式信息
      @param: COORDINATE    二维坐标
      @param: ANGLE         角度
    */
};

enum Robot_ID{
    HERO = 1,
    ENGINEERING,
    INFANTRY,
    UAV = 6,
    SENTRY,
    /**
      @brief: 描述当前机器人ID信息
      @param: HERO          英雄
      @param: ENGINEERING   工程
      @param: INFANTRY      步兵
      @param: UAV           无人机
      @param: SENTRY        哨兵
    */
};

/*---------------------------------------------------*/
/**
 * @brief Armor param
 * @param RED_ARMOR_GRAY_TH     红色灰度阈值
 * @param RED_ARMOR_COLOR_TH    红色颜色阈值
 * @param BLUE_ARMOR_GRAY_TH    蓝色灰度阈值
 * @param BLUE_ARMOR_COLOR_TH   蓝色颜色阈值
 *
 *
 *
 *
 */
const int RED_ARMOR_GRAY_TH = 40;
const int RED_ARMOR_COLOR_TH = 100;
const int BLUE_ARMOR_GRAY_TH = 80;
const int BLUE_ARMOR_COLOR_TH = 135;

/*---------------------------------------------------*/

#endif // DEBUG_CONTROL_H
