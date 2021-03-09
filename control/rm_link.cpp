#include "rm_link.h"

/**
 * @brief Construct a new rm vision init::rm vision init object
 *
 */
RM_Vision_Init::RM_Vision_Init():capture(USB_CAPTURE_DEFULT),cap(ISOPEN_INDUSTRY_CAPTURE){
}

/**
 * @brief Destroy the rm vision init::rm vision init object
 *
 */
RM_Vision_Init::~RM_Vision_Init(){
}

void RM_Vision_Init::updateControl_information(int arr[REC_BUFF_LENGTH]){
    /*更新颜色信息　update color*/
    switch (arr[1]) {
    case RED:
        g_Ctrl.my_color = RED;
        break;
    case BLUE:
        g_Ctrl.my_color = BLUE;
        break;
    default:
        g_Ctrl.my_color = ALL_COLOR;
        break;
    }

    /*更新模式信息 update mode*/
    switch (arr[2]) {
    case SUP_SHOOT:
        g_Ctrl.now_run_mode = SUP_SHOOT;
        break;
    case ENERGY_AGENCY:
        g_Ctrl.now_run_mode = ENERGY_AGENCY;
        break;
    case SENTRY_MODE:
        g_Ctrl.now_run_mode = SENTRY;
        break;
    case BASE_MODE:
        g_Ctrl.now_run_mode = BASE_MODE;
        break;
    default:
        g_Ctrl.now_run_mode = SUP_SHOOT;
        break;
    }

    /*更新当前机器人ID update Robot ID*/
    switch (arr[3]) {
    case HERO:
        g_Ctrl.my_Robot_ID = HERO;
        break;
    case ENGINEERING:
        g_Ctrl.my_Robot_ID = ENGINEERING;
        break;
    case INFANTRY:
        g_Ctrl.my_Robot_ID = INFANTRY;
        break;
    case UAV:
        g_Ctrl.my_Robot_ID = UAV;
        break;
    case SENTRY:
        g_Ctrl.my_Robot_ID = SENTRY;
        break;
    default:
        g_Ctrl.my_Robot_ID = INFANTRY;
        break;
    }

    /*更新陀螺仪数据 update gyroscope data*/
}

/**
 * @brief 视觉功能执行函数
 *
 */
void RM_Vision_Init::Run(){

    //get Image
    if(cap.isindustryimgInput()){
        src_img = cvarrToMat(cap.iplImage,true);//这里只是进行指针转换，将IplImage转换成Mat类型
    }else{
        capture >> src_img;
    }
#if IS_SERIAL_OPEN == 1
    //读取串口数据
    int ctrl_arr[REC_BUFF_LENGTH];
    SerialPort::RMreceiveData(ctrl_arr);
    updateControl_information(ctrl_arr);
#endif
#if MY_COLOR == 0
    g_Ctrl.my_color = ALL_COLOR;
#elif MY_COLOR == 1
    g_Ctrl.my_color = RED;
#elif MY_COLOR == 2
    g_Ctrl.my_color = BLUE;
#else
    // cout <<"串口决定"<<endl;
#endif

#if MY_MODE == 0
    g_Ctrl.now_run_mode = DEFAULT_MODE
#elif MY_MODE == 1
    g_Ctrl.now_run_mode = SUP_SHOOT;
#elif MY_MODE == 2
    g_Ctrl.now_run_mode = ENERGY_AGENCY;
#elif MY_MODE == 3
    g_Ctrl.now_run_mode = SENTRY_MODE;
#elif MY_MODE == 4
    g_Ctrl.now_run_mode = BASE_MODE;
#else
    // cout <<"串口决定"<<endl;
#endif 

    // if(g_Ctrl.my_color == RED){
    //     cout <<"己方颜色为红色: "<< RED <<endl;
    // } else if(g_Ctrl.my_color == BLUE){
    //     cout <<"己方颜色为蓝色: "<< BLUE <<endl;
    // } else {
    //     cout <<"敌我不分模式: "<< ALL_COLOR <<endl;
    // }
    //选择模式
    switch (g_Ctrl.now_run_mode) {
    case SUP_SHOOT:
    {
        armor.imageProcessing(src_img, g_Ctrl.my_color);
        armor.armorFitted();
        // cout<<"armorFitted mode"<<endl;

    } break;
    case ENERGY_AGENCY:
    {
        buff.buffDetect_Task(src_img);
        // cout<<"Enegry agency mode"<<endl;
    } break;
    default:
    {
        armor.imageProcessing(src_img, g_Ctrl.my_color);
        armor.armorFitted();
    } break;
    }
    cap.cameraReleasebuff();
}

/**
 * @brief 程序退出条件
 *
 * @return true 结束程序
 * @return false 继续执行
 */
bool RM_Vision_Init::is_exit()
{
    int key = waitKey(1);
    if(char(key) == 27)
        return true;
    else
        return false;
}
