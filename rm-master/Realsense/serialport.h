/*
 * @Author: joyce
 * @Date: 2021-03-04 21:48:10
 * @LastEditTime: 2021-03-05 14:55:55
 * @LastEditors: Please set LastEditors
 * @Description:: 
 */
#ifndef SERIALPORT_H
#define SERIALPORT_H

// #include"configure    dsfds.h"
// #include "rm-master/configure.h"
#include "rm-master/configure.h"
//#include "control/debug_control.h"

/** ------------------------------------------------------**
  @brief: 串口部分参数
*/
/** Serious **/
#define WRITE_BUFF_LENGTH 20//19
/**
  @brief: 写入串口的Buff长度
  */

#define CRC_BUFF_LENGTH 16//15
/**
  @brief: 进入CRC校验的Buff长度
  */

#define REC_BUFF_LENGTH 5
/**
 * @brief: 接收的Buff长度
 */

#define PARMODE_CRC_1
/**
  @brief: 有无CRC校验
  @param: PARMODE_CRC_0  无CRC校验位
  @param: PARMODE_CRC_1  有CRC校验位
*/
/** ------------------------------------------------------**/

class SerialPort
{
public:
    //串口标志量
    static int fd;
    static char g_write_buf[WRITE_BUFF_LENGTH];
    static char g_CRC_buf[CRC_BUFF_LENGTH];
    static char g_rec_buf[REC_BUFF_LENGTH];

public:
    /**              函数声明            **/
    SerialPort();
    ~SerialPort();

    static uint8_t Checksum_CRC8(char *buf,uint16_t len);
    //自定义串口发送
    // static void RMserialWrite(int _yaw,int yaw,int _pitch,int pitch,int depth,int data_type,int is_shooting);
    static void RMserialWrite(float rect_x,float rect_y);

    //接收并处理串口数据
    static void RMreceiveData(int arr[REC_BUFF_LENGTH]);

};

const unsigned char CRC8Tab[300]=
{
    0,94,188,226,97,63,221,131,194,156,126,32,163,253,31,65,
    157,195,33,127,252,162,64,30, 95,1,227,189,62,96,130,220,
    35,125,159,193,66,28,254,160,225,191,93,3,128,222,60,98,
    190,224,2,92,223,129,99,61,124,34,192,158,29,67,161,255,
    70,24,250,164,39,121,155,197,132,218,56,102,229,187,89,7,
    219,133,103,57,186,228,6,88,25,71,165,251,120,38,196,154,
    101,59,217,135,4,90,184,230,167,249,27,69,198,152,122,36,
    248,166,68,26,153,199,37,123,58,100,134,216,91,5,231,185,
    140,210,48,110,237,179,81,15,78,16,242,172,47,113,147,205,
    17,79,173,243,112,46,204,146,211,141,111,49,178,236,14,80,
    175,241,19,77,206,144,114,44,109,51,209,143,12,82,176,238,
    50,108,142,208,83,13,239,177,240,174,76,18,145,207,45,115,
    202,148,118,40,171,245,23,73,8,86,180,234,105,55,213,139,
    87,9,235,181,54,104,138,212,149,203, 41,119,244,170,72,22,
    233,183,85,11,136,214,52,106,43,117,151,201,74,20,246,168,
    116,42,200,150,21,75,169,247,182,232,10,84,215,137,107,53
};

#endif // SERIALPORT_H
