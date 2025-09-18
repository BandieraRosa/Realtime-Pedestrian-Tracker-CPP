/*
 * @Author: BandieraRossa 3132716198@qq.com
 * @Date: 2025-09-16 21:10:13
 * @LastEditors: BandieraRossa 3132716198@qq.com
 * @LastEditTime: 2025-09-19 02:31:36
 * @FilePath: /workspace_test_1/include/Track.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/**
 * @file Track.hpp
 * @brief 代表单个被追踪目标的类的头文件。
 * @details
 * 每个 Track 对象都包含一个卡尔曼滤波器来预测和更新目标的状态，
 * 以及用于管理的元数据（如ID，命中次数等）。
 */

#pragma once

#include <opencv2/video/tracking.hpp>

class Track {
public:
    /**
     * @brief Track 构造函数，用于创建一个新的追踪轨迹。
     * 
     * @param track_id 分配给这个轨迹的唯一ID。
     * @param bbox 第一次检测到该目标时的边界框。
     */
    Track(int track_id, const cv::Rect& bbox);

    /**
     * @brief 使用卡尔曼滤波器预测目标的下一个状态。
     */
    void predict();

    /**
     * @brief 使用新的检测结果更新卡尔曼滤波器。
     * 
     * @param bbox 与该轨迹匹配上的新检测框。
     */
    void update(const cv::Rect& bbox);

    /**
     * @brief 获取卡尔曼滤波器当前估计的状态，并以边界框形式返回。
     * @return cv::Rect 当前估计的边界框。
     */
    cv::Rect get_state_as_bbox() const;

    // --- 轨迹元数据 ---
    int id;                 ///< 轨迹的唯一ID。
    int hits;               ///< 轨迹连续被检测到的次数（命中数）。
    int time_since_update;  ///< 距离上次成功更新（匹配上检测）已经过去的帧数。

private:
    /**
     * @brief 卡尔曼滤波器对象。
     * @details
     * 用于平滑轨迹并预测目标在未被检测到的帧中的位置。
     * 状态向量为8维: [cx, cy, a, h, vx, vy, va, vh]
     *   - cx, cy: 边界框中心点x, y
     *   - a: 宽高比 (aspect ratio)
     *   - h: 高度
     *   - vx, vy, va, vh: 对应状态量的速度
     * 测量向量为4维: [cx, cy, a, h]，直接来自检测框。
     */
    cv::KalmanFilter kf;
};
