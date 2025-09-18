/*
 * @Author: BandieraRossa 3132716198@qq.com
 * @Date: 2025-09-16 22:36:02
 * @LastEditors: BandieraRossa 3132716198@qq.com
 * @LastEditTime: 2025-09-19 02:31:07
 * @FilePath: /workspace_test_1/include/Sort.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/**
 * @file Sort.hpp
 * @brief SORT (Simple Online and Realtime Tracking) 追踪器核心逻辑的头文件。
 * @details
 * 该类实现了SORT算法，负责管理多个追踪轨迹，
 * 并将新的检测结果与现有轨迹进行匹配。
 */

#pragma once

#include "Track.hpp"
#include "Detector.hpp" // For Detection struct
#include <vector>

class SORTTracker {
public:
    /**
     * @brief SORTTracker 构造函数。
     * 
     * @param max_age 一个轨迹在连续多少帧未被检测到后被视为丢失并删除。
     * @param min_hits 一个新轨迹需要连续多少帧被检测到才能被确认为可靠的追踪目标。
     * @param iou_threshold 匹配检测框和轨迹框时的最小IoU（交并比）阈值。
     */
    SORTTracker(int max_age = 30, int min_hits = 3, double iou_threshold = 0.3);

    /**
     * @brief 追踪器的主更新函数，处理新一帧的检测结果。
     * 
     * @param detections 当前帧的检测结果列表。
     * @return std::vector<Track> 当前帧所有活跃且可靠的轨迹列表。
     */
    std::vector<Track> update(const std::vector<Detection>& detections);

private:
    // --- SORT 算法参数 ---
    int max_age_;           ///< 轨迹最大存活帧数。
    int min_hits_;          ///< 成为可靠轨迹所需的最小连续命中次数。
    double iou_threshold_;  ///< IoU匹配阈值。
    
    // --- 追踪器内部状态 ---
    std::vector<Track> tracks_; ///< 当前所有（包括不活跃的）轨迹的列表。
    int next_id_;           ///< 用于分配给新轨迹的下一个唯一ID。
};
