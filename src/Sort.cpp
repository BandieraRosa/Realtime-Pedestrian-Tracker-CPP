/**
 * @file Sort.cpp
 * @brief SORT追踪器实现文件。
 * @details
 * 实现了 SORT 算法的核心步骤：预测、关联、更新。
 */

#include "Sort.hpp"
#include "Hungarian.hpp"
#include <set>

// 声明在 IoU.cpp 中定义的外部函数
double calculate_iou(const cv::Rect& box1, const cv::Rect& box2);

SORTTracker::SORTTracker(int max_age, int min_hits, double iou_threshold)
    : max_age_(max_age), 
      min_hits_(min_hits), 
      iou_threshold_(iou_threshold), 
      next_id_(1) {} // 初始化追踪ID从1开始

std::vector<Track> SORTTracker::update(const std::vector<Detection>& detections) {
    // --- 1. 预测 (Prediction) ---
    // 对所有现有轨迹，使用其内部的卡尔曼滤波器来预测它们在当前帧的位置。
    // 同时，每个轨迹的 `time_since_update` 计数器加一。
    for (auto& track : tracks_) {
        track.predict();
    }

    // --- 2. 关联 (Association) ---
    // 将预测后的轨迹与当前帧的检测结果进行匹配。
    
    std::set<int> matched_track_indices;
    std::set<int> matched_det_indices;

    // 只有在同时存在待匹配的轨迹和检测结果时，才执行复杂的匹配算法
    if (!tracks_.empty() && !detections.empty()) {
        std::vector<cv::Rect> predicted_boxes;
        for (const auto& track : tracks_) {
            predicted_boxes.push_back(track.get_state_as_bbox());
        }

        std::vector<cv::Rect> detection_boxes;
        for (const auto& det : detections) {
            detection_boxes.push_back(det.box);
        }

        // a. 构建成本矩阵：成本定义为 1 - IoU。匈牙利算法将寻找成本最小的分配方案。
        std::vector<std::vector<double>> cost_matrix(tracks_.size(), std::vector<double>(detections.size()));
        for (size_t i = 0; i < tracks_.size(); ++i) {
            for (size_t j = 0; j < detections.size(); ++j) {
                cost_matrix[i][j] = 1.0 - calculate_iou(predicted_boxes[i], detection_boxes[j]);
            }
        }

        // b. 使用匈牙利算法求解分配问题，找到最优匹配。
        HungarianAlgorithm hungarian_solver;
        std::vector<int> assignment; // assignment[i] = j 表示第i个轨迹匹配到了第j个检测
        hungarian_solver.Solve(cost_matrix, assignment);

        // c. 过滤并记录匹配结果。
        for (size_t i = 0; i < tracks_.size(); ++i) {
            // 检查分配是否有效 (assignment[i] != -1)
            // 并且匹配的IoU是否满足阈值 (这是一个关键优化，拒绝劣质匹配)
            if (assignment[i] != -1 && (1.0 - cost_matrix[i][assignment[i]]) >= iou_threshold_) {
                // 如果匹配成功，更新轨迹状态
                tracks_[i].update(detection_boxes[assignment[i]]);
                // 记录已匹配的轨迹和检测的索引
                matched_track_indices.insert(i);
                matched_det_indices.insert(assignment[i]);
            }
        }
    }
    
    // --- 3. 轨迹管理 (Track Management) ---

    // a. 为未匹配的检测创建新轨迹。
    for (size_t i = 0; i < detections.size(); ++i) {
        if (matched_det_indices.find(i) == matched_det_indices.end()) {
            tracks_.emplace_back(next_id_++, detections[i].box);
        }
    }

    // b. 移除生命周期结束的轨迹。
    //    对于未匹配的轨迹，它们的 `time_since_update` 会持续增加。
    //    当这个值超过 `max_age` 时，轨迹将被移除。
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), 
        [this](const Track& track) {
            return track.time_since_update > this->max_age_;
        }), tracks_.end());

    // --- 4. 准备输出 ---
    // 只输出那些当前帧被更新过，并且已经达到“可靠”状态的轨迹。
    std::vector<Track> active_tracks;
    for (const auto& track : tracks_) {
        // `track.time_since_update == 0` 保证了只输出当前帧匹配到的轨迹。
        // `track.hits >= min_hits_` 保证了轨迹足够稳定，过滤掉了初始阶段的噪声。
        if (track.time_since_update == 0 && track.hits >= min_hits_) {
            active_tracks.push_back(track);
        }
    }
    
    return active_tracks;
}
