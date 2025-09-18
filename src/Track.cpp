/**
 * @file Track.cpp
 * @brief 单个追踪轨迹的实现文件。
 */

#include "Track.hpp"

Track::Track(int track_id, const cv::Rect& bbox) {
    id = track_id;
    hits = 1; // 初始命中数为1
    time_since_update = 0; // 刚刚创建，所以是0

    // 初始化卡尔曼滤波器 (8个状态量, 4个测量量)
    kf = cv::KalmanFilter(8, 4, 0);

    // --- 1. 初始化状态向量 (statePost) ---
    // [cx, cy, a, h, vx, vy, va, vh]
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    float aspect_ratio = (bbox.height > 0) ? (static_cast<float>(bbox.width) / bbox.height) : 0;
    
    kf.statePost.at<float>(0) = cx;
    kf.statePost.at<float>(1) = cy;
    kf.statePost.at<float>(2) = aspect_ratio;
    kf.statePost.at<float>(3) = bbox.height;
    // 初始速度分量设置为0
    kf.statePost.at<float>(4) = 0;
    kf.statePost.at<float>(5) = 0;
    kf.statePost.at<float>(6) = 0;
    kf.statePost.at<float>(7) = 0;

    // --- 2. 定义状态转移矩阵 F (transitionMatrix) ---
    // 定义了状态如何从一个时间步转移到下一个时间步（基于物理模型）
    // x_k = F * x_{k-1}
    // cx_k = cx_{k-1} + vx_{k-1} * dt (dt=1帧)
    cv::setIdentity(kf.transitionMatrix);
    kf.transitionMatrix.at<float>(0, 4) = 1;
    kf.transitionMatrix.at<float>(1, 5) = 1;
    kf.transitionMatrix.at<float>(2, 6) = 1;
    kf.transitionMatrix.at<float>(3, 7) = 1;

    // --- 3. 定义测量矩阵 H (measurementMatrix) ---
    // 定义了如何从状态向量中提取出测量向量
    // z_k = H * x_k
    // 我们只能直接测量 cx, cy, a, h
    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    kf.measurementMatrix.at<float>(0, 0) = 1; // cx
    kf.measurementMatrix.at<float>(1, 1) = 1; // cy
    kf.measurementMatrix.at<float>(2, 2) = 1; // aspect ratio
    kf.measurementMatrix.at<float>(3, 3) = 1; // height
    
    // --- 4. 定义噪声协方差矩阵 ---
    // Q (processNoiseCov): 过程噪声，反映了我们对物理模型（恒定速度）的不确定性。
    // 速度分量的不确定性应该更大。
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    kf.processNoiseCov.at<float>(4, 4) = 1e-1;
    kf.processNoiseCov.at<float>(5, 5) = 1e-1;
    kf.processNoiseCov.at<float>(6, 6) = 1e-2; // 宽高比变化通常较小
    kf.processNoiseCov.at<float>(7, 7) = 1e-2; // 高度变化通常较小

    // R (measurementNoiseCov): 测量噪声，反映了我们对检测器结果的不确定性。
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    
    // P (errorCovPost): 后验误差协方差，反映了对初始状态估计的不确定性。
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

void Track::predict() {
    kf.predict();
    time_since_update++; // 增加“失踪”时间
}

void Track::update(const cv::Rect& bbox) {
    time_since_update = 0; // 重置“失踪”时间
    hits++; // 增加命中数

    // 从检测框创建测量向量
    cv::Mat measurement = cv::Mat::zeros(4, 1, CV_32F);
    measurement.at<float>(0) = bbox.x + bbox.width / 2.0f;
    measurement.at<float>(1) = bbox.y + bbox.height / 2.0f;
    measurement.at<float>(2) = (bbox.height > 0) ? (static_cast<float>(bbox.width) / bbox.height) : 0;
    measurement.at<float>(3) = bbox.height;
    
    // 使用新的测量值校正卡尔曼滤波器
    kf.correct(measurement);
}

cv::Rect Track::get_state_as_bbox() const {
    // 从状态向量中提取出边界框信息
    // 使用 `statePost`，因为它代表了经过最近一次测量更新后的最优估计
    cv::Mat state = kf.statePost;
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float aspect_ratio = state.at<float>(2);
    float h = state.at<float>(3);
    float w = aspect_ratio * h;

    return cv::Rect(
        static_cast<int>(cx - w / 2), 
        static_cast<int>(cy - h / 2), 
        static_cast<int>(w), 
        static_cast<int>(h)
    );
}
