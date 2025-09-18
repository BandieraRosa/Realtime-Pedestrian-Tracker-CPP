/**
 * @file main.cpp
 * @brief 项目主程序入口。
 * @details
 * 负责初始化检测器和追踪器，从视频源读取帧，
 * 协调检测和追踪过程，并最终将结果可视化。
 */

#include "Detector.hpp"
#include "Sort.hpp"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp> // For imshow, waitKey, etc.

int main(int argc, char* argv[]) {

    // =========================================================================
    // 1. 参数配置 (Configuration)
    // =========================================================================
    
    // --- 路径配置 ---
    std::string model_path = "your_model_path";
    std::string video_path = "your_video_path";
    // 如果要使用摄像头，请修改为摄像头索引，例如:
    // int camera_index = 0; 
    
    // --- 推理设备 ---
    std::string device = "CPU";

    // --- 检测器参数 ---
    // 重要：如果看不到任何检测框，首先应该尝试降低此值！
    float conf_threshold = 0.35f; // 置信度阈值
    float nms_threshold = 0.5f;   // 非极大值抑制的IoU阈值
    int pedestrian_class_id = 0;  // COCO数据集中 "person" 的类别ID通常是0

    // --- SORT追踪器参数 ---
    int max_age = 60;       // 轨迹最大存活帧数。增加此值可应对更长时间的遮挡。
    int min_hits = 3;       // 成为可靠轨迹所需的最小连续命中次数。
    double iou_threshold = 0.3; // 匹配检测框和轨迹时的IoU阈值。

    try {
        // =========================================================================
        // 2. 初始化 (Initialization)
        // =========================================================================
        
        // 初始化检测器，传入模型路径和相关参数
        Detector detector(model_path, device, conf_threshold, nms_threshold, pedestrian_class_id);
        
        // 初始化SORT追踪器，传入为行人追踪优化的参数
        SORTTracker tracker(max_age, min_hits, iou_threshold); 
        
        // 打开视频文件或摄像头
        cv::VideoCapture cap(video_path);
        // cv::VideoCapture cap(camera_index, cv::CAP_V4L2); // 使用摄像头的示例

        // 检查视频源是否成功打开
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video source: " << video_path << std::endl;
            return -1;
        }

        cv::Mat frame;
        int frame_count = 0;

        // =========================================================================
        // 3. 主循环 (Main Loop)
        // =========================================================================
        while (cap.read(frame)) {
            if (frame.empty()) {
                std::cout << "End of video stream." << std::endl;
                break;
            }

            frame_count++;
            
            // a. 执行检测
            auto detections = detector.detect(frame);

            // b. 更新追踪器
            auto tracked_objects = tracker.update(detections);

            // c. 可视化结果
            for (const auto& track : tracked_objects) {
                cv::Rect box = track.get_state_as_bbox();
                // 绘制边界框
                cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
                
                // 准备标签文本
                std::string label = "ID: " + std::to_string(track.id);
                int baseLine;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseLine);
                
                // 绘制标签背景
                cv::rectangle(frame, cv::Point(box.x, box.y - label_size.height - baseLine), 
                              cv::Point(box.x + label_size.width, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
                // 绘制标签文本
                cv::putText(frame, label, cv::Point(box.x, box.y - baseLine), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
            }

            cv::imshow("YOLO+SORT Pedestrian Tracking", frame);

            // 按 'q' 键或ESC键退出
            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }

        // 释放资源
        cap.release();
        cv::destroyAllWindows();

    } catch (const ov::Exception& e) {
        std::cerr << "OpenVINO error: " << e.what() << std::endl;
        return -1;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
