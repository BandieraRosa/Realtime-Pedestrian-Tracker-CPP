/**
 * @file Detector.hpp
 * @brief 封装了YOLOv11n目标检测器的头文件。
 * @author BandieraRossa
 * @date 2025-09-18
 * 
 * @details
 * 这个类负责加载OpenVINO模型，执行推理，并对结果进行后处理。
 * 它集成了 letterbox 预处理和 NMS（非极大值抑制）等功能，
 * 专门用于检测特定类别的目标（本项目中为“行人”）。
 */

#pragma once

#include <opencv2/opencv.hpp>
#include "openvino/openvino.hpp"
#include <vector>

/**
 * @struct Detection
 * @brief 用于存储单个检测结果的结构体。
 */
struct Detection {
    cv::Rect box;       ///< 目标的边界框 (bounding box)。
    float confidence;   ///< 检测结果的置信度。
    int class_id;       ///< 目标的类别ID。
};

class Detector {
public:
    /**
     * @brief Detector 类的构造函数。
     * 
     * @param model_path ONNX模型文件的路径。
     * @param device 推理设备名称 (例如 "CPU", "GPU")。
     * @param conf_threshold 置信度阈值，低于此值的检测将被忽略。
     * @param nms_threshold NMS（非极大值抑制）的IoU阈值。
     * @param target_class_id 要检测的目标类别ID (例如，COCO数据集中 "person" 的ID为0)。
     */
    Detector(const std::string& model_path, const std::string& device, float conf_threshold, float nms_threshold, int target_class_id);

    /**
     * @brief 对单张图像执行目标检测。
     * 
     * @param image 输入的OpenCV图像 (cv::Mat)。
     * @return std::vector<Detection> 检测到的目标列表。
     */
    std::vector<Detection> detect(const cv::Mat& image);

private:
    /**
     * @brief 对模型的原始输出进行后处理。
     * 
     * @details
     * 该函数负责解析模型的输出张量，执行转置、解码、坐标还原和NMS。
     * @param original_image 用于坐标还原的原始图像 (未经过letterbox处理)。
     * @param output_tensor 模型的原始输出张量。
     * @return std::vector<Detection> 经过后处理和过滤的检测结果列表。
     */
    std::vector<Detection> postprocess(const cv::Mat& original_image, const ov::Tensor& output_tensor);

    // --- OpenVINO 核心成员 ---
    ov::Core core_;                     ///< OpenVINO 核心对象。
    ov::CompiledModel compiled_model_;  ///< 已编译到目标设备的模型。
    ov::InferRequest infer_request_;    ///< 推理请求对象。
    
    // --- 模型与处理参数 ---
    cv::Size input_size_;               ///< 模型的输入尺寸 (例如 640x640)。
    float conf_threshold_;              ///< 置信度阈值。
    float nms_threshold_;               ///< NMS阈值。
    int target_class_id_;               ///< 目标追踪的类别ID。

    // --- Letterbox 预处理参数 ---
    // 这些参数在 detect() 函数中被计算，并在 postprocess() 函数中使用，用于精确还原坐标
    float letterbox_scale_;             ///< letterbox计算出的缩放比例。
    int letterbox_pad_x_;               ///< letterbox在x轴上的填充值。
    int letterbox_pad_y_;               ///< letterbox在y轴上的填充值。
};
