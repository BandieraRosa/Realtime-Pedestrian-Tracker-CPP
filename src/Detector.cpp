/**
 * @file Detector.cpp
 * @brief YOLOv11n目标检测器实现文件。
 * @details
 * 实现了模型的加载、预处理、推理和后处理的完整流程。
 * 采用了手动的 letterbox + blobFromImage 预处理，以确保最高的可控性和准确性。
 */

#include "Detector.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <algorithm>


// =================================================================================
// Letterbox 辅助函数
// =================================================================================

/**
 * @struct LetterboxResult
 * @brief 存储letterbox预处理结果的结构体。
 */
struct LetterboxResult {
    cv::Mat image;  ///< letterbox处理后的图像。
    float scale;    ///< 保持宽高比的实际缩放比例。
    int pad_x;      ///< x轴方向的填充像素数（单边）。
    int pad_y;      ///< y轴方向的填充像素数（单边）。
};

/**
 * @brief 使用letterbox方法对图像进行缩放和居中填充。
 * 
 * @details
 * 该方法可以保持图像的原始宽高比，避免目标因拉伸而变形。
 * 对于检测任务，这通常能带来比直接 `cv::resize` 更好的精度。
 * 
 * @param source 原始输入图像。
 * @param target_size 目标尺寸 (模型的输入尺寸)。
 * @return LetterboxResult 包含处理后图像和相关参数的结构体。
 */
LetterboxResult letterbox(const cv::Mat& source, const cv::Size& target_size) {
    int target_w = target_size.width;
    int target_h = target_size.height;
    
    // 1. 计算缩放比例，选择原始到目标尺寸的最小缩放比，以确保整个图像能被放入目标框内。
    float scale = std::min(static_cast<float>(target_w) / source.cols, static_cast<float>(target_h) / source.rows);
    
    // 2. 计算缩放后的新尺寸。
    int new_w = static_cast<int>(source.cols * scale);
    int new_h = static_cast<int>(source.rows * scale);
    
    // 3. 对原始图像进行缩放。
    cv::Mat resized_image;
    cv::resize(source, resized_image, cv::Size(new_w, new_h));

    // 4. 创建一个目标大小的灰色画布。
    cv::Mat letterbox_image = cv::Mat::zeros(target_h, target_w, CV_8UC3);
    letterbox_image.setTo(cv::Scalar(114, 114, 114)); // YOLOv5/v8常用的填充颜色
    
    // 5. 计算为了居中放置所需的填充量 (padding)。
    int pad_x = (target_w - new_w) / 2;
    int pad_y = (target_h - new_h) / 2;

    // 6. 将缩放后的图像拷贝到画布中央。
    resized_image.copyTo(letterbox_image(cv::Rect(pad_x, pad_y, new_w, new_h)));

    return {letterbox_image, scale, pad_x, pad_y};
}

// =================================================================================
// Detector 类成员函数实现
// =================================================================================

Detector::Detector(const std::string& model_path, const std::string& device, float conf_threshold, float nms_threshold, int target_class_id)
    : conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold),
      target_class_id_(target_class_id),
      letterbox_scale_(1.0f), // 默认初始化
      letterbox_pad_x_(0),
      letterbox_pad_y_(0) {
    
    // 1. 读取ONNX模型文件。
    auto model = core_.read_model(model_path);
    
    // 2. 将模型编译到目标设备上（例如 "CPU"）。
    //    这一步会将模型优化为特定硬件的格式，以获得最佳性能。
    //    注意：我们没有使用OpenVINO的PrePostProcessor (PPP)，因为所有预处理都将手动完成，
    //    这给予我们完全的控制权，并避免了复杂的PPP配置问题。
    compiled_model_ = core_.compile_model(model, device);
    
    // 3. 创建一个推理请求。后续的推理将通过这个对象进行。
    infer_request_ = compiled_model_.create_infer_request();
    
    // 4. 获取并保存模型的输入尺寸。
    //    我们假设模型的输入布局是 NCHW (Batch, Channels, Height, Width)。
    auto input_shape = compiled_model_.input().get_shape();
    input_size_ = cv::Size(static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2]));
}


std::vector<Detection> Detector::detect(const cv::Mat& image) {
    if (image.empty()) return {};

    // --- 1. 预处理 (Preprocessing) ---
    // 使用 letterbox 对图像进行缩放和填充，以匹配模型的输入尺寸，同时保持宽高比。
    auto letterbox_result = letterbox(image, input_size_);
    // 保存 letterbox 的参数，这些参数在后处理阶段用于精确地将坐标还原到原始图像空间。
    letterbox_scale_ = letterbox_result.scale;
    letterbox_pad_x_ = letterbox_result.pad_x;
    letterbox_pad_y_ = letterbox_result.pad_y;

    // --- 2. 创建 Blob ---
    // 使用 OpenCV 的 dnn 模块将处理后的图像转换为一个 "blob"。
    // 这个函数会完成以下所有操作：
    //   a. 布局转换: HWC (Height, Width, Channels) -> NCHW (Batch, Channels, Height, Width)
    //   b. 归一化: 将像素值从 [0, 255] 缩放到 [0.0, 1.0] (通过乘以 1.0/255.0)。
    //   c. 颜色空间转换: BGR -> RGB (因为 'swapRB' 参数为 true)。
    //   d. 数据类型转换: CV_8U -> CV_32F。
    cv::Mat blob;
    cv::dnn::blobFromImage(letterbox_result.image, blob, 1.0 / 255.0, input_size_, true, false);

    // --- 3. 设置输入张量 ---
    // 创建一个OpenVINO张量，其数据指向我们刚刚创建的blob。
    ov::Tensor input_tensor(ov::element::f32, {1, 3, (size_t)input_size_.height, (size_t)input_size_.width}, blob.data);
    infer_request_.set_input_tensor(input_tensor);

    // --- 4. 执行推理 ---
    infer_request_.infer();

    // --- 5. 获取输出并进行后处理 ---
    const auto output_tensor = infer_request_.get_output_tensor();
    return postprocess(image, output_tensor);
}


std::vector<Detection> Detector::postprocess(const cv::Mat& original_image, const ov::Tensor& output_tensor) {
    const auto output_shape = output_tensor.get_shape();
    const size_t num_proposals = output_shape[2]; // 8400 for 640x640 model
    const size_t num_classes = output_shape[1] - 4; // 80 for COCO dataset

    // --- 1. 转置输出数据 (CRITICAL STEP!) ---
    // 模型的原始输出形状为 [1, 84, 8400]，其内存布局是按通道连续的。
    // (即，所有8400个cx, 然后是所有8400个cy, ...)。
    // 为了方便按“提议”(proposal)进行处理，我们必须将其转置为 [8400, 84] 的形状。
    // 这样，每一行都代表一个完整的提议（cx, cy, w, h, class_scores...）。
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, const_cast<float*>(output_tensor.data<float>()));
    cv::transpose(output_buffer, output_buffer); // In-place transpose

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // --- 2. 解码和过滤提议 ---
    for (int i = 0; i < num_proposals; ++i) {
        const float* proposal = output_buffer.ptr<float>(i); // 获取第 i 个提议的数据行
        const float* class_scores_ptr = proposal + 4; // 前4个是bbox，后面是类别分数

        // 找到分数最高的类别及其分数
        auto max_score_it = std::max_element(class_scores_ptr, class_scores_ptr + num_classes);
        float max_class_score = *max_score_it;
        int class_id = std::distance(class_scores_ptr, max_score_it);

        // 应用置信度阈值
        if (max_class_score > conf_threshold_) {
            // 应用类别过滤器
            if (class_id == target_class_id_) {
                float cx = proposal[0]; // center_x
                float cy = proposal[1]; // center_y
                float w = proposal[2];  // width
                float h = proposal[3];  // height

                // --- 3. 坐标还原 (CRITICAL STEP!) ---
                // 将坐标从 letterbox 空间 (e.g., 640x640) 映射回原始图像空间。
                // 这是一个严格的逆运算。
                
                // a. 减去填充(padding)的影响，得到在缩放后图像上的坐标。
                float corrected_cx = cx - letterbox_pad_x_;
                float corrected_cy = cy - letterbox_pad_y_;
                
                // b. 除以缩放比例(scale)，将其放大回原始图像的尺寸。
                float original_cx = corrected_cx / letterbox_scale_;
                float original_cy = corrected_cy / letterbox_scale_;
                float original_w = w / letterbox_scale_;
                float original_h = h / letterbox_scale_;
                
                // c. 从中心点坐标(cx,cy)转换为左上角坐标(left,top)。
                int left = static_cast<int>(original_cx - original_w / 2);
                int top = static_cast<int>(original_cy - original_h / 2);

                boxes.push_back(cv::Rect(left, top, static_cast<int>(original_w), static_cast<int>(original_h)));
                confidences.push_back(max_class_score);
                class_ids.push_back(class_id);
            }
        }
    }

    // --- 4. 非极大值抑制 (NMS) ---
    // 合并同一目标的重叠检测框，只保留置信度最高的那个。
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);
    
    std::vector<Detection> detections;
    for (int idx : indices) {
        detections.push_back({boxes[idx], confidences[idx], class_ids[idx]});
    }

    return detections;
}
