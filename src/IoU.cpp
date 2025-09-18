/**
 * @file IoU.cpp
 * @brief 提供了计算两个边界框之间交并比（Intersection over Union, IoU）的函数。
 * @details
 * IoU 是目标检测和追踪领域一个核心的评价指标，用于衡量两个边界框的重叠程度。
 * 它的值域为 [0, 1]，值越大表示重叠程度越高。
 * 在本项目中，它被SORT追踪器用来构建成本矩阵，以进行检测与轨迹的匹配。
 */

#include <opencv2/opencv.hpp>
#include <algorithm> // For std::max and std::min

/**
 * @brief 计算两个边界框（cv::Rect）的交并比 (IoU)。
 * 
 * @details
 * IoU 的计算公式为：
 *   IoU = Area(Intersection) / Area(Union)
 * 其中:
 *   - Area(Intersection) 是两个边界框的交集面积。
 *   - Area(Union) 是两个边界框的并集面积，计算方式为：Area(box1) + Area(box2) - Area(Intersection)。
 * 
 * @param box1 第一个边界框。
 * @param box2 第二个边界框。
 * @return double 计算出的IoU值，范围在 [0.0, 1.0] 之间。
 */
double calculate_iou(const cv::Rect& box1, const cv::Rect& box2) {
    // 1. 计算交集区域的左上角和右下角坐标。
    //    交集的左上角(x1, y1)是两个框左上角坐标中较大的那个。
    //    交集的右下角(x2, y2)是两个框右下角坐标中较小的那个。
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    // 2. 计算交集区域的宽度和高度。
    //    如果 x2 < x1 或 y2 < y1，表示两个框没有重叠，宽度或高度会是负数。
    //    使用 std::max(0, ...) 来确保非重叠情况下的面积为0。
    double intersection_width = std::max(0, x2 - x1);
    double intersection_height = std::max(0, y2 - y1);
    
    // 3. 计算交集面积。
    double intersection_area = intersection_width * intersection_height;

    // 4. 计算并集面积。
    double union_area = static_cast<double>(box1.area()) + static_cast<double>(box2.area()) - intersection_area;

    // 5. 计算并返回IoU。
    //    进行除零检查，如果并集面积为0（例如两个空框），则IoU为0。
    if (union_area == 0) {
        return 0.0;
    }

    return intersection_area / union_area;
}
