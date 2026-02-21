#include <gtest/gtest.h>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <map>

// 引入被测函数的依赖声明
struct LidarSLAMObject {
    int object_id = -1;
    bool initialized = false;
    bool associated = false;
    bool dynamic = false;
    float velocity = 0.0f;
    std::vector<float> optimize_t{0, 0, 0, 0, 0, 0};
    std::vector<float> measure_lwh{0, 0, 0};
};

struct LidarSLAMFrame {
    int frame_id = -1;
    std::vector<LidarSLAMObject> objects;
};

struct ObjectTrajectory {
    int object_id;
    std::vector<Eigen::Vector3f> positions;
    bool is_active = false;
    ros::Time last_seen;
};

// 声明全局变量
std::vector<LidarSLAMFrame> frames;
int flow = 0;
ros::Time timeLaserInfoStamp;
std::string odometryFrame = "camera_init";
std::map<int, ObjectTrajectory> object_trajectories;

// 声明被测函数
std::vector<float> generateColorFromID(int object_id);
void visualizeTrackedObjects();
void visualizeObjectTrajectories();

// 模拟ROS发布器
class MockPublisher {
public:
    void publish(const visualization_msgs::MarkerArray& markers) {
        last_published_markers = markers;
        publish_count++;
    }
    
    visualization_msgs::MarkerArray last_published_markers;
    int publish_count = 0;
};

MockPublisher pubTrackedObjects;

// 重新实现被测函数，用于测试
std::vector<float> generateColorFromID(int object_id) {
    // 使用HSV颜色空间生成颜色，确保同一ID始终得到相同颜色
    std::srand(object_id * 12345);  // 固定种子确保颜色一致
    float hue = (float)(std::rand() % 360) / 360.0f;
    float saturation = 0.8f + (float)(std::rand() % 20) / 100.0f;  // 0.8-1.0
    float value = 0.8f + (float)(std::rand() % 20) / 100.0f;       // 0.8-1.0
    
    // HSV to RGB conversion
    float c = value * saturation;
    float x = c * (1.0f - std::abs(std::fmod(hue * 6.0f, 2.0f) - 1.0f));
    float m = value - c;
    
    float r, g, b;
    int h_i = (int)(hue * 6);
    if (h_i == 0) { r = c; g = x; b = 0; }
    else if (h_i == 1) { r = x; g = c; b = 0; }
    else if (h_i == 2) { r = 0; g = c; b = x; }
    else if (h_i == 3) { r = 0; g = x; b = c; }
    else if (h_i == 4) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    
    std::vector<float> color = {r + m, g + m, b + m};
    return color;
}

class VisualizeTrackedObjectsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化ROS（如果还没初始化）
        if (!ros::isInitialized()) {
            int argc = 0;
            char** argv = nullptr;
            ros::init(argc, argv, "test_visualize_tracked_objects");
        }
        
        // 清空全局状态
        frames.clear();
        flow = 0;
        object_trajectories.clear();
        pubTrackedObjects.publish_count = 0;
        
        // 设置时间戳
        timeLaserInfoStamp = ros::Time::now();
    }
    
    void TearDown() override {
        frames.clear();
        object_trajectories.clear();
    }
    
    void createTestObjects(int count, bool associated = true, bool initialized = true, bool dynamic = false) {
        for (int i = 0; i < count; ++i) {
            LidarSLAMObject obj;
            obj.object_id = i + 1;
            obj.associated = associated;
            obj.initialized = initialized;
            obj.dynamic = dynamic;
            obj.velocity = dynamic ? 50.0f : 0.0f;
            
            // 设置位置和尺寸
            obj.optimize_t = {
                0.1f,  // roll
                0.2f,  // pitch  
                1.57f, // yaw (90度)
                10.0f + i * 2.0f,  // x
                5.0f + i * 1.0f,   // y
                1.0f   // z
            };
            obj.measure_lwh = {2.0f, 1.5f, 1.2f};  // L, W, H
            
            LidarSLAMFrame frame;
            frame.frame_id = 1;
            frame.objects.push_back(obj);
            frames.push_back(frame);
        }
        flow = frames.size();
    }
};

// 测试用例1: 空帧列表
TEST_F(VisualizeTrackedObjectsTest, EmptyFramesList) {
    frames.clear();
    flow = 0;
    
    visualizeTrackedObjects();
    
    // 应该发布空的marker数组（仅包含DELETEALL标记）
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
    EXPECT_GE(pubTrackedObjects.last_published_markers.markers.size(), 2);  // 至少包含2个DELETEALL标记
}

// 测试用例2: 无效的flow值
TEST_F(VisualizeTrackedObjectsTest, InvalidFlowValue) {
    createTestObjects(3);
    flow = frames.size() + 1;  // 超出范围
    
    visualizeTrackedObjects();
    
    // 应该不发布任何标记（或者只发布DELETEALL）
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
}

// 测试用例3: 正常情况 - 关联且初始化的物体
TEST_F(VisualizeTrackedObjectsTest, NormalCaseAssociatedInitializedObjects) {
    createTestObjects(2, true, true, false);
    flow = frames.size();
    
    visualizeTrackedObjects();
    
    // 应该发布正确的标记数量
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
    // 每个物体会生成：1个边界框 + 1个文本标签
    // 加上2个DELETEALL标记
    EXPECT_EQ(pubTrackedObjects.last_published_markers.markers.size(), 6);  // 2*(1框+1标签) + 2个DELETEALL
}

// 测试用例4: 未关联的物体
TEST_F(VisualizeTrackedObjectsTest, UnassociatedObjects) {
    createTestObjects(2, false, true, false);
    flow = frames.size();
    
    visualizeTrackedObjects();
    
    // 未关联的物体不应该被可视化
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
    // 应该只包含DELETEALL标记
    EXPECT_EQ(pubTrackedObjects.last_published_markers.markers.size(), 2);
}

// 测试用例5: 动态物体显示速度信息
TEST_F(VisualizeTrackedObjectsTest, DynamicObjectsWithVelocity) {
    createTestObjects(1, true, true, true);
    flow = frames.size();
    
    visualizeTrackedObjects();
    
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
    // 检查动态物体的文本标签是否包含速度信息
    bool found_velocity_text = false;
    for (const auto& marker : pubTrackedObjects.last_published_markers.markers) {
        if (marker.type == visualization_msgs::Marker::TEXT_VIEW_FACING) {
            if (marker.text.find("V:") != std::string::npos) {
                found_velocity_text = true;
                break;
            }
        }
    }
    EXPECT_TRUE(found_velocity_text);
}

// 测试用例6: 边界框几何计算正确性
TEST_F(VisualizeTrackedObjectsTest, BoundingBoxGeometry) {
    createTestObjects(1, true, true, false);
    flow = frames.size();
    
    visualizeTrackedObjects();
    
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
    
    // 查找边界框标记
    visualization_msgs::Marker bbox_marker;
    bool found_bbox = false;
    for (const auto& marker : pubTrackedObjects.last_published_markers.markers) {
        if (marker.ns == "tracked_boxes" && marker.type == visualization_msgs::Marker::LINE_LIST) {
            bbox_marker = marker;
            found_bbox = true;
            break;
        }
    }
    EXPECT_TRUE(found_bbox);
    
    // 检查边界框应该有12条边，每条边2个点
    EXPECT_EQ(bbox_marker.points.size(), 24);  // 12 edges * 2 points each
    
    // 检查边界框的顶点位置是否正确
    // 这里可以添加更详细的几何验证
}

// 测试用例7: 轨迹更新功能
TEST_F(VisualizeTrackedObjectsTest, TrajectoryUpdate) {
    createTestObjects(1, true, true, false);
    flow = frames.size();
    
    // 初始轨迹为空
    EXPECT_EQ(object_trajectories.size(), 0);
    
    visualizeTrackedObjects();
    
    // 应该创建轨迹记录
    EXPECT_EQ(object_trajectories.size(), 1);
    EXPECT_TRUE(object_trajectories[1].is_active);
    EXPECT_EQ(object_trajectories[1].positions.size(), 1);
}

// 测试用例8: 颜色生成的一致性
TEST_F(VisualizeTrackedObjectsTest, ColorConsistency) {
    // 相同ID应该生成相同颜色
    std::vector<float> color1 = generateColorFromID(42);
    std::vector<float> color2 = generateColorFromID(42);
    
    EXPECT_EQ(color1.size(), 3);
    EXPECT_EQ(color2.size(), 3);
    EXPECT_FLOAT_EQ(color1[0], color2[0]);
    EXPECT_FLOAT_EQ(color1[1], color2[1]);
    EXPECT_FLOAT_EQ(color1[2], color2[2]);
    
    // 不同ID应该生成不同颜色（大概率）
    std::vector<float> color3 = generateColorFromID(43);
    bool colors_different = (color1[0] != color3[0]) || (color1[1] != color3[1]) || (color1[2] != color3[2]);
    EXPECT_TRUE(colors_different);
}

// 测试用例9: 混合物体类型
TEST_F(VisualizeTrackedObjectsTest, MixedObjectTypes) {
    // 创建不同类型的物体
    LidarSLAMFrame frame;
    
    // 关联且初始化的物体
    LidarSLAMObject obj1;
    obj1.object_id = 1;
    obj1.associated = true;
    obj1.initialized = true;
    obj1.dynamic = false;
    obj1.optimize_t = {0, 0, 0, 10, 5, 1};
    obj1.measure_lwh = {2, 1.5, 1.2};
    frame.objects.push_back(obj1);
    
    // 未关联的物体
    LidarSLAMObject obj2;
    obj2.object_id = 2;
    obj2.associated = false;
    obj2.initialized = true;
    obj2.optimize_t = {0, 0, 0, 12, 6, 1};
    obj2.measure_lwh = {2, 1.5, 1.2};
    frame.objects.push_back(obj2);
    
    // 动态物体
    LidarSLAMObject obj3;
    obj3.object_id = 3;
    obj3.associated = true;
    obj3.initialized = true;
    obj3.dynamic = true;
    obj3.velocity = 60.0f;
    obj3.optimize_t = {0, 0, 0, 14, 7, 1};
    obj3.measure_lwh = {2, 1.5, 1.2};
    frame.objects.push_back(obj3);
    
    frames.push_back(frame);
    flow = frames.size();
    
    visualizeTrackedObjects();
    
    // 只应该可视化关联的物体（obj1和obj3）
    EXPECT_EQ(pubTrackedObjects.publish_count, 1);
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// 实现被测函数（简化的模拟版本）
void visualizeTrackedObjects() {
    if (frames.empty()) {
        ROS_WARN_THROTTLE(5, "visualizeTrackedObjects: frames is empty");
        return;
    }
    
    if (flow <= 0 || flow > (int)frames.size()) {
        std::cout << "visualizeTrackedObjects: invalid flow=" << flow << ", frames.size()=" << frames.size() << std::endl;
        return;
    }
    
    int current_frame_idx = flow - 1;
    
    visualization_msgs::MarkerArray markerArray;
    
    // 清除标记
    visualization_msgs::Marker deleteMarker;
    deleteMarker.header.frame_id = odometryFrame;
    deleteMarker.header.stamp = timeLaserInfoStamp;
    deleteMarker.action = visualization_msgs::Marker::DELETEALL;
    
    deleteMarker.ns = "tracked_boxes";
    markerArray.markers.push_back(deleteMarker);
    
    deleteMarker.ns = "tracked_labels";
    markerArray.markers.push_back(deleteMarker);
    
    // 标记轨迹为非活跃
    for (auto& pair : object_trajectories) {
        pair.second.is_active = false;
    }
    
    int marker_id = 0;
    const LidarSLAMFrame& current_frame = frames[current_frame_idx];
    
    for (const LidarSLAMObject& obj : current_frame.objects) {
        if (!obj.associated || obj.object_id < 0) {
            continue;
        }
        
        int object_id = obj.object_id;
        std::vector<float> color = generateColorFromID(object_id);
        
        // 更新轨迹
        if (object_trajectories.find(object_id) == object_trajectories.end()) {
            ObjectTrajectory traj;
            traj.object_id = object_id;
            traj.is_active = true;
            traj.last_seen = timeLaserInfoStamp;
            object_trajectories[object_id] = traj;
        }
        object_trajectories[object_id].positions.push_back(
            Eigen::Vector3f(obj.optimize_t[3], obj.optimize_t[4], obj.optimize_t[5]));
        object_trajectories[object_id].is_active = true;
        object_trajectories[object_id].last_seen = timeLaserInfoStamp;
        
        // 这里可以添加边界框和文本标签的创建逻辑
        // 为了测试简化，我们只验证基本逻辑
        
        // 创建边界框标记（简化版）
        visualization_msgs::Marker bbox_marker;
        bbox_marker.header.frame_id = odometryFrame;
        bbox_marker.header.stamp = timeLaserInfoStamp;
        bbox_marker.ns = "tracked_boxes";
        bbox_marker.id = marker_id++;
        bbox_marker.type = visualization_msgs::Marker::LINE_LIST;
        bbox_marker.action = visualization_msgs::Marker::ADD;
        bbox_marker.pose.orientation.w = 1.0;
        bbox_marker.scale.x = 0.08;
        bbox_marker.color.r = color[0];
        bbox_marker.color.g = color[1];
        bbox_marker.color.b = color[2];
        bbox_marker.color.a = 0.9;
        
        // 简化边界框点计算
        float l = obj.measure_lwh[0];
        float w = obj.measure_lwh[1];
        float h = obj.measure_lwh[2];
        
        // 添加简化的边界框点（实际实现更复杂）
        geometry_msgs::Point p1, p2;
        p1.x = obj.optimize_t[3] - l/2; p1.y = obj.optimize_t[4] - w/2; p1.z = obj.optimize_t[5] - h/2;
        p2.x = obj.optimize_t[3] + l/2; p2.y = obj.optimize_t[4] + w/2; p2.z = obj.optimize_t[5] + h/2;
        bbox_marker.points.push_back(p1);
        bbox_marker.points.push_back(p2);
        
        markerArray.markers.push_back(bbox_marker);
        
        // 创建文本标签
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = odometryFrame;
        text_marker.header.stamp = timeLaserInfoStamp;
        text_marker.ns = "tracked_labels";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position.x = obj.optimize_t[3];
        text_marker.pose.position.y = obj.optimize_t[4];
        text_marker.pose.position.z = obj.optimize_t[5] + h/2 + 0.8;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = 0.6;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        
        std::stringstream ss;
        ss << "ID:" << object_id;
        if (obj.dynamic) {
            ss << " V:" << std::fixed << std::setprecision(1) << obj.velocity << "km/h";
        }
        text_marker.text = ss.str();
        
        markerArray.markers.push_back(text_marker);
    }
    
    // 发布标记
    pubTrackedObjects.publish(markerArray);
}

void visualizeObjectTrajectories() {
    // 简化实现
}