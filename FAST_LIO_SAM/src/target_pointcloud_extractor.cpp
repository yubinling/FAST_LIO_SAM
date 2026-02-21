#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
// #include <pcl/segmentation/extract_clusters.h>  // 暂时注释掉聚类
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <Eigen/Dense>
#include <deque>
#include <mutex>
#include <limits>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <cmath>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class TargetPointCloudExtractor {
private:
    ros::NodeHandle nh_;
    
    // 订阅器
    ros::Subscriber sub_cloud_;
    
    // 发布器
    ros::Publisher pub_target_cloud_;
    ros::Publisher pub_box_cloud_;
    ros::Publisher pub_original_cloud_;  // 原始点云
    ros::Publisher pub_box_marker_;       // 检测框Marker（单个）
    ros::Publisher pub_box_marker_array_; // 检测框MarkerArray（多个）
    ros::Publisher pub_aligned_cloud_;     // 配准后的点云
    ros::Publisher pub_unmatched_cloud_;   // 未匹配的点云
    
    // TF广播器
    tf::TransformBroadcaster tf_broadcaster_;
    
    // 功能开关
    bool publish_original_and_box_;  // 是否发布原始点云和检测框
    bool enable_registration_;      // 是否启用配准功能
    bool publish_all_boxes_;        // 是否发布所有检测框（true=所有框，false=只发布前方固定范围的框）
    
    // 上一帧的框内点云（用于配准）
    PointCloudXYZI::Ptr prev_box_cloud_;
    bool has_prev_frame_;            // 是否有上一帧数据
    Eigen::Matrix4f prev_transformation_;  // 上一帧的变换矩阵（用于下一帧的初始估计）
    bool has_prev_transformation_;    // 是否有上一帧的变换矩阵
    
    // 累积的配准后点云（在局部坐标系中，用于下一帧配准）
    PointCloudXYZI::Ptr prev_box_cloud_local_;  // 累积所有配准后的点云（局部坐标系）
    
    // 第一帧的检测位姿（物体系到雷达系的变换 T0）
    Eigen::Matrix4f T0_;              // 第一帧的变换矩阵
    bool has_T0_;                     // 是否已保存第一帧的变换
    
    // 上一帧的框中心位置和yaw（用于坐标转换）
    double prev_box_x_, prev_box_y_, prev_box_z_, prev_box_yaw_;
    bool has_prev_box_pose_;          // 是否有上一帧的框位姿
    
    // 检测数据队列（从文件加载）
    std::deque<std_msgs::Float64MultiArray> detect_queue_;
    std::mutex detect_mutex_;
    
    // 当前处理的点云
    sensor_msgs::PointCloud2::ConstPtr current_cloud_;
    std::mutex cloud_mutex_;
    
    // 参数
    double target_distance_;      // 目标距离（米）
    double distance_tolerance_;   // 距离容差（米）
    double cluster_tolerance_;    // 聚类容差（米）
    int min_cluster_size_;        // 最小聚类点数
    int max_cluster_size_;        // 最大聚类点数
    double sync_tolerance_;       // 时间同步容差（秒）
    std::string detect_file_path_; // 检测文件路径
    
    // 统计信息
    int frame_count_;
    int success_count_;
    
public:
    TargetPointCloudExtractor() : nh_("~") {
        // 参数初始化
        nh_.param<double>("target_distance", target_distance_, 10.0);
        nh_.param<double>("distance_tolerance", distance_tolerance_, 2.0);
        nh_.param<double>("cluster_tolerance", cluster_tolerance_, 0.3);
        nh_.param<int>("min_cluster_size", min_cluster_size_, 10);
        nh_.param<int>("max_cluster_size", max_cluster_size_, 50000);
        nh_.param<double>("sync_tolerance", sync_tolerance_, 0.001);
        
        std::string cloud_topic;
        nh_.param<std::string>("cloud_topic", cloud_topic, "/cloud_registered_body");
        nh_.param<std::string>("detect_file_path", detect_file_path_, "");
        
        ROS_INFO("[INIT] ========== Target PointCloud Extractor Initializing ==========");
        ROS_INFO("[INIT] Parameter 'cloud_topic' = '%s'", cloud_topic.c_str());
        ROS_INFO("[INIT] Parameter 'detect_file_path' = '%s'", detect_file_path_.c_str());
        
        // 订阅器
        ROS_INFO("[INIT] Subscribing to point cloud topic: '%s'", cloud_topic.c_str());
        sub_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>(
            cloud_topic, 10, &TargetPointCloudExtractor::cloudCallback, this);
        ROS_INFO("[INIT] Subscriber created successfully");
        
        // 功能开关参数
        nh_.param<bool>("publish_original_and_box", publish_original_and_box_, false);
        nh_.param<bool>("enable_registration", enable_registration_, false);
        nh_.param<bool>("publish_all_boxes", publish_all_boxes_, false);
        
        // 发布器
        pub_target_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/target_pointcloud", 10);
        pub_box_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/box_pointcloud", 10);
        pub_original_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/original_pointcloud", 10);
        pub_box_marker_ = nh_.advertise<visualization_msgs::Marker>("/detection_box_marker", 10);
        pub_box_marker_array_ = nh_.advertise<visualization_msgs::MarkerArray>("/detection_box_marker_array", 10);
        pub_aligned_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/aligned_pointcloud", 10);
        pub_unmatched_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/unmatched_pointcloud", 10);
        
        ROS_INFO("[INIT] Publishers created:");
        ROS_INFO("[INIT]   - /target_pointcloud");
        ROS_INFO("[INIT]   - /box_pointcloud");
        ROS_INFO("[INIT]   - /original_pointcloud (enabled: %s)", publish_original_and_box_ ? "true" : "false");
        ROS_INFO("[INIT]   - /detection_box_marker (enabled: %s)", publish_original_and_box_ ? "true" : "false");
        ROS_INFO("[INIT]   - /detection_box_marker_array (enabled: %s)", publish_original_and_box_ ? "true" : "false");
        ROS_INFO("[INIT]   - /aligned_pointcloud (enabled: %s)", enable_registration_ ? "true" : "false");
        ROS_INFO("[INIT]   - /unmatched_pointcloud (enabled: %s)", enable_registration_ ? "true" : "false");
        ROS_INFO("[INIT] Box publishing mode: %s", publish_all_boxes_ ? "ALL boxes" : "Front fixed range only");
        
        // 初始化上一帧数据
        prev_box_cloud_.reset(new PointCloudXYZI);
        has_prev_frame_ = false;
        prev_transformation_.setIdentity();  // 初始化为单位矩阵
        has_prev_transformation_ = false;     // 第一帧没有上一帧的变换
        
        // 初始化累积的配准后点云（局部坐标系）
        prev_box_cloud_local_.reset(new PointCloudXYZI);
        
        // 初始化第一帧检测位姿
        T0_.setIdentity();
        has_T0_ = false;  // 第一帧还没有保存
        
        frame_count_ = 0;
        success_count_ = 0;
        
        // 从文件加载检测数据（现在不再需要，但保留代码）
        if (!detect_file_path_.empty()) {
            ROS_INFO("[INIT] Loading detection data from file: %s", detect_file_path_.c_str());
            loadDetectionsFromFile(detect_file_path_);
        } else {
            ROS_INFO("[INIT] No detection file specified, using fixed box extraction mode");
        }
        
        ROS_INFO("[INIT] ========== Initialization Complete ==========");
        ROS_INFO("[INIT]   Subscribed topic: '%s'", cloud_topic.c_str());
        ROS_INFO("[INIT]   Fixed box range: x[0-20m], y[±5m], z[±3m]");
        ROS_INFO("[INIT]   Publishing topics:");
        ROS_INFO("[INIT]     - /box_pointcloud (detection box points)");
        ROS_INFO("[INIT]     - TF: map -> <cloud_frame_id> (identity)");
        ROS_INFO("[INIT] =============================================");
        ROS_INFO("[INIT] Waiting for point cloud messages on topic '%s'...", cloud_topic.c_str());
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        ROS_INFO("[CALLBACK] ===== Received point cloud message =====");
        ROS_INFO("[CALLBACK]   Topic: (subscribed topic)");
        ROS_INFO("[CALLBACK]   Frame ID: %s", msg->header.frame_id.c_str());
        ROS_INFO("[CALLBACK]   Timestamp: %.6f", msg->header.stamp.toSec());
        ROS_INFO("[CALLBACK]   Width: %u, Height: %u", msg->width, msg->height);
        ROS_INFO("[CALLBACK]   Point step: %u, Row step: %u", msg->point_step, msg->row_step);
        ROS_INFO("[CALLBACK]   Data size: %zu bytes", msg->data.size());
        
        // 设置当前点云（需要锁保护）
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            current_cloud_ = msg;
        }
        
        // 调用处理函数（处理函数内部会获取锁）
        processData();
    }
    
    void loadDetectionsFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            ROS_ERROR_STREAM("Failed to open detection file: " << filename);
            return;
        }
        
        ROS_INFO_STREAM("Loading detection data from file: " << filename);
        
        std::string line;
        int count = 0;
        int skipped = 0;
        
        while (std::getline(file, line)) {
            // 跳过空行
            if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
                continue;
            }
            
            // 跳过注释行（以#开头）
            std::string trimmed = line;
            trimmed.erase(0, trimmed.find_first_not_of(" \t"));
            if (trimmed.empty() || trimmed[0] == '#') {
                continue;
            }
            
            // 移除方括号和多余字符
            line.erase(std::remove(line.begin(), line.end(), '['), line.end());
            line.erase(std::remove(line.begin(), line.end(), ']'), line.end());
            
            // 将逗号替换为空格
            std::replace(line.begin(), line.end(), ',', ' ');
            
            std::istringstream iss(line);
            std_msgs::Float64MultiArray detectMsg;
            double value;
            
            // 读取该行的所有数值
            while (iss >> value) {
                detectMsg.data.push_back(value);
            }
            
            // 确保数据格式正确：至少有时间戳，且后续数据是9的倍数
            if (detectMsg.data.size() >= 1 && (detectMsg.data.size() - 1) % 9 == 0) {
                std::lock_guard<std::mutex> lock(detect_mutex_);
                detect_queue_.push_back(detectMsg);
                count++;
            } else {
                skipped++;
            }
        }
        
        file.close();
        ROS_INFO("[LOAD] Successfully loaded %d frames of detection data (skipped %d lines)", count, skipped);
        ROS_INFO("[LOAD] detect_queue_ size: %zu", detect_queue_.size());
    }
    
    void processData() {
        ROS_INFO("[PROCESS] ===== Entering processData() =====");
        
        // 获取点云数据的副本（避免长时间持有锁）
        sensor_msgs::PointCloud2::ConstPtr cloud_ptr;
        {
            std::lock_guard<std::mutex> lock1(cloud_mutex_);
            ROS_INFO("[PROCESS] Acquired cloud_mutex");
            if (!current_cloud_) {
                ROS_WARN("[PROCESS] No cloud data available");
                return;
            }
            cloud_ptr = current_cloud_;  // 复制shared_ptr，不需要持有锁
            ROS_INFO("[PROCESS] current_cloud_ is valid");
        }
        ROS_INFO("[PROCESS] Released cloud_mutex");
        
        // 获取检测数据的副本
        std::deque<std_msgs::Float64MultiArray> detect_queue_copy;
        {
            std::lock_guard<std::mutex> lock2(detect_mutex_);
            ROS_INFO("[PROCESS] Acquired detect_mutex");
            if (detect_queue_.empty()) {
                ROS_WARN("[PROCESS] No detection data available (detect_queue_ size: %zu)", detect_queue_.size());
                return;
            }
            detect_queue_copy = detect_queue_;  // 复制队列
            ROS_INFO("[PROCESS] detect_queue_ has %zu elements", detect_queue_.size());
        }
        ROS_INFO("[PROCESS] Released detect_mutex");
        
        frame_count_++;
        ROS_INFO("[PROCESS] ========== Processing frame %d ==========", frame_count_);
        
        // 获取点云时间戳（转换为秒）
        double cloud_time = cloud_ptr->header.stamp.toSec();
        ROS_INFO("[PROCESS] Cloud timestamp: %.6f", cloud_time);
        ROS_INFO("[PROCESS] Cloud frame_id: %s, width: %u, height: %u", 
                 cloud_ptr->header.frame_id.c_str(), 
                 cloud_ptr->width, cloud_ptr->height);
        
        // 转换点云
        PointCloudXYZI::Ptr cloud(new PointCloudXYZI);
        try {
            pcl::fromROSMsg(*cloud_ptr, *cloud);
            ROS_INFO("[PROCESS] Converted to PCL: %zu points", cloud->size());
        } catch (const std::exception& e) {
            ROS_ERROR("[PROCESS] Failed to convert ROS message to PCL: %s", e.what());
            return;
        }
        
        if (cloud->empty()) {
            ROS_WARN("[PROCESS] Frame %d: Empty point cloud received (width=%u, height=%u)", 
                     frame_count_, cloud_ptr->width, cloud_ptr->height);
            return;
        }
        
        // 查找时间戳最接近的检测数据
        std_msgs::Float64MultiArray detect;
        bool found_match = false;
        double min_time_diff = std::numeric_limits<double>::max();
        int matched_detect_idx = -1;
        
        ROS_INFO("[PROCESS] Searching in %zu detection frames...", detect_queue_copy.size());
        
        // 从队列中查找匹配的检测数据
        for (size_t idx = 0; idx < detect_queue_copy.size(); ++idx) {
            if (detect_queue_copy[idx].data.empty()) continue;
            
            double detect_time = detect_queue_copy[idx].data[0]; // 检测数据的时间戳
            double time_diff = fabs(detect_time - cloud_time);
            
            ROS_DEBUG("[PROCESS] Detection[%zu]: time=%.6f, diff=%.6f", idx, detect_time, time_diff);
            
            if (time_diff < sync_tolerance_ && time_diff < min_time_diff) {
                min_time_diff = time_diff;
                detect = detect_queue_copy[idx];
                matched_detect_idx = idx;
                found_match = true;
            }
        }
        
        if (!found_match) {
            ROS_WARN("[PROCESS] Frame %d: No matching detection data (cloud_time=%.6f, tolerance=%.3f)", 
                     frame_count_, cloud_time, sync_tolerance_);
            return;
        }
        
        ROS_INFO("[PROCESS] Matched detection[%d]: time_diff=%.6f, detect_time=%.6f", 
                 matched_detect_idx, min_time_diff, detect.data[0]);
        
        // 处理检测数据
        // 检测数据格式：[timestamp, type, x, y, z, l, w, h, yaw, score, ...]
        // 每个物体需要9个值，加上时间戳，至少需要10个元素才能有一个完整的物体
        if (detect.data.size() < 1) {
            ROS_WARN("[PROCESS] Detection data empty");
            return;
        }
        
        // 检查数据格式：除了时间戳外，剩余数据必须是9的倍数（每个物体9个值）
        if ((detect.data.size() - 1) % 9 != 0) {
            ROS_WARN("[PROCESS] Detection data format error: %zu elements (should be 1 + 9*N, where N is number of objects)", 
                     detect.data.size());
            ROS_WARN("[PROCESS]   Format: [timestamp, type, x, y, z, l, w, h, yaw, score, ...]");
            return;
        }
        
        // 解析检测数据：格式为 [timestamp, type, x, y, z, l, w, h, yaw, score, ...]
        int ob_num = (detect.data.size() - 1) / 9;
        ROS_INFO("[PROCESS] Found %d objects in detection data (total elements: %zu)", ob_num, detect.data.size());
        
        if (ob_num == 0) {
            ROS_DEBUG("[PROCESS] No objects in detection data (only timestamp)");
            return;
        }
        
        // 固定范围：x前方0-20m，y左右5m，z上下3m
        double x_min = 0.0;
        double x_max = 20.0;
        double y_min = -5.0;
        double y_max = 5.0;
        double z_min = -3.0;
        double z_max = 3.0;
        
        // 根据参数决定处理模式
        std::vector<int> selected_indices;
        
        if (publish_all_boxes_) {
            // 发布所有检测框，但只选择 label=0 的目标
            ROS_INFO("[PROCESS] Publishing detection boxes with label=0 (%d objects total)", ob_num);
            for (int i = 0; i < ob_num; i++) {
                double type = detect.data[1 + 9 * i];
                // 只选择 label=0 的目标
                if (type == 0.0) {
                    selected_indices.push_back(i);
                    double x = detect.data[2 + 9 * i];
                    double y = detect.data[3 + 9 * i];
                    double z = detect.data[4 + 9 * i];
                    ROS_DEBUG("[PROCESS] Selected object[%d] (label=0): pos=(%.2f, %.2f, %.2f)", i, x, y, z);
                }
            }
            ROS_INFO("[PROCESS] Found %zu objects with label=0", selected_indices.size());
        } else {
            // 只发布前方固定范围内的框，且只选择 label=0 的目标
            ROS_INFO("[PROCESS] Searching for objects with label=0 in fixed range: x[%.1f-%.1f], y[%.1f-%.1f], z[%.1f-%.1f]",
                     x_min, x_max, y_min, y_max, z_min, z_max);
            
            for (int i = 0; i < ob_num; i++) {
                double type = detect.data[1 + 9 * i];
                double x = detect.data[2 + 9 * i];
                double y = detect.data[3 + 9 * i];
                double z = detect.data[4 + 9 * i];
                double l = detect.data[5 + 9 * i];
                double w = detect.data[6 + 9 * i];
                double h = detect.data[7 + 9 * i];
                double score = detect.data[9 + 9 * i];
                
                ROS_DEBUG("[PROCESS] Object[%d]: type=%.0f, pos=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f), score=%.2f",
                         i, type, x, y, z, l, w, h, score);
                
                // 只选择 label=0 的目标
                if (type != 0.0) {
                    continue;
                }
                
                // 检查目标位置是否在固定范围内
                if (x >= x_min && x <= x_max &&
                    y >= y_min && y <= y_max &&
                    z >= z_min && z <= z_max) {
                    selected_indices.push_back(i);
                    ROS_INFO("[PROCESS] Selected object[%d] (label=0): pos=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f), score=%.2f",
                             i, x, y, z, l, w, h, score);
                    break; // 找到第一个符合条件的就退出
                }
            }
        }
        
        if (selected_indices.empty()) {
            ROS_DEBUG("[PROCESS] Frame %d: No object found", frame_count_);
            return;
        }
        
        // 处理选中的框（可能是一个或多个）
        PointCloudXYZI::Ptr all_box_cloud(new PointCloudXYZI);
        visualization_msgs::MarkerArray marker_array;
        int primary_idx = selected_indices[0];  // 第一个框作为主要框（用于配准等）
        
        for (size_t box_idx = 0; box_idx < selected_indices.size(); box_idx++) {
            int selected_idx = selected_indices[box_idx];
            
            // 提取检测框信息
            double obj_x = detect.data[2 + 9 * selected_idx];
            double obj_y = detect.data[3 + 9 * selected_idx];
            double obj_z = detect.data[4 + 9 * selected_idx];
            double obj_l = detect.data[5 + 9 * selected_idx];
            double obj_w = detect.data[6 + 9 * selected_idx];
            double obj_h = detect.data[7 + 9 * selected_idx];
            double obj_yaw = detect.data[8 + 9 * selected_idx];
            
            ROS_INFO("[PROCESS] Processing box[%zu/%zu]: center=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f), yaw=%.2f",
                     box_idx + 1, selected_indices.size(), obj_x, obj_y, obj_z, obj_l, obj_w, obj_h, obj_yaw);
            
            // 提取检测框内的点云
            PointCloudXYZI::Ptr box_cloud = extractBoxCloud(cloud, obj_x, obj_y, obj_z, obj_l, obj_w, obj_h, obj_yaw);
            
            if (box_cloud->empty()) {
                ROS_DEBUG("[PROCESS] Box[%zu]: No points in detection box", box_idx);
                continue;
            }
            
            // 合并点云（用于发布）
            *all_box_cloud += *box_cloud;
            
            // 如果启用发布Marker，添加到MarkerArray
            if (publish_original_and_box_) {
                visualization_msgs::Marker box_marker;
                box_marker.header = cloud_ptr->header;
                box_marker.header.frame_id = cloud_ptr->header.frame_id;
                box_marker.ns = "detection_box";
                box_marker.id = frame_count_ * 1000 + box_idx;  // 确保每个框有唯一ID
                box_marker.type = visualization_msgs::Marker::CUBE;
                box_marker.action = visualization_msgs::Marker::ADD;
                
                // 设置框的位置和姿态
                box_marker.pose.position.x = obj_x;
                box_marker.pose.position.y = obj_y;
                box_marker.pose.position.z = obj_z;
                
                // 设置旋转（绕z轴旋转yaw）
                tf::Quaternion q;
                q.setRPY(0, 0, obj_yaw);
                box_marker.pose.orientation.x = q.x();
                box_marker.pose.orientation.y = q.y();
                box_marker.pose.orientation.z = q.z();
                box_marker.pose.orientation.w = q.w();
                
                // 设置框的尺寸
                box_marker.scale.x = obj_l;
                box_marker.scale.y = obj_w;
                box_marker.scale.z = obj_h;
                
                // 设置颜色（半透明绿色）
                box_marker.color.r = 0.0;
                box_marker.color.g = 1.0;
                box_marker.color.b = 0.0;
                box_marker.color.a = 0.3;
                
                marker_array.markers.push_back(box_marker);
            }
        }
        
        if (all_box_cloud->empty()) {
            ROS_WARN("[PROCESS] Frame %d: No points in any detection box", frame_count_);
            return;
        }
        
        // 使用第一个框的信息进行后续处理（配准等）
        double obj_x = detect.data[2 + 9 * primary_idx];
        double obj_y = detect.data[3 + 9 * primary_idx];
        double obj_z = detect.data[4 + 9 * primary_idx];
        double obj_l = detect.data[5 + 9 * primary_idx];
        double obj_w = detect.data[6 + 9 * primary_idx];
        double obj_h = detect.data[7 + 9 * primary_idx];
        double obj_yaw = detect.data[8 + 9 * primary_idx];
        
        ROS_INFO("[PROCESS] Extracted %zu points from %zu detection box(es)", all_box_cloud->size(), selected_indices.size());
        
        // 从检测数据构建当前帧的变换矩阵（物体系到雷达系）- 使用第一个框
        Eigen::Matrix4f Tn = buildTransformationFromDetection(obj_x, obj_y, obj_z, obj_yaw);
        ROS_INFO("[PROCESS] Built transformation Tn from primary box: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f", 
                 obj_x, obj_y, obj_z, obj_yaw);
        
        // 将第一个框的点云转换到框的局部坐标系（用于配准）
        PointCloudXYZI::Ptr primary_box_cloud = extractBoxCloud(cloud, obj_x, obj_y, obj_z, obj_l, obj_w, obj_h, obj_yaw);
        PointCloudXYZI::Ptr box_cloud_local = transformToBoxLocalFrame(primary_box_cloud, obj_x, obj_y, obj_z, obj_yaw);
        
        // 如果启用配准功能，从第二帧开始进行配准
        PointCloudXYZI::Ptr aligned_cloud(new PointCloudXYZI);
        Eigen::Matrix4f current_transformation;
        bool registration_success = false;
        
        if (enable_registration_ && has_prev_frame_ && prev_box_cloud_local_ && !prev_box_cloud_local_->empty()) {
            ROS_INFO("[REGISTRATION] Starting registration: accumulated_frame=%zu points, current_frame=%zu points", 
                     prev_box_cloud_local_->size(), box_cloud_local->size());
            
            // 使用累积的配准后点云（prev_box_cloud_local_）作为配准目标
            // 这个点云已经在局部坐标系中，包含了之前所有帧配准后的点云
            
            // 计算初始估计：T0^(-1) * Tn（从当前帧物体系到第一帧物体系）
            Eigen::Matrix4f initial_guess;
            if (has_T0_) {
                initial_guess = Eigen::Matrix4f::Identity();
                // initial_guess = T0_.inverse() * Tn;
                // ROS_INFO("[REGISTRATION] Using detection-based initial guess: T0^(-1) * Tn");
            } else {
                initial_guess = Eigen::Matrix4f::Identity();
                ROS_WARN("[REGISTRATION] T0 not available, using identity matrix as initial guess");
            }
            
            // 在框的局部坐标系中进行配准
            // 当前帧的点云（box_cloud_local）配准到累积的点云（prev_box_cloud_local_）
            registration_success = performRegistration(prev_box_cloud_local_, box_cloud_local, aligned_cloud, current_transformation, initial_guess);
            
            if (registration_success) {
                ROS_INFO("[REGISTRATION] Registration successful: aligned=%zu points", aligned_cloud->size());
                
                // 发布配准后的点云（当前帧框内点云对齐到上一帧坐标系）
                sensor_msgs::PointCloud2 aligned_msg;
                pcl::toROSMsg(*aligned_cloud, aligned_msg);
                aligned_msg.header = cloud_ptr->header;
                aligned_msg.header.frame_id = cloud_ptr->header.frame_id;
                pub_aligned_cloud_.publish(aligned_msg);
                ROS_INFO("[PUBLISH] Published /aligned_pointcloud: %zu points", aligned_cloud->size());
                
                // 发布当前帧的原始完整点云（不是框内的，是整个原始点云）
                sensor_msgs::PointCloud2 unmatched_msg;
                pcl::toROSMsg(*cloud, unmatched_msg);
                unmatched_msg.header = cloud_ptr->header;
                unmatched_msg.header.frame_id = cloud_ptr->header.frame_id;
                pub_unmatched_cloud_.publish(unmatched_msg);
                ROS_INFO("[PUBLISH] Published /unmatched_pointcloud: %zu points (original full cloud)", cloud->size());
                
                // 将配准后的点云追加到累积的点云中（累积之前所有帧的配准结果）
                // aligned_cloud 已经在局部坐标系中（对齐到累积点云的坐标系）
                // 直接累积到 prev_box_cloud_local_ 中，作为下一帧的配准目标
                *prev_box_cloud_local_ += *aligned_cloud;
                ROS_INFO("[SAVE] Appended aligned cloud (%zu points) to accumulated cloud. Total accumulated: %zu points", 
                         aligned_cloud->size(), prev_box_cloud_local_->size());
                
                // 保存当前帧的变换矩阵，作为下一帧的初始估计
                prev_transformation_ = current_transformation;
                has_prev_transformation_ = true;
                ROS_INFO("[SAVE] Saved transformation matrix for next frame initial guess");
                
                // 保存当前帧的框位姿，供下一帧转换使用
                prev_box_x_ = obj_x;
                prev_box_y_ = obj_y;
                prev_box_z_ = obj_z;
                prev_box_yaw_ = obj_yaw;
                has_prev_box_pose_ = true;
            } else {
                ROS_WARN("[REGISTRATION] Registration failed, skipping accumulation");
                // 即使配准失败，也保存当前帧的框位姿
                prev_box_x_ = obj_x;
                prev_box_y_ = obj_y;
                prev_box_z_ = obj_z;
                prev_box_yaw_ = obj_yaw;
                has_prev_box_pose_ = true;
            }
            has_prev_frame_ = true;
        } else if (enable_registration_ && !has_prev_frame_) {
            ROS_INFO("[REGISTRATION] First frame, skipping registration");
            // 第一帧：保存检测位姿 T0（物体系到雷达系）
            T0_ = Tn;
            has_T0_ = true;
            ROS_INFO("[SAVE] Saved first frame detection pose T0 (object frame to lidar frame)");
            ROS_INFO("[SAVE] T0: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f", obj_x, obj_y, obj_z, obj_yaw);
            
            // 第一帧：将框内点云转换到框的局部坐标系后初始化累积点云
            *prev_box_cloud_local_ = *box_cloud_local;
            has_prev_frame_ = true;
            
            // 保存第一帧的框位姿
            prev_box_x_ = obj_x;
            prev_box_y_ = obj_y;
            prev_box_z_ = obj_z;
            prev_box_yaw_ = obj_yaw;
            has_prev_box_pose_ = true;
            
            // 第一帧没有变换矩阵，下一帧将使用检测数据计算的初始估计
            prev_transformation_.setIdentity();
            has_prev_transformation_ = false;
            ROS_INFO("[SAVE] Initialized accumulated cloud with first frame (%zu points) in local frame", box_cloud_local->size());
        } else {
            // 非配准模式或没有上一帧：保存当前帧的框位姿供下一帧使用
            prev_box_x_ = obj_x;
            prev_box_y_ = obj_y;
            prev_box_z_ = obj_z;
            prev_box_yaw_ = obj_yaw;
            has_prev_box_pose_ = true;
        }
        
        // 发布框内的点云（原始雷达坐标系中的点云，可能是多个框合并的点云）
        sensor_msgs::PointCloud2 box_msg;
        try {
            pcl::toROSMsg(*all_box_cloud, box_msg);
            box_msg.header = cloud_ptr->header;
            box_msg.header.frame_id = cloud_ptr->header.frame_id;
            pub_box_cloud_.publish(box_msg);
            ROS_INFO("[PUBLISH] Published /box_pointcloud: %zu points from %zu box(es) (original in lidar frame), frame_id=%s", 
                     all_box_cloud->size(), selected_indices.size(), box_msg.header.frame_id.c_str());
        } catch (const std::exception& e) {
            ROS_ERROR("[PUBLISH] Failed to publish box cloud: %s", e.what());
            return;
        }
        
        // 如果启用，发布原始点云和检测框
        if (publish_original_and_box_) {
            // 发布原始点云
            sensor_msgs::PointCloud2 original_msg;
            try {
                pcl::toROSMsg(*cloud, original_msg);
                original_msg.header = cloud_ptr->header;
                original_msg.header.frame_id = cloud_ptr->header.frame_id;
                pub_original_cloud_.publish(original_msg);
                ROS_DEBUG("[PUBLISH] Published /original_pointcloud: %zu points", cloud->size());
            } catch (const std::exception& e) {
                ROS_ERROR("[PUBLISH] Failed to publish original cloud: %s", e.what());
            }
            
            // 发布检测框MarkerArray（包含所有框）
            if (!marker_array.markers.empty()) {
                pub_box_marker_array_.publish(marker_array);
                ROS_DEBUG("[PUBLISH] Published /detection_box_marker_array: %zu markers", marker_array.markers.size());
            }
            
            // 为了兼容性，也发布单个Marker（只发布第一个框）
            if (selected_indices.size() > 0) {
                visualization_msgs::Marker box_marker;
                box_marker.header = cloud_ptr->header;
                box_marker.header.frame_id = cloud_ptr->header.frame_id;
                box_marker.ns = "detection_box";
                box_marker.id = frame_count_;
                box_marker.type = visualization_msgs::Marker::CUBE;
                box_marker.action = visualization_msgs::Marker::ADD;
                
                // 设置框的位置和姿态
                box_marker.pose.position.x = obj_x;
                box_marker.pose.position.y = obj_y;
                box_marker.pose.position.z = obj_z;
                
                // 设置旋转（绕z轴旋转yaw）
                tf::Quaternion q;
                q.setRPY(0, 0, obj_yaw);
                box_marker.pose.orientation.x = q.x();
                box_marker.pose.orientation.y = q.y();
                box_marker.pose.orientation.z = q.z();
                box_marker.pose.orientation.w = q.w();
                
                // 设置框的尺寸
                box_marker.scale.x = obj_l;
                box_marker.scale.y = obj_w;
                box_marker.scale.z = obj_h;
                
                // 设置颜色（半透明绿色）
                box_marker.color.r = 0.0;
                box_marker.color.g = 1.0;
                box_marker.color.b = 0.0;
                box_marker.color.a = 0.3;
                
                pub_box_marker_.publish(box_marker);
                ROS_DEBUG("[PUBLISH] Published /detection_box_marker: center=(%.2f, %.2f, %.2f), size=(%.2f, %.2f, %.2f)", 
                         obj_x, obj_y, obj_z, obj_l, obj_w, obj_h);
            }
        }
        
        // 发布map到frame_id的静态tf（identity变换，即0变换）
        std::string cloud_frame_id = cloud_ptr->header.frame_id;
        try {
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(0, 0, 0));
            transform.setRotation(tf::Quaternion(0, 0, 0, 1));
            tf_broadcaster_.sendTransform(tf::StampedTransform(transform, 
                                                               cloud_ptr->header.stamp,
                                                               "map", 
                                                               cloud_frame_id));
            ROS_DEBUG("[TF] Published TF: map -> %s (identity transform)", cloud_frame_id.c_str());
        } catch (const std::exception& e) {
            ROS_ERROR("[TF] Failed to publish TF: %s", e.what());
        }
        
        success_count_++;
        ROS_INFO("[PROCESS] Frame %d completed successfully", frame_count_);
        ROS_INFO("[PROCESS] ========================================");
        
        if (frame_count_ % 10 == 0) {
            ROS_INFO("[STATS] Processed %d frames, success: %d (%.1f%%), last box: %zu points from %zu box(es)",
                     frame_count_, success_count_, 100.0 * success_count_ / frame_count_, all_box_cloud->size(), selected_indices.size());
        }
        
        // 暂时注释掉聚类部分
        /*
        // 对框内点云进行聚类
        PointCloudXYZI::Ptr target_cloud = clusterAndExtract(box_cloud);
        
        if (target_cloud->empty()) {
            ROS_WARN("No points after clustering");
            return;
        }
        
        // 发布聚类后的目标点云
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*target_cloud, output_msg);
        output_msg.header = current_cloud_->header;
        output_msg.header.frame_id = current_cloud_->header.frame_id;
        pub_target_cloud_.publish(output_msg);
        
        ROS_INFO("Published clustered target cloud with %zu points", target_cloud->size());
        */
    }
    
    // 提取检测框内的点云（使用检测框的位置、尺寸和旋转）
    PointCloudXYZI::Ptr extractBoxCloud(const PointCloudXYZI::Ptr& cloud,
                                        double x, double y, double z,
                                        double l, double w, double h, double yaw) {
        PointCloudXYZI::Ptr box_cloud(new PointCloudXYZI);
        
        ROS_INFO("[EXTRACT] Extracting points from detection box:");
        ROS_INFO("[EXTRACT]   Center: (%.2f, %.2f, %.2f)", x, y, z);
        ROS_INFO("[EXTRACT]   Size: (%.2f, %.2f, %.2f)", l, w, h);
        ROS_INFO("[EXTRACT]   Yaw: %.2f rad (%.2f deg)", yaw, yaw * 180.0 / M_PI);
        
        // 框的半尺寸
        double half_l = l / 2.0;
        double half_w = w / 2.0;
        double half_h = h / 2.0;
        
        // 计算旋转矩阵（绕z轴旋转yaw，用于将点从雷达系转换到框的局部坐标系）
        // yaw是雷达系到框系的转角，所以从雷达系到框系需要旋转yaw
        double cos_yaw = cos(-yaw);
        double sin_yaw = sin(-yaw);
        
        // 遍历所有点，判断是否在旋转框内
        int filtered_count = 0;
        for (const auto& pt : cloud->points) {
            // 将点转换到框中心为原点的坐标系（相对于框中心的偏移）
            double dx = pt.x - x;
            double dy = pt.y - y;
            double dz = pt.z - z;
            
            // 旋转到框的局部坐标系（从雷达系旋转yaw到框系）
            // 旋转矩阵：R_z(yaw) = [cos(yaw) -sin(yaw); sin(yaw) cos(yaw)]
            double local_x = dx * cos_yaw - dy * sin_yaw;
            double local_y = dx * sin_yaw + dy * cos_yaw;
            double local_z = dz;
            
            // 判断点是否在框内（局部坐标系中，框的中心在原点）
            if (local_x >= -half_l && local_x <= half_l &&
                local_y >= -half_w && local_y <= half_w &&
                local_z >= -half_h && local_z <= half_h) {
                box_cloud->push_back(pt);
                filtered_count++;
            }
        }
        
        ROS_INFO("[EXTRACT] Extracted %d/%zu points from detection box", filtered_count, cloud->size());
        
        return box_cloud;
    }
    
    // 从检测数据（xyz, yaw）构建变换矩阵（物体系到雷达系）
    Eigen::Matrix4f buildTransformationFromDetection(double x, double y, double z, double yaw) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        
        // 旋转矩阵（绕z轴旋转yaw角度）
        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);
        T(0, 0) = cos_yaw;
        T(0, 1) = -sin_yaw;
        T(1, 0) = sin_yaw;
        T(1, 1) = cos_yaw;
        
        // 平移向量
        T(0, 3) = x;
        T(1, 3) = y;
        T(2, 3) = z;
        
        return T;
    }
    
    // 将点云从雷达系转换到框的局部坐标系（以框中心为原点）
    PointCloudXYZI::Ptr transformToBoxLocalFrame(const PointCloudXYZI::Ptr& cloud,
                                                  double box_x, double box_y, double box_z, double box_yaw) {
        PointCloudXYZI::Ptr local_cloud(new PointCloudXYZI);
        local_cloud->reserve(cloud->size());
        
        // 计算旋转矩阵（从雷达系到框系：先平移，再旋转-yaw）
        double cos_yaw = cos(-box_yaw);
        double sin_yaw = sin(-box_yaw);
        
        for (const auto& pt : cloud->points) {
            PointType local_pt;
            
            // 先平移到框中心为原点
            double dx = pt.x - box_x;
            double dy = pt.y - box_y;
            double dz = pt.z - box_z;
            
            // 再旋转到框的局部坐标系
            local_pt.x = dx * cos_yaw - dy * sin_yaw;
            local_pt.y = dx * sin_yaw + dy * cos_yaw;
            local_pt.z = dz;
            local_pt.intensity = pt.intensity;
            
            local_cloud->push_back(local_pt);
        }
        
        ROS_INFO("[TRANSFORM] Transformed %zu points to box local frame (center: %.2f, %.2f, %.2f, yaw: %.2f)", 
                 local_cloud->size(), box_x, box_y, box_z, box_yaw);
        
        return local_cloud;
    }
    
    // 配准功能：将当前帧与上一帧进行匹配
    bool performRegistration(const PointCloudXYZI::Ptr& prev_cloud,    // 上一帧框内点云（目标）
                            const PointCloudXYZI::Ptr& curr_cloud,     // 当前帧框内点云（源）
                            PointCloudXYZI::Ptr& aligned_cloud,         // 输出：配准后的当前帧点云
                            Eigen::Matrix4f& final_transformation,      // 输出：最终的变换矩阵
                            const Eigen::Matrix4f& initial_guess) {     // 输入：初始变换估计
        if (prev_cloud->empty() || curr_cloud->empty()) {
            ROS_WARN("[REGISTRATION] Empty point cloud: prev=%zu, curr=%zu", prev_cloud->size(), curr_cloud->size());
            return false;
        }
        
        ROS_INFO("[REGISTRATION] Starting ICP: prev_frame=%zu points, current_frame=%zu points", 
                 prev_cloud->size(), curr_cloud->size());
        
        // 使用ICP进行配准
        // 将当前帧（curr_cloud）对齐到上一帧（prev_cloud）
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setInputSource(curr_cloud);       // 当前帧作为源（要变换的点云）
        icp.setInputTarget(prev_cloud);       // 上一帧作为目标（参考点云）
        icp.setMaxCorrespondenceDistance(0.1); // 最大对应距离（米）
        icp.setMaximumIterations(50);          // 最大迭代次数
        icp.setTransformationEpsilon(1e-6);    // 变换收敛阈值
        icp.setEuclideanFitnessEpsilon(1e-6);  // 欧氏距离收敛阈值
        
        // 使用传入的初始估计（基于检测数据计算的 T0^(-1) * Tn）
        ROS_INFO("[REGISTRATION] Using provided initial guess from detection data");
        
        // 执行配准（通过 align 的第二个参数设置初始变换）
        PointCloudXYZI::Ptr aligned(new PointCloudXYZI);
        icp.align(*aligned, initial_guess);
        
        if (!icp.hasConverged()) {
            ROS_WARN("[REGISTRATION] ICP did not converge");
            return false;
        }
        
        // 获取变换矩阵
        final_transformation = icp.getFinalTransformation();
        ROS_INFO("[REGISTRATION] ICP converged after %d iterations", icp.nr_iterations_);
        ROS_INFO("[REGISTRATION] Transformation matrix:");
        ROS_INFO("[REGISTRATION]   [%.4f %.4f %.4f %.4f]", 
                 final_transformation(0,0), final_transformation(0,1), final_transformation(0,2), final_transformation(0,3));
        ROS_INFO("[REGISTRATION]   [%.4f %.4f %.4f %.4f]", 
                 final_transformation(1,0), final_transformation(1,1), final_transformation(1,2), final_transformation(1,3));
        ROS_INFO("[REGISTRATION]   [%.4f %.4f %.4f %.4f]", 
                 final_transformation(2,0), final_transformation(2,1), final_transformation(2,2), final_transformation(2,3));
        ROS_INFO("[REGISTRATION]   [%.4f %.4f %.4f %.4f]", 
                 final_transformation(3,0), final_transformation(3,1), final_transformation(3,2), final_transformation(3,3));
        
        double fitness_score = icp.getFitnessScore();
        ROS_INFO("[REGISTRATION] Fitness score: %.6f", fitness_score);
        
        // 配准后的点云（当前帧对齐到上一帧坐标系）
        *aligned_cloud = *aligned;
        
        return true;
    }
    
    // 原来的检测框提取函数（暂时保留，但不再使用）
    /*
    PointCloudXYZI::Ptr extractBoxCloud(const PointCloudXYZI::Ptr& cloud,
                                        double x, double y, double z,
                                        double l, double w, double h, double yaw) {
        PointCloudXYZI::Ptr box_cloud(new PointCloudXYZI);
        
        // 使用PCL的CropBox过滤器提取旋转框内的点
        pcl::CropBox<PointType> crop_box;
        crop_box.setInputCloud(cloud);
        
        // 设置框的边界（在框的局部坐标系中，中心在原点）
        Eigen::Vector4f min_pt, max_pt;
        min_pt << -l/2.0, -w/2.0, -h/2.0, 1.0;
        max_pt << l/2.0, w/2.0, h/2.0, 1.0;
        
        crop_box.setMin(min_pt);
        crop_box.setMax(max_pt);
        
        // 构建从局部坐标系到全局坐标系的变换矩阵
        // 局部坐标系：框的中心在原点，x轴沿长度方向，y轴沿宽度方向，z轴沿高度方向
        // 全局坐标系：点云的坐标系
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        
        // 旋转矩阵（绕z轴旋转yaw角度）
        Eigen::Matrix3f rotation;
        rotation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        transform.linear() = rotation;
        
        // 平移（框的中心位置）
        transform.translation() = Eigen::Vector3f(x, y, z);
        
        // 设置变换矩阵
        crop_box.setTransform(transform);
        
        // 提取点云
        crop_box.filter(*box_cloud);
        
        return box_cloud;
    }
    */
    
    // 聚类函数暂时注释掉
    /*
    PointCloudXYZI::Ptr clusterAndExtract(const PointCloudXYZI::Ptr& cloud) {
        if (cloud->empty()) {
            return PointCloudXYZI::Ptr(new PointCloudXYZI);
        }
        
        // 创建KdTree用于搜索
        pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
        tree->setInputCloud(cloud);
        
        // 欧氏聚类
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointType> ec;
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
        
        if (cluster_indices.empty()) {
            ROS_WARN("No clusters found");
            return PointCloudXYZI::Ptr(new PointCloudXYZI);
        }
        
        // 选择最大的聚类作为目标
        size_t max_size = 0;
        int max_idx = -1;
        for (size_t i = 0; i < cluster_indices.size(); i++) {
            if (cluster_indices[i].indices.size() > max_size) {
                max_size = cluster_indices[i].indices.size();
                max_idx = i;
            }
        }
        
        if (max_idx == -1) {
            return PointCloudXYZI::Ptr(new PointCloudXYZI);
        }
        
        // 提取最大聚类的点云
        PointCloudXYZI::Ptr target_cloud(new PointCloudXYZI);
        for (const auto& idx : cluster_indices[max_idx].indices) {
            target_cloud->push_back(cloud->points[idx]);
        }
        
        ROS_INFO("Found %zu clusters, selected largest with %zu points",
                 cluster_indices.size(), target_cloud->size());
        
        return target_cloud;
    }
    */
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "target_pointcloud_extractor");
    
    ROS_INFO("[MAIN] Starting target_pointcloud_extractor node...");
    
    try {
        TargetPointCloudExtractor extractor;
        ROS_INFO("[MAIN] Node initialized successfully, entering spin loop...");
        ros::spin();
    } catch (const std::exception& e) {
        ROS_FATAL("[MAIN] Fatal error: %s", e.what());
        return 1;
    }
    
    ROS_INFO("[MAIN] Node shutting down");
    return 0;
}

