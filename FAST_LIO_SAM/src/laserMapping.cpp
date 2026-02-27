// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <limits>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

// gstam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <Scancontext.h>

#include "factorgraph_opt.h"
#include "tracked_object.h"
// gnss
#include "GNSS_Processing.hpp"
#include "sensor_msgs/NavSatFix.h"

// save map
#include "fast_lio_sam/save_map.h"
#include "fast_lio_sam/save_pose.h"

// save data in kitti / tum format 
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>

// using namespace gtsam;

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
double current_mot_tracking_time_ms = 0.0, current_mot_graph_time_ms = 0.0, current_global_graph_time_ms = 0.0, current_backend_graph_time_ms = 0.0;
double total_mot_tracking_time_ms = 0.0, total_mot_graph_time_ms = 0.0, total_global_graph_time_ms = 0.0, total_backend_graph_time_ms = 0.0;
int mot_tracking_frame_count = 0, backend_graph_frame_count = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true, saveObjectResult = false;
/**************************/

float res_last[100000] = {0.0}; //残差，点到面距离平方和
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

gtsam::Pose3 Affine3f2Pose3(Eigen::Affine3f T);
PointTypePose trans2PointTypePose(float transformIn[]);
void saveObjectTrackingResult();
void accumulateTrackedObjectCloud();
void saveTrackedObjectCloudsIfNeeded(bool force_save);
int addObjectLocalMapFactorsBeforeOptimization();
void accumulateObjectLocalMaps();
void publishObjectLocalMaps();
void saveObjectLocalMapsIfNeeded(bool force_save);
void recordIntermediateFrameForMap(const PointTypePose &currentPose6D);
void flushPendingIntermediateFramesToAnchor(int new_keyframe_index);
void saulioOdomHandler(const nav_msgs::Odometry::ConstPtr &msg);
bool waitSaulioOdom(double target_time, nav_msgs::Odometry &odom_out);
void applySaulioOdomToEskfState(const nav_msgs::Odometry &odom_msg);
void publishFilteredCloudForSaulio();

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_start_time=0,lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0}; // 是否为平面特征点
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool dense_keyframe = true;  // true: 关键帧存储feats_undistort(去畸变完整点云), false: 存储feats_down_body(降采样点云)

vector<vector<int>> pointSearchInd_surf;
vector<BoxPointType> cub_needrm; // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> t_imu_gnss_vec(3, 0.0);
deque<double> time_buffer;               // 记录lidar时间
deque<PointCloudXYZI::Ptr> lidar_buffer; //记录特征提取或间隔采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<nav_msgs::Odometry> saulio_odom_buffer;
mutex saulio_odom_mutex;

// 前端里程计来源: 0 使用当前 FAST-LIO 前端, 1 使用外部 SAULIO 里程计话题
int frontend_odom_mode = 0;
string saulio_odom_topic = "/aft_mapped_to_init";
string saulio_filtered_cloud_topic = "/sat_slam/saulio_filtered_points";
double saulio_odom_time_tolerance = 0.05;
double saulio_odom_wait_timeout = 0.08;
bool saulio_publish_filtered_cloud = true;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，w系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1)); //特征点在地图中对应点的，局部平面参数,w系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //对应点法相量？
PointCloudXYZI::Ptr _featsArray;                                  // ikd-tree中，map需要移除的点云

pcl::VoxelGrid<PointType> downSizeFilterSurf; //单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap;  //未使用

KD_TREE ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d); // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);  // R lidar to imu (imu = r * lidar + t)
V3D t_imu_gnss(Zero3d);      // T IMU to GNSS (GNSS = IMU + t_imu_gnss)

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 状态，噪声维度，输入
state_ikfom state_point;
vect3 pos_lid; // world系下lidar坐标

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/*back end*/
vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames; // 历史所有关键帧的角点集合（降采样）
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         // 历史关键帧位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // 历史关键帧位姿
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

pcl::PointCloud<PointTypePose>::Ptr fastlio_unoptimized_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  存储fastlio 未优化的位姿
pcl::PointCloud<PointTypePose>::Ptr gnss_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); //  gnss 轨迹

struct IntermediateFrameRecord {
    int frame_id = -1;
    double time = 0.0;
    PointTypePose relative_pose_to_anchor;
    pcl::PointCloud<PointType>::Ptr cloud;
};

struct PendingIntermediateFrameRecord {
    int frame_id = -1;
    double time = 0.0;
    PointTypePose absolute_pose_body;
    pcl::PointCloud<PointType>::Ptr cloud;
};

vector<vector<IntermediateFrameRecord>> intermediateFramesPerKeyframe;
vector<PendingIntermediateFrameRecord> pendingIntermediateFrames;
vector<PointTypePose> keyframeAnchorCapturePoses;


// 物体轨迹管理
struct ObjectTrajectory {
    int object_id;
    std::vector<Eigen::Vector3f> positions;  // 历史位置
    bool is_active;  // 是否仍在追踪
    ros::Time last_seen;  // 最后一次观测时间
};
std::map<int, ObjectTrajectory> object_trajectories;  // object_id -> trajectory

// 指定 object_id 的局部点云累计（默认关闭，不影响原有跟踪/建图流程）
int frontBoxTargetObjectId = -1;
int frontBoxPublishMaxFrames = 0;  // 0: disabled, -1: unlimited until exit, >0: save after N valid frames
bool frontBoxAccumulateForPcd = false;
double frontBoxAccumulateRandomOffsetXMin = -0.2;
double frontBoxAccumulateRandomOffsetXMax = 0.2;
double frontBoxAccumulateRandomOffsetYMin = -0.3;
double frontBoxAccumulateRandomOffsetYMax = 0.3;
double frontBoxAccumulateRandomOffsetZMin = -0.3;
double frontBoxAccumulateRandomOffsetZMax = 0.3;
double frontBoxAccumulateRandomYawDegMin = -10.0;
double frontBoxAccumulateRandomYawDegMax = 10.0;
bool frontBoxIcpMap = false;
int frontBoxPublishFrameCount = 0;
int frontBoxSkipNoDetection = 0;
int frontBoxSkipEmptyCloud = 0;
bool frontBoxAccumulatedCloudSaved = false;
bool frontBoxIcpMapSaved = false;
std::ofstream frontBoxLogStream;
PointCloudXYZI::Ptr frontBoxAccumulatedCloud;
PointCloudXYZI::Ptr frontBoxIcpMapCloud;

// 多目标局部点云地图：默认关闭，只在需要验证论文里的目标局部建图思路时启用
struct ObjectLocalMapState {
    ObjectLocalMapState()
        : accumulated_cloud(new PointCloudXYZI()),
          icp_map_cloud(new PointCloudXYZI())
    {}

    PointCloudXYZI::Ptr accumulated_cloud;
    PointCloudXYZI::Ptr icp_map_cloud;
    Eigen::Affine3f anchor_pose_world = Eigen::Affine3f::Identity();
    int anchor_frame_id = -1;
    int valid_frames = 0;
    int skipped_empty_cloud = 0;
    int icp_check_count = 0;
    int icp_backend_candidate_count = 0;
    int icp_reject_count = 0;
    int backend_factor_count = 0;
    bool has_anchor_pose = false;
    bool has_backend_candidate = false;
    Eigen::Affine3f latest_backend_candidate_correction = Eigen::Affine3f::Identity();
    int latest_backend_candidate_frame = -1;
    double latest_backend_candidate_fitness = std::numeric_limits<double>::infinity();
    bool accumulated_saved = false;
    bool icp_saved = false;
};

bool objectLocalMapEnable = false;
bool objectLocalMapAccumulateForPcd = true;
bool objectLocalMapIcpMap = false;
int objectLocalMapMinValidFrames = 5;
int objectLocalMapLogInterval = 20;
bool objectLocalMapPublish = false;
int objectLocalMapPublishInterval = 5;
bool objectLocalMapIcpCheckEnable = false;
int objectLocalMapIcpCheckMinMapPoints = 300;
int objectLocalMapIcpCheckMinScanPoints = 50;
int objectLocalMapIcpCheckMaxIterations = 30;
double objectLocalMapIcpCheckMaxCorrespondence = 0.8;
double objectLocalMapIcpCheckFitnessThreshold = 0.08;
double objectLocalMapIcpCheckMaxAbsX = 1.0;
double objectLocalMapIcpCheckMaxAbsY = 1.0;
double objectLocalMapIcpCheckMaxAbsZ = 1.0;
double objectLocalMapIcpCheckMaxAbsYaw = 0.2;
bool objectLocalMapBackendFactorEnable = false;
double objectLocalMapBackendFactorNoise = 1.0;
int backendResultMode = 1; // 0: local_only, 1: local_plus_global
bool localGraphFeedbackToGlobal = false;
double localGraphFeedbackNoise = 10.0;
int localGraphFeedbackMinObjectFactors = 1;
double localGraphFeedbackMaxTranslation = 1.0;
double localGraphFeedbackMaxRotation = 0.3;
bool hasLastLocalGraphKeyPoseForGlobal = false;
Eigen::Affine3f lastLocalGraphKeyPoseForGlobal = Eigen::Affine3f::Identity();
std::map<int, ObjectLocalMapState> objectLocalMaps;
std::ofstream objectLocalMapLogStream;


// voxel filter paprams
float odometrySurfLeafSize;
float mappingCornerLeafSize;
float mappingSurfLeafSize;

float z_tollerance;
float rotation_tollerance;

// CPU Params
int numberOfCores = 4;
double mappingProcessInterval;

/*loop clousre*/
bool startFlag = true;
bool loopClosureEnableFlag;
float loopClosureFrequency; //   回环检测频率
int surroundingKeyframeSize;
float historyKeyframeSearchRadius;   // 回环检测 radius kdtree搜索半径
float historyKeyframeSearchTimeDiff; //  帧间时间阈值
int historyKeyframeSearchNum;        //   回环时多少个keyframe拼成submap
float historyKeyframeFitnessScore;   // icp 匹配阈值
float loopClosureNoiseScore;         // 闭环检测噪声值（用于因子图优化）
bool potentialLoopFlag = false;

// limot
int window_size;
double egoP_egoP;
double egoP_objP;
double objP_objP_chgP;
double chgP_chgP;
double ego_stationObjP;
double rubostNum;
bool if_dynamic;
// bool pubtrackedobjects;
bool if_priorfactor;
bool publishOffsetDetect = false;
int publishOffsetDetectSourceId = 1;
int publishOffsetDetectOutputId = 2;
double publishOffsetDetectShiftX = -0.4;   // local x (forward+)
double publishOffsetDetectShiftY = -0.5;   // local y (left+)
double publishOffsetDetectYawDeg = 2.0;    // yaw offset in degree
std::string sequence;
int laserCloudInfoHandler_size;
float vel_threshold;
float Scorethre;
int detection_wait_ms = 60; // 实时检测等待时间，TRT 首帧/预热可能比普通网络更慢
bool associationDescriptorGateEnable = false;
double associationDescriptorCosThreshold = 0.55;
int associationDescriptorGridX = 4;
int associationDescriptorGridY = 4;
int associationDescriptorMinPointsPerCell = 2;
int associationDescriptorMinValidCells = 3;

Eigen::Vector3d last_pos;
Eigen::Vector3d last_rot;//roll pitch yaw

factorgraph::FactorGraph local_graph;
gtsam::NonlinearFactorGraph graph_temp;
vector<LidarSLAMFrame, Eigen::aligned_allocator<LidarSLAMFrame>> frames;
Tracker tracker;
int flow = 0;
std::ofstream objectResultStream;  // 保存每帧目标跟踪结果
map<int, std::vector<int>> priorfactor_id;
map<int, std::vector<int>> priorfactor_egoP_id;
map<int, std::vector<std::pair<int, int>>> priorfactor_objP_id; //<flow,<f_id,obj_k>>

ros::Publisher pubHistoryKeyFrames; //  发布 loop history keyframe submap
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubRecentKeyFrames;
ros::Publisher pubRecentKeyFrame;
ros::Publisher pubCloudRegisteredRaw;
ros::Publisher pubLoopConstraintEdge;
// ros::Publisher pubTrackObjects;  //发布跟踪物体的边界框
ros::Publisher pubTrackedObjects;
ros::Publisher pubExternalTrackedObject;
ros::Publisher pubObjectTrajectories;
ros::Publisher pubTrackedObjectLocalCloud;
ros::Publisher pubObjectLocalMaps;
ros::Publisher pubSaulioFilteredCloud;
ros::Subscriber subDetect;


bool aLoopIsClosed = false;
bool gpsisfound  = false ;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
// vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
vector<gtsam::SharedNoiseModel> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;
std::deque<std_msgs::Float64MultiArray> decQueue;

nav_msgs::Path globalPath;

//track需要用的前后帧位姿数据
Eigen::Affine3f transPointAssociateToMap;
Eigen::Affine3f incrementalOdometryAffineFront;
Eigen::Affine3f incrementalOdometryAffineBack;

// // loop detector 回环检测
SCManager scManager;

// 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());

// 降采样
pcl::VoxelGrid<PointType> downSizeFilterCorner;
// pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterICP;
pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization

// 动态剔除点相关
// filter dynamic points / detected boxes
std::queue<std::vector<double>> dynamBoxBuf; // 来自 limot tracker 的动态框: [x, y, z, ?, l, w, h, yaw]
pcl::CropBox<PointType> box_filter;
// three modes:
// 0 - no filtering (保持原始行为)
// 1 - 在每帧处理开始时, 用"这一帧的所有检测框"直接从输入点云中剔除对应区域, 后面的建图/发布都使用已剔除后的点云
// 2 - 仅剔除动态框: 建图仍用原始点云, 只在发布 world 点云时, 在 world 系对动态框对应区域做剔除
int dynamic_filter_mode = 0;

float transformTobeMapped[6]; //  当前帧的位姿(world系下)

std::mutex mtx;
std::mutex mtxLoopInfo;
std::mutex decLock;

// Surrounding map
float surroundingkeyframeAddingDistThreshold;  //  判断是否为关键帧的距离阈值
float surroundingkeyframeAddingAngleThreshold; //  判断是否为关键帧的角度阈值
float surroundingKeyframeDensity;
float surroundingKeyframeSearchRadius;

// gtsam
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd poseCovariance;

ros::Publisher pubLaserCloudSurround;
ros::Publisher pubOptimizedGlobalMap ;           //   发布最后优化的地图
ros::Publisher pubOptimizedKeyFrames;            //   发布优化后的关键帧点云（5Hz）
ros::Publisher pubGnssPoseGT;                   //   发布GNSS IMU位姿真值（TUM格式）

//reconstruct
bool    reconstructKdTree = false;
bool if_reconstruct_ikdtree_map_add_gps = false;               // true: reconstruct ikdtree map when add gps factor
bool if_reconstruct_ikdtree_map_loop_closure = true;          // true: reconstruct ikdtree map when loop closure
int  reconstruct_ikdtree_mode_add_gps = 1;                     // 0: reconstruct the ikdtree map with riduas frame; 1: reconstruct the ikdtree map with a fixed frame(例如时间上的邻近五个关键帧)
int  reconstruct_ikdtree_mode_loop_closure = 0;
int  reconstruct_ikdtree_frame = 5;                   // number of frames for reconstructing ikdtree map when reconstruct_ikdtree_mode=1
float reconstruct_ikdtree_search_radius = 100.0;  
float reconstruct_pose_density = 10.0;
float reconstruct_map_leafsize = 0.5;                             // meters, downsample point cloud for reconstructing ikdtree map
int add_gps_num_reconstruct = 0;

int updateKdtreeCount = 0 ;        //  重构的请求次数统计
bool visulize_IkdtreeMap = false;            //  visual iktree submap

// gnss
bool use_gnss = false ;
double last_timestamp_gnss = -1.0 ;
deque<nav_msgs::Odometry> gnss_buffer;
geometry_msgs::PoseStamped msg_gnss_pose;
string gnss_topic ;
bool useImuHeadingInitialization;   
bool useGpsElevation;             //  是否使用gps高层优化
float gpsCovThreshold;          //   gps方向角和高度差的协方差阈值
float gpsNoiseMin;              //   GPS 因子最小噪声下限（原来固定 1.0）
float poseCovThreshold;       //  位姿协方差阈值  from isam2

M3D Gnss_R_wrt_Lidar(Eye3d) ;         // gnss  与 imu 的外参
V3D Gnss_T_wrt_Lidar(Zero3d);
bool gnss_inited = false ;                        //  是否完成gnss初始化
shared_ptr<GnssProcess> p_gnss(new GnssProcess());
GnssProcess gnss_data;
ros::Publisher pubGnssPath ;
nav_msgs::Path gps_path ;
vector<double>       extrinT_Gnss2Lidar(3, 0.0);
vector<double>       extrinR_Gnss2Lidar(9, 0.0);
Eigen::Matrix3d R_world_imu; //初始化gnss时，东北天坐标系到imu系的旋转矩阵
// 第一帧GNSS的IMU位姿（用于计算相对位姿）
Eigen::Quaterniond first_gnss_imu_rot = Eigen::Quaterniond::Identity();  // 第一帧GNSS的东北天到IMU旋转（用于计算相对旋转）

// global map visualization radius
float globalMapVisualizationSearchRadius;
float globalMapVisualizationPoseDensity;
float globalMapVisualizationLeafSize;



// saveMap
ros::ServiceServer srvSaveMap;
ros::ServiceServer srvSavePose;
bool savePCD;               // 是否保存地图
string savePCDDirectory;    // 保存路径
int mapSaveInterval = -1;   // 地图保存间隔（帧数），-1表示只在程序结束时保存，1表示每帧保存，N表示每N帧保存一次
int saveMapFrameMode = 0;   // 0: only keyframes, 1: keyframes + intermediate frames
bool saveMapRecordIntermediateFrames = false;
int saveMapIntermediateFrameAnchorMode = 0; // 0: previous keyframe, 1: nearest keyframe
bool saveMapIncludeIntermediateFramesInGlobalMap = false;


/**
 * 更新里程计轨迹
 */
void updatePath(const PointTypePose &pose_in)
{
    string odometryFrame = "camera_init";
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

/**
 * 对点云cloudIn进行变换transformIn，返回结果点云， 修改liosam, 考虑到外参的表示
 */
pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    
   // 注意：lio_sam 中的姿态用的euler表示，而fastlio存的姿态角是旋转矢量。而 pcl::getTransformation是将euler_angle 转换到rotation_matrix 不合适，注释
  // Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_b_lidar(state_point.offset_R_L_I  );       //  获取  body2lidar  外参
    T_b_lidar.pretranslate(state_point.offset_T_L_I);        

    Eigen::Affine3f T_w_b_ = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
    Eigen::Isometry3d T_w_b ;          //   world2body  
    T_w_b.matrix() = T_w_b_.matrix().cast<double>();

    Eigen::Isometry3d  T_w_lidar  =  T_w_b * T_b_lidar  ;           //  T_w_lidar  转换矩阵

    Eigen::Isometry3d transCur = T_w_lidar;        

#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
        cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
        cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

/**
 * 位姿格式变换
 */
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

/**
 * 位姿格式变换
 */
gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

void setTransformTobeMappedFromAffine(const Eigen::Affine3f &pose)
{
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(pose, x, y, z, roll, pitch, yaw);
    transformTobeMapped[0] = roll;
    transformTobeMapped[1] = pitch;
    transformTobeMapped[2] = yaw;
    transformTobeMapped[3] = x;
    transformTobeMapped[4] = y;
    transformTobeMapped[5] = z;
}

void applyPoseToEskfState(const Eigen::Affine3f &pose)
{
    setTransformTobeMappedFromAffine(pose);

    // local_only 模式下，把局部联合图结果作为当前后端结果写回 ESKF 状态
    state_ikfom state_updated = kf.get_x();
    Eigen::Vector3f t = pose.translation();
    Eigen::Quaternionf qf(pose.rotation());
    qf.normalize();
    state_updated.pos = Eigen::Vector3d(t.x(), t.y(), t.z());
    state_updated.rot = Eigen::Quaterniond(qf.w(), qf.x(), qf.y(), qf.z());
    state_point = state_updated;
    kf.change_x(state_updated);
}

void saulioOdomHandler(const nav_msgs::Odometry::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(saulio_odom_mutex);
    saulio_odom_buffer.push_back(*msg);
    while (saulio_odom_buffer.size() > 200)
        saulio_odom_buffer.pop_front();
}

bool waitSaulioOdom(double target_time, nav_msgs::Odometry &odom_out)
{
    double wait_start = ros::Time::now().toSec();
    ros::Rate wait_rate(1000);

    while (ros::ok())
    {
        {
            std::lock_guard<std::mutex> lock(saulio_odom_mutex);
            while (!saulio_odom_buffer.empty() &&
                   saulio_odom_buffer.front().header.stamp.toSec() < target_time - saulio_odom_time_tolerance)
            {
                saulio_odom_buffer.pop_front();
            }

            int best_index = -1;
            double best_dt = std::numeric_limits<double>::max();
            for (int i = 0; i < static_cast<int>(saulio_odom_buffer.size()); ++i)
            {
                double dt = std::fabs(saulio_odom_buffer[i].header.stamp.toSec() - target_time);
                if (dt < best_dt)
                {
                    best_dt = dt;
                    best_index = i;
                }
                if (saulio_odom_buffer[i].header.stamp.toSec() > target_time + saulio_odom_time_tolerance)
                    break;
            }

            if (best_index >= 0 && best_dt <= saulio_odom_time_tolerance)
            {
                odom_out = saulio_odom_buffer[best_index];
                saulio_odom_buffer.erase(saulio_odom_buffer.begin(), saulio_odom_buffer.begin() + best_index + 1);
                return true;
            }
        }

        if (ros::Time::now().toSec() - wait_start > saulio_odom_wait_timeout)
            return false;

        ros::spinOnce();
        wait_rate.sleep();
    }

    return false;
}

void applySaulioOdomToEskfState(const nav_msgs::Odometry &odom_msg)
{
    const geometry_msgs::Pose &pose_msg = odom_msg.pose.pose;
    Eigen::Quaterniond q(pose_msg.orientation.w,
                         pose_msg.orientation.x,
                         pose_msg.orientation.y,
                         pose_msg.orientation.z);
    if (q.norm() < 1e-6)
    {
        ROS_WARN_THROTTLE(1.0, "[saulio_frontend] invalid odometry quaternion, skip this frame");
        return;
    }
    q.normalize();

    state_ikfom state_updated = kf.get_x();
    state_updated.pos = Eigen::Vector3d(pose_msg.position.x,
                                        pose_msg.position.y,
                                        pose_msg.position.z);
    state_updated.rot = q;
    state_point = state_updated;
    kf.change_x(state_updated);

    Eigen::Affine3f pose = Eigen::Affine3f::Identity();
    pose.linear() = q.toRotationMatrix().cast<float>();
    pose.translation() = state_updated.pos.cast<float>();
    setTransformTobeMappedFromAffine(pose);

    // 外部 SAULIO 前端作为当前帧里程计来源时，后端和可视化仍沿用原来的状态变量。
    geoQuat.x = q.x();
    geoQuat.y = q.y();
    geoQuat.z = q.z();
    geoQuat.w = q.w();
}

void publishFilteredCloudForSaulio()
{
    if (frontend_odom_mode != 1 || !saulio_publish_filtered_cloud)
        return;

    if (pubSaulioFilteredCloud.getNumSubscribers() == 0)
        return;

    if (!feats_undistort || feats_undistort->empty())
        return;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*feats_undistort, cloud_msg);
    cloud_msg.header.stamp = ros::Time().fromSec(lidar_start_time);
    cloud_msg.header.frame_id = "body";
    pubSaulioFilteredCloud.publish(cloud_msg);
}

void appendLocalOnlyKeyFrame()
{
    PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
    thisPose6D.time = lidar_end_time;

    PointType thisPose3D;
    thisPose3D.x = thisPose6D.x;
    thisPose3D.y = thisPose6D.y;
    thisPose3D.z = thisPose6D.z;
    thisPose3D.intensity = cloudKeyPoses3D->size();
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.intensity = thisPose3D.intensity;
    cloudKeyPoses6D->push_back(thisPose6D);
    keyframeAnchorCapturePoses.push_back(thisPose6D);
    intermediateFramesPerKeyframe.emplace_back();
    if (saveMapFrameMode == 1 && saveMapRecordIntermediateFrames && saveMapIntermediateFrameAnchorMode == 1)
    {
        flushPendingIntermediateFramesToAnchor(static_cast<int>(cloudKeyPoses6D->size()) - 1);
    }

    poseCovariance = Eigen::MatrixXd::Identity(6, 6);
}

void addLocalGraphFeedbackToGlobalIfNeeded(int global_key,
                                           const Eigen::Affine3f &fast_lio_pose,
                                           const Eigen::Affine3f &local_optimized_pose,
                                           int object_map_factor_count)
{
    if (!localGraphFeedbackToGlobal)
        return;

    if (global_key <= 0 || !hasLastLocalGraphKeyPoseForGlobal)
        return;

    if (object_map_factor_count < localGraphFeedbackMinObjectFactors)
        return;

    Eigen::Affine3f delta = fast_lio_pose.inverse() * local_optimized_pose;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles(delta, dx, dy, dz, droll, dpitch, dyaw);
    double trans_norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    double rot_norm = std::sqrt(droll * droll + dpitch * dpitch + dyaw * dyaw);
    if (trans_norm > localGraphFeedbackMaxTranslation || rot_norm > localGraphFeedbackMaxRotation)
    {
        ROS_WARN_STREAM("[localGraphFeedback] skip global feedback: trans=" << trans_norm
                        << ", rot=" << rot_norm
                        << ", object_factors=" << object_map_factor_count);
        return;
    }

    // local_plus_global 模式下，只把 local_graph 的相邻关键帧增量作为弱 Between 因子反馈给全局自车图。
    Eigen::Affine3f local_between =
        lastLocalGraphKeyPoseForGlobal.inverse() * local_optimized_pose;
    gtsam::Vector feedbackNoiseVector6(6);
    feedbackNoiseVector6 << localGraphFeedbackNoise, localGraphFeedbackNoise, localGraphFeedbackNoise,
                            localGraphFeedbackNoise, localGraphFeedbackNoise, localGraphFeedbackNoise;
    gtsam::noiseModel::Base::shared_ptr feedbackNoise =
        gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(rubostNum),
            gtsam::noiseModel::Diagonal::Variances(feedbackNoiseVector6));
    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
        global_key - 1, global_key, Affine3f2Pose3(local_between), feedbackNoise));
}

void updateLocalGraphKeyPoseForGlobal(const Eigen::Affine3f &local_optimized_pose)
{
    // 只记录已经成为全局关键帧的 local_graph 位姿，下一关键帧用它形成局部增量约束。
    lastLocalGraphKeyPoseForGlobal = local_optimized_pose;
    hasLastLocalGraphKeyPoseForGlobal = true;
}

/**
 * 位姿格式变换
 */
PointTypePose trans2PointTypePose(float transformIn[])
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn[3];
    thisPose6D.y = transformIn[4];
    thisPose6D.z = transformIn[5];
    thisPose6D.roll = transformIn[0];
    thisPose6D.pitch = transformIn[1];
    thisPose6D.yaw = transformIn[2];
    return thisPose6D;
}

PointTypePose affine3fToPointTypePose(const Eigen::Affine3f &transformIn, double time = 0.0)
{
    PointTypePose thisPose6D;
    thisPose6D.x = transformIn.translation().x();
    thisPose6D.y = transformIn.translation().y();
    thisPose6D.z = transformIn.translation().z();
    Eigen::Vector3f rpy = transformIn.rotation().eulerAngles(0, 1, 2);
    thisPose6D.roll = rpy[0];
    thisPose6D.pitch = rpy[1];
    thisPose6D.yaw = rpy[2];
    thisPose6D.time = time;
    thisPose6D.intensity = 0.0f;
    return thisPose6D;
}

pcl::PointCloud<PointType>::Ptr captureFrameCloudForMap()
{
    pcl::PointCloud<PointType>::Ptr frameCloud(new pcl::PointCloud<PointType>());
    if (dense_keyframe)
        pcl::copyPointCloud(*feats_undistort, *frameCloud);
    else
        pcl::copyPointCloud(*feats_down_body, *frameCloud);
    return frameCloud;
}

void appendIntermediateFrameToAnchor(
    int anchor_keyframe_index,
    const PointTypePose &anchor_pose_body,
    const PointTypePose &frame_pose_body,
    pcl::PointCloud<PointType>::Ptr frame_cloud,
    int frame_id,
    double time)
{
    if (anchor_keyframe_index < 0 || frame_cloud == nullptr || frame_cloud->empty())
        return;

    if (anchor_keyframe_index >= static_cast<int>(intermediateFramesPerKeyframe.size()))
        intermediateFramesPerKeyframe.resize(anchor_keyframe_index + 1);

    Eigen::Affine3f T_w_b_anchor = pclPointToAffine3f(anchor_pose_body);
    Eigen::Affine3f T_w_b_frame = pclPointToAffine3f(frame_pose_body);
    Eigen::Affine3f T_anchor_to_frame = T_w_b_anchor.inverse() * T_w_b_frame;

    IntermediateFrameRecord record;
    record.frame_id = frame_id;
    record.time = time;
    record.relative_pose_to_anchor = affine3fToPointTypePose(T_anchor_to_frame, time);
    record.cloud = frame_cloud;
    intermediateFramesPerKeyframe[anchor_keyframe_index].push_back(record);
}

void recordIntermediateFrameForMap(const PointTypePose &currentPose6D)
{
    bool feature_enabled =
        saveMapFrameMode == 1 &&
        saveMapRecordIntermediateFrames;

    if (!feature_enabled)
        return;

    if (cloudKeyPoses6D->empty() || keyframeAnchorCapturePoses.empty())
        return;

    pcl::PointCloud<PointType>::Ptr frameCloud = captureFrameCloudForMap();
    if (!frameCloud || frameCloud->empty())
        return;

    if (saveMapIntermediateFrameAnchorMode == 0)
    {
        int anchor_idx = static_cast<int>(cloudKeyPoses6D->size()) - 1;
        appendIntermediateFrameToAnchor(
            anchor_idx,
            keyframeAnchorCapturePoses.back(),
            currentPose6D,
            frameCloud,
            flow,
            lidar_end_time);
    }
    else
    {
        PendingIntermediateFrameRecord pending;
        pending.frame_id = flow;
        pending.time = lidar_end_time;
        pending.absolute_pose_body = currentPose6D;
        pending.cloud = frameCloud;
        pendingIntermediateFrames.push_back(pending);
    }
}

void flushPendingIntermediateFramesToAnchor(int new_keyframe_index)
{
    if (pendingIntermediateFrames.empty())
        return;

    if (new_keyframe_index < 0 || new_keyframe_index >= static_cast<int>(keyframeAnchorCapturePoses.size()))
        return;

    int prev_keyframe_index = new_keyframe_index - 1;
    const PointTypePose &new_anchor_pose = keyframeAnchorCapturePoses[new_keyframe_index];

    if (prev_keyframe_index < 0)
    {
        for (const auto &pending : pendingIntermediateFrames)
        {
            appendIntermediateFrameToAnchor(
                new_keyframe_index,
                new_anchor_pose,
                pending.absolute_pose_body,
                pending.cloud,
                pending.frame_id,
                pending.time);
        }
        pendingIntermediateFrames.clear();
        return;
    }

    const PointTypePose &prev_anchor_pose = keyframeAnchorCapturePoses[prev_keyframe_index];
    for (const auto &pending : pendingIntermediateFrames)
    {
        double dist_prev = std::hypot(
            pending.absolute_pose_body.x - prev_anchor_pose.x,
            pending.absolute_pose_body.y - prev_anchor_pose.y);
        double dist_new = std::hypot(
            pending.absolute_pose_body.x - new_anchor_pose.x,
            pending.absolute_pose_body.y - new_anchor_pose.y);

        int anchor_idx = dist_prev <= dist_new ? prev_keyframe_index : new_keyframe_index;
        const PointTypePose &anchor_pose = dist_prev <= dist_new ? prev_anchor_pose : new_anchor_pose;
        appendIntermediateFrameToAnchor(
            anchor_idx,
            anchor_pose,
            pending.absolute_pose_body,
            pending.cloud,
            pending.frame_id,
            pending.time);
    }
    pendingIntermediateFrames.clear();
}

/**
 * 发布thisCloud，返回thisCloud对应msg格式
 */
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

/**
 * 点到坐标系原点距离
 */
float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

/**
 * 两点之间距离
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

/**
 * 初始化
 */
void allocateMemory()
{
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

    laserCloudOri.reset(new pcl::PointCloud<PointType>());

    kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }
}

//  eulerAngle 2 Quaterniond
Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_)
{
    Eigen::Quaterniond q ;            //   四元数 q 和 -q 是相等的
    Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
    q = yaw * pitch * roll ;
    q.normalize();
    return q ;
}

// 将更新的pose赋值到 transformTobeMapped
void getCurPose(state_ikfom cur_state)
{
    //  欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
    // V3D eulerAngle  =  SO3ToEuler(cur_state.rot)/57.3 ;     //   fastlio 自带  roll pitch yaw  单位: 度，旋转顺序 zyx

    // transformTobeMapped[0] = eulerAngle(0);                //  roll     使用 SO3ToEuler 方法时，顺序是 rpy
    // transformTobeMapped[1] = eulerAngle(1);                //  pitch
    // transformTobeMapped[2] = eulerAngle(2);                //  yaw
    
    transformTobeMapped[0] = eulerAngle(2);                //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = eulerAngle(1);                //  pitch
    transformTobeMapped[2] = eulerAngle(0);                //  yaw
    transformTobeMapped[3] = cur_state.pos(0);          //  x
    transformTobeMapped[4] = cur_state.pos(1);          //   y
    transformTobeMapped[5] = cur_state.pos(2);          // z
}

/**
 * rviz展示闭环边
 */

//     gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint) {
//     return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
//         gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
// }

// gtsam::Pose3 trans2gtsamPose(float transformIn[]) {
//     return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
//         gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
// }

// Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint) {
//     return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
// }

// Eigen::Affine3f trans2Affine3f(float transformIn[]) {
//     return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
// }
Eigen::Affine3f gtsamPose3toAffine3f(gtsam::Pose3 transformIn) {
    return pcl::getTransformation(transformIn.x(), transformIn.y(), transformIn.z(), transformIn.rotation().rpy()[0], transformIn.rotation().rpy()[1], transformIn.rotation().rpy()[2]);
}
// Affine3f -> Pose3
gtsam::Pose3 Affine3f2Pose3(Eigen::Affine3f T) {
    float roll, pitch, yaw, x, y, z;
    roll = T.rotation().eulerAngles(0, 1, 2)[0];
    pitch = T.rotation().eulerAngles(0, 1, 2)[1];
    yaw = T.rotation().eulerAngles(0, 1, 2)[2];
    x = T.translation()[0];
    y = T.translation()[1];
    z = T.translation()[2];
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw),
        gtsam::Point3(x, y, z));
}

gtsam::Pose3 vec6ftoPose3(vector<float>& v) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(v[0], v[1], v[2]),
        gtsam::Point3(v[3], v[4], v[5]));
}

bool Pose3tovec6f(gtsam::Pose3 transformIn, vector<float>& transformOut) {
    transformOut.clear();
    transformOut.resize(6);
    Eigen::Affine3f T = gtsamPose3toAffine3f(transformIn);
    transformOut[0] = T.rotation().eulerAngles(0, 1, 2)[0];
    transformOut[1] = T.rotation().eulerAngles(0, 1, 2)[1];
    transformOut[2] = T.rotation().eulerAngles(0, 1, 2)[2];
    transformOut[3] = T.translation()[0];
    transformOut[4] = T.translation()[1];
    transformOut[5] = T.translation()[2];
    return true;
}
bool Pose3tovec3f(gtsam::Pose3 transformIn, vector<float>& transformOut) {
    transformOut.clear();
    transformOut.resize(3);
    Eigen::Affine3f T = gtsamPose3toAffine3f(transformIn);
    transformOut[0] = T.translation()[0];
    transformOut[1] = T.translation()[1];
    transformOut[2] = T.translation()[2];
    return true;
}

bool Affine3ftovec6f(Eigen::Affine3f T, vector<float>& transformOut) {
    transformOut.clear();
    transformOut.resize(6);
    transformOut[0] = T.rotation().eulerAngles(0, 1, 2)[0];
    transformOut[1] = T.rotation().eulerAngles(0, 1, 2)[1];
    transformOut[2] = T.rotation().eulerAngles(0, 1, 2)[2];
    transformOut[3] = T.translation()[0];
    transformOut[4] = T.translation()[1];
    transformOut[5] = T.translation()[2];
    return true;
}

float* gtsamPose3totrans(gtsam::Pose3 transformIn) {
    float* transformOut = new float[6];
    Eigen::Affine3f T = gtsamPose3toAffine3f(transformIn);
    transformOut[0] = T.rotation().eulerAngles(0, 1, 2)[0];
    transformOut[1] = T.rotation().eulerAngles(0, 1, 2)[1];
    transformOut[2] = T.rotation().eulerAngles(0, 1, 2)[2];
    transformOut[3] = T.translation()[0];
    transformOut[4] = T.translation()[1];
    transformOut[5] = T.translation()[2];
    return transformOut;
}


// PointTypePose trans2PointTypePose(float transformIn[]) {
//     PointTypePose thisPose6D;
//     thisPose6D.x = transformIn[3];
//     thisPose6D.y = transformIn[4];
//     thisPose6D.z = transformIn[5];
//     thisPose6D.roll = transformIn[0];
//     thisPose6D.pitch = transformIn[1];
//     thisPose6D.yaw = transformIn[2];
//     return thisPose6D;
// }

void visualizeLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";

    if (loopIndexContainer.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

//可视化跟踪
// 根据object_id生成一致的颜色
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

    // 可视化物体轨迹
void visualizeObjectTrajectories() {
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    string odometryFrame = "camera_init";
    visualization_msgs::MarkerArray trajectoryMarkers;
    int marker_id = 0;
    
    for (auto& pair : object_trajectories) {
        int object_id = pair.first;
        ObjectTrajectory& traj = pair.second;
        
        if (traj.positions.empty())
            continue;
        
        std::vector<float> color = generateColorFromID(object_id);
        
        // 创建轨迹线
        visualization_msgs::Marker lineMarker;
        lineMarker.header.frame_id = odometryFrame;
        lineMarker.header.stamp = timeLaserInfoStamp;
        lineMarker.ns = "object_trajectories";
        lineMarker.id = marker_id++;
        lineMarker.type = visualization_msgs::Marker::LINE_STRIP;
        lineMarker.action = visualization_msgs::Marker::ADD;
        lineMarker.pose.orientation.w = 1.0;
        lineMarker.scale.x = 0.05;  // 线宽
        lineMarker.color.r = color[0];
        lineMarker.color.g = color[1];
        lineMarker.color.b = color[2];
        lineMarker.color.a = traj.is_active ? 0.8 : 0.4;  // 失效的轨迹更透明
        
        for (const Eigen::Vector3f& pos : traj.positions) {
            geometry_msgs::Point p;
            p.x = pos.x();
            p.y = pos.y();
            p.z = pos.z();
            lineMarker.points.push_back(p);
        }
        
        trajectoryMarkers.markers.push_back(lineMarker);
        
        // 为每个位置添加小球体
        for (size_t i = 0; i < traj.positions.size(); i++) {
            visualization_msgs::Marker sphereMarker;
            sphereMarker.header.frame_id = odometryFrame;
            sphereMarker.header.stamp = timeLaserInfoStamp;
            sphereMarker.ns = "trajectory_points";
            sphereMarker.id = marker_id++;
            sphereMarker.type = visualization_msgs::Marker::SPHERE;
            sphereMarker.action = visualization_msgs::Marker::ADD;
            
            sphereMarker.pose.position.x = traj.positions[i].x();
            sphereMarker.pose.position.y = traj.positions[i].y();
            sphereMarker.pose.position.z = traj.positions[i].z();
            sphereMarker.pose.orientation.w = 1.0;
            
            // 最后一个球体稍大
            if (i == traj.positions.size() - 1) {
                sphereMarker.scale.x = 0.3;
                sphereMarker.scale.y = 0.3;
                sphereMarker.scale.z = 0.3;
            } else {
                sphereMarker.scale.x = 0.15;
                sphereMarker.scale.y = 0.15;
                sphereMarker.scale.z = 0.15;
            }
            
            sphereMarker.color.r = color[0];
            sphereMarker.color.g = color[1];
            sphereMarker.color.b = color[2];
            sphereMarker.color.a = traj.is_active ? 0.9 : 0.6;  // 失效的球体更透明
            
            trajectoryMarkers.markers.push_back(sphereMarker);
        }
    }
    
    pubObjectTrajectories.publish(trajectoryMarkers);
}

void visualizeTrackedObjects() {
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    string odometryFrame = "camera_init";
    if (frames.empty()) {
        ROS_WARN_THROTTLE(5, "visualizeTrackedObjects: frames is empty");
        return;
    }
    
    if (flow <= 0 || flow > (int)frames.size()) {
        cout << "visualizeTrackedObjects: invalid flow=" << flow << ", frames.size()=" << frames.size() << endl;
        return;
    }
    
    // flow 已经递增了，所以当前要可视化的是 flow-1 帧
    int current_frame_idx = flow - 1;
    cout << "visualizeTrackedObjects called: flow=" << flow << ", current_frame_idx=" << current_frame_idx << ", frames.size()=" << frames.size() << endl;
    
    visualization_msgs::MarkerArray markerArray;
    visualization_msgs::MarkerArray externalMarkerArray;
    
    // 【重要】先清除所有旧的边界框和标签，避免累积显示
    visualization_msgs::Marker deleteMarker;
    deleteMarker.header.frame_id = odometryFrame;
    deleteMarker.header.stamp = timeLaserInfoStamp;
    deleteMarker.action = visualization_msgs::Marker::DELETEALL;
    
    // 清除边界框
    deleteMarker.ns = "tracked_boxes";
    markerArray.markers.push_back(deleteMarker);
    
    // 清除标签
    deleteMarker.ns = "tracked_labels";
    markerArray.markers.push_back(deleteMarker);

    deleteMarker.ns = "external_detect_boxes";
    externalMarkerArray.markers.push_back(deleteMarker);
    deleteMarker.ns = "external_detect_labels";
    externalMarkerArray.markers.push_back(deleteMarker);
    
    // 标记当前帧所有物体为非活跃
    for (auto& pair : object_trajectories) {
        pair.second.is_active = false;
    }
    
    int marker_id = 0;
    const LidarSLAMFrame& current_frame = frames[current_frame_idx];
    
    // 统计信息
    int total_objects = current_frame.objects.size();
    int initialized_count = 0;
    int associated_count = 0;
    
    for (const LidarSLAMObject& obj : current_frame.objects) {
        if (obj.initialized) initialized_count++;
        if (obj.associated) associated_count++;
        
        // 显示所有已关联的物体（不仅仅是已初始化的）
        // 这样可以更早地看到跟踪效果
        if (!obj.associated || obj.object_id < 0) {
            continue;
        }
        
        int object_id = obj.object_id;
        std::vector<float> color = generateColorFromID(object_id);
        
        // 使用优化后的位姿 (optimize_t: [roll, pitch, yaw, x, y, z])
        float roll = obj.optimize_t[0];
        float pitch = obj.optimize_t[1];
        float yaw = obj.optimize_t[2];
        float x = obj.optimize_t[3];
        float y = obj.optimize_t[4];
        float z = obj.optimize_t[5];
        
        float l = obj.measure_lwh[0];  // 长度
        float w = obj.measure_lwh[1];  // 宽度
        float h = obj.measure_lwh[2];  // 高度
        
        // 更新轨迹
        if (object_trajectories.find(object_id) == object_trajectories.end()) {
            ObjectTrajectory traj;
            traj.object_id = object_id;
            traj.is_active = true;
            traj.last_seen = timeLaserInfoStamp;
            object_trajectories[object_id] = traj;
        }
        object_trajectories[object_id].positions.push_back(Eigen::Vector3f(x, y, z));
        object_trajectories[object_id].is_active = true;
        object_trajectories[object_id].last_seen = timeLaserInfoStamp;
        
        // 创建3D边界框，直接用 CUBE 显示实体长方体
        visualization_msgs::Marker marker;
        marker.header.frame_id = odometryFrame;
        marker.header.stamp = timeLaserInfoStamp;
        marker.ns = "tracked_boxes";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.color.r = color[0];
        marker.color.g = color[1];
        marker.color.b = color[2];
        marker.color.a = 0.55;
        
        // 定义边界框的8个顶点（在物体坐标系中）
        std::vector<Eigen::Vector3f> corners(8);
        // 输出L、W、H和roll、pitch、yaw的值（用于调试）
        // cout << "Object ID: " << object_id 
        //      << ", L=" << l << "m, W=" << w << "m, H=" << h << "m"
        //      << ", Roll=" << roll << "rad (" << roll*180.0/M_PI << "deg)"
        //      << ", Pitch=" << pitch << "rad (" << pitch*180.0/M_PI << "deg)"
        //      << ", Yaw=" << yaw << "rad (" << yaw*180.0/M_PI << "deg)" << endl;
        corners[0] = Eigen::Vector3f(-l/2, -w/2, -h/2);
        corners[1] = Eigen::Vector3f( l/2, -w/2, -h/2);
        corners[2] = Eigen::Vector3f( l/2,  w/2, -h/2);
        corners[3] = Eigen::Vector3f(-l/2,  w/2, -h/2);
        corners[4] = Eigen::Vector3f(-l/2, -w/2,  h/2);
        corners[5] = Eigen::Vector3f( l/2, -w/2,  h/2);
        corners[6] = Eigen::Vector3f( l/2,  w/2,  h/2);
        corners[7] = Eigen::Vector3f(-l/2,  w/2,  h/2);
        
        // 创建旋转矩阵（ZYX欧拉角顺序：先绕Z轴旋转yaw，再绕Y轴旋转pitch，最后绕X轴旋转roll）
        // 对于车辆物体，通常只需要yaw角度（水平面旋转），roll和pitch为0
        Eigen::Matrix3f R;
        // 使用ZYX顺序，但注意Eigen的AngleAxis是右乘，所以顺序是：R_z * R_y * R_x
        // 这意味着先应用X旋转，再应用Y旋转，最后应用Z旋转
        // 但我们需要的是先Z（yaw），再Y（pitch），最后X（roll），所以顺序应该是正确的
        // 但为了确保正确，我们直接使用欧拉角构造函数
        R = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());

        Eigen::Quaternionf q_box(R);
        q_box.normalize();
        marker.pose.orientation.x = q_box.x();
        marker.pose.orientation.y = q_box.y();
        marker.pose.orientation.z = q_box.z();
        marker.pose.orientation.w = q_box.w();
        marker.scale.x = l;
        marker.scale.y = w;
        marker.scale.z = h;
        
        // 转换顶点到世界坐标系
        std::vector<geometry_msgs::Point> points(8);
        for (int j = 0; j < 8; j++) {
            Eigen::Vector3f corner_world = R * corners[j] + Eigen::Vector3f(x, y, z);
            points[j].x = corner_world.x();
            points[j].y = corner_world.y();
            points[j].z = corner_world.z();
        }
        
        markerArray.markers.push_back(marker);

        // 叠加线框边缘，确保四条竖边在任意视角都清楚可见
        visualization_msgs::Marker edgeMarker;
        edgeMarker.header.frame_id = odometryFrame;
        edgeMarker.header.stamp = timeLaserInfoStamp;
        edgeMarker.ns = "tracked_box_edges";
        edgeMarker.id = marker_id++;
        edgeMarker.type = visualization_msgs::Marker::LINE_LIST;
        edgeMarker.action = visualization_msgs::Marker::ADD;
        edgeMarker.pose.orientation.w = 1.0;
        edgeMarker.scale.x = 0.06;
        edgeMarker.color.r = color[0];
        edgeMarker.color.g = color[1];
        edgeMarker.color.b = color[2];
        edgeMarker.color.a = 1.0;
        edgeMarker.points.push_back(points[0]); edgeMarker.points.push_back(points[1]);
        edgeMarker.points.push_back(points[1]); edgeMarker.points.push_back(points[2]);
        edgeMarker.points.push_back(points[2]); edgeMarker.points.push_back(points[3]);
        edgeMarker.points.push_back(points[3]); edgeMarker.points.push_back(points[0]);
        edgeMarker.points.push_back(points[4]); edgeMarker.points.push_back(points[5]);
        edgeMarker.points.push_back(points[5]); edgeMarker.points.push_back(points[6]);
        edgeMarker.points.push_back(points[6]); edgeMarker.points.push_back(points[7]);
        edgeMarker.points.push_back(points[7]); edgeMarker.points.push_back(points[4]);
        edgeMarker.points.push_back(points[0]); edgeMarker.points.push_back(points[4]);
        edgeMarker.points.push_back(points[1]); edgeMarker.points.push_back(points[5]);
        edgeMarker.points.push_back(points[2]); edgeMarker.points.push_back(points[6]);
        edgeMarker.points.push_back(points[3]); edgeMarker.points.push_back(points[7]);
        markerArray.markers.push_back(edgeMarker);
        
        // 添加物体ID文本标签
        visualization_msgs::Marker textMarker;
        textMarker.header.frame_id = odometryFrame;
        textMarker.header.stamp = timeLaserInfoStamp;
        textMarker.ns = "tracked_labels";
        textMarker.id = marker_id++;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.action = visualization_msgs::Marker::ADD;
        textMarker.pose.position.x = x;
        textMarker.pose.position.y = y;
        textMarker.pose.position.z = z + h/2 + 0.8;
        textMarker.pose.orientation.w = 1.0;
        textMarker.scale.z = 0.6;
        textMarker.color.r = 1.0;
        textMarker.color.g = 1.0;
        textMarker.color.b = 1.0;
        textMarker.color.a = 1.0;
        
        std::stringstream ss;
        ss << "ID:" << object_id;
        if (obj.dynamic) {
            ss << " V:" << std::fixed << std::setprecision(1) << obj.velocity << "km/h";
        }
        textMarker.text = ss.str();
        
        markerArray.markers.push_back(textMarker);

        if (publishOffsetDetect && object_id == publishOffsetDetectSourceId) {
            Eigen::Vector3f offset_local(
                static_cast<float>(publishOffsetDetectShiftX),
                static_cast<float>(publishOffsetDetectShiftY),
                0.0f);
            Eigen::Vector3f center_world(x, y, z);
            Eigen::Vector3f center_offset = center_world + R * offset_local;
            const float yaw_offset_rad = static_cast<float>(publishOffsetDetectYawDeg) * static_cast<float>(M_PI) / 180.0f;
            Eigen::Matrix3f R_offset = R * Eigen::AngleAxisf(yaw_offset_rad, Eigen::Vector3f::UnitZ()).toRotationMatrix();

            visualization_msgs::Marker detectMarker;
            detectMarker.header.frame_id = odometryFrame;
            detectMarker.header.stamp = timeLaserInfoStamp;
            detectMarker.ns = "external_detect_boxes";
            detectMarker.id = publishOffsetDetectOutputId;
            detectMarker.type = visualization_msgs::Marker::CUBE;
            detectMarker.action = visualization_msgs::Marker::ADD;
            detectMarker.pose.position.x = center_offset.x();
            detectMarker.pose.position.y = center_offset.y();
            detectMarker.pose.position.z = center_offset.z();
            Eigen::Quaternionf q_offset(R_offset);
            q_offset.normalize();
            detectMarker.pose.orientation.x = q_offset.x();
            detectMarker.pose.orientation.y = q_offset.y();
            detectMarker.pose.orientation.z = q_offset.z();
            detectMarker.pose.orientation.w = q_offset.w();
            detectMarker.scale.x = l;
            detectMarker.scale.y = w;
            detectMarker.scale.z = h;
            detectMarker.color.r = 1.0;
            detectMarker.color.g = 0.2;
            detectMarker.color.b = 0.2;
            detectMarker.color.a = 0.35;
            externalMarkerArray.markers.push_back(detectMarker);

            visualization_msgs::Marker detectText;
            detectText.header.frame_id = odometryFrame;
            detectText.header.stamp = timeLaserInfoStamp;
            detectText.ns = "external_detect_labels";
            detectText.id = publishOffsetDetectOutputId;
            detectText.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            detectText.action = visualization_msgs::Marker::ADD;
            detectText.pose.position.x = center_offset.x();
            detectText.pose.position.y = center_offset.y();
            detectText.pose.position.z = center_offset.z() + h / 2 + 0.8;
            detectText.pose.orientation.w = 1.0;
            detectText.scale.z = 0.6;
            detectText.color.r = 1.0;
            detectText.color.g = 1.0;
            detectText.color.b = 1.0;
            detectText.color.a = 1.0;
            detectText.text = std::string("ID:") + std::to_string(publishOffsetDetectOutputId);
            externalMarkerArray.markers.push_back(detectText);
        }
    }
    
    // 打印统计信息
    int visualized_count = (markerArray.markers.size() - 2) / 2; // 减去2个DELETEALL，除以2（框+标签）
    cout << "Tracked Objects - Total: " << total_objects 
            << ", Associated: " << associated_count 
            << ", Initialized: " << initialized_count 
            << ", Visualized: " << visualized_count << endl;
    
    // 发布边界框（即使为空也发布，以清除旧的marker）
    pubTrackedObjects.publish(markerArray);
    pubExternalTrackedObject.publish(externalMarkerArray);
    cout << "Published " << markerArray.markers.size() << " markers to /limot/tracked_objects" << endl;
    
    // 可视化所有物体的轨迹
    visualizeObjectTrajectories();
}
    
/**
 * 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
 */
bool saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    // 前一帧位姿
    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    // 当前帧位姿
    Eigen::Affine3f transFinal = trans2Affine3f(transformTobeMapped);
    // Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5],
    //                                                     transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                    
    // 位姿变换增量
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿

    // 旋转和平移量都较小，当前帧不设为关键帧
    if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
        abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        return false;
    return true;
}

/**
 * 添加激光里程计因子
 */
void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        // 第一帧初始化先验因子
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) <<1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter   // indoor 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12    //  1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        // 变量节点设置初始值
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }
    else
    {
        // 添加激光里程计因子
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back()); /// pre
        gtsam::Pose3 poseTo = trans2gtsamPose(transformTobeMapped);                   // cur
        // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // 变量节点设置初始值
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

/**
 * 添加闭环因子
 */
void addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    // 闭环队列
    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        // 闭环边对应两帧的索引
        int indexFrom = loopIndexQueue[i].first; //   cur
        int indexTo = loopIndexQueue[i].second;  //    pre
        // 闭环边的位姿变换
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        auto noiseBetween = loopNoiseQueue[i];
        // gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

/**
 * 添加GPS因子
*/
void addGPSFactor()
{
    if (gnss_buffer.empty())
    {
        ROS_INFO("[GPS] addGPSFactor: gnss_buffer is empty, return");
        return;
    }
    // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
    if (cloudKeyPoses3D->points.empty())
    {
        ROS_INFO("[GPS] addGPSFactor: cloudKeyPoses3D is empty, return");
        return;
    }
    else
    {
        ROS_INFO("[GPS] addGPSFactor: current keyframe count = %lu", cloudKeyPoses3D->points.size());
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 0)
        {
            ROS_INFO("[GPS] addGPSFactor: first and last keyframe distance < 5m, return");
            return;
        }
    }
    // 位姿协方差很小，没必要加入GPS数据进行校正
    // if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
        
    // {
    //     cout<<"11111111"<<endl;
    //     ROS_INFO("[GPS] addGPSFactor: pose covariance below threshold, return");
    //     return; }
    static PointType lastGPSPoint;      // 最新的gps数据
    while (!gnss_buffer.empty())
    {
        // 删除当前帧0.2s之前的里程计
        if (gnss_buffer.front().header.stamp.toSec() < lidar_end_time - 0.05)
        {
            gnss_buffer.pop_front();
        }
        // 超过当前帧0.2s之后，退出
        else if (gnss_buffer.front().header.stamp.toSec() > lidar_end_time + 0.05)
        {
            ROS_INFO("[GPS] addGPSFactor: gnss time > lidar_end_time + 0.05, break");
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();
            // GPS噪声协方差太大，不能用
            float noise_x = thisGPS.pose.covariance[0];         //  x 方向的协方差
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];      //   z(高层)方向的协方差
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)

                {
                    // cout<<"22222222"<<endl;
                    ROS_INFO("[GPS] addGPSFactor: GNSS covariance too large, continue");
                    continue;}
            // GPS里程计位置
            // 获取当前位姿值
            float current_pose_x = transformTobeMapped[3];  // 当前位姿 x
            float current_pose_y = transformTobeMapped[4];  // 当前位姿 y
            float current_pose_z = transformTobeMapped[5];  // 当前位姿 z
            float current_pose_roll = transformTobeMapped[0];  // 当前位姿 roll
            float current_pose_pitch = transformTobeMapped[1];  // 当前位姿 pitch
            float current_pose_yaw = transformTobeMapped[2];  // 当前位姿 yaw
            
            // 从欧拉角计算当前旋转矩阵（世界系到当前IMU系的旋转）
            // 注意：世界系(camera_init)在第一帧时与第一帧IMU系对齐
            // 所以这个旋转矩阵表示：当前IMU系相对于第一帧IMU系的旋转
            Eigen::Quaterniond current_rot_quat = EulerToQuat(current_pose_roll, current_pose_pitch, current_pose_yaw);
            Eigen::Matrix3d current_rot_matrix = current_rot_quat.toRotationMatrix();
            
            // GPS坐标写成三维向量
            Eigen::Vector3d gps_pos(thisGPS.pose.pose.position.x, 
                                    thisGPS.pose.pose.position.y, 
                                    thisGPS.pose.pose.position.z);
            
            // 计算新的GPS坐标：原GPS坐标 + 当前旋转矩阵 * IMU到GNSS的平移向量
            Eigen::Vector3d gps_pos_corrected = gps_pos + current_rot_matrix * t_imu_gnss;
            
            float gps_x = gps_pos_corrected(0);
            float gps_y = gps_pos_corrected(1);
            float gps_z = gps_pos_corrected(2);
            
            if (!useGpsElevation)           //  是否使用gps的高度
            {
                gps_z = transformTobeMapped[5];
                noise_z = 0.01;
            }

            // (0,0,0)无效数据
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
            {
                ROS_INFO("[GPS] addGPSFactor: invalid (0,0,*) GNSS position, continue");
                continue;
            }
            // 每隔5m添加一个GPS里程计
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0 && pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back())  > 6.0){
                // cout<<"33333333"<<endl;
                ROS_INFO("[GPS] addGPSFactor: distance to last GPS < 5m, continue");
                continue;
            }
            
            else
                lastGPSPoint = curGPSPoint;
            // 添加GPS因子
            gtsam::Vector Vector3(3);
            Vector3 << max(noise_x, gpsNoiseMin), max(noise_y, gpsNoiseMin), max(noise_z, gpsNoiseMin);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
            gtSAMgraph.add(gps_factor);

            gpsisfound = true;
            ROS_INFO("GPS Factor Added");
            ROS_INFO("[GPS] addGPSFactor: added GPS factor and break loop");
            break;
        }
    }
}

bool computeAssociationHeightDescriptor(const Eigen::Vector3f &center_lidar,
                                        const Eigen::Vector3f &measure_lwh,
                                        float yaw_lidar,
                                        std::vector<float> &descriptor)
{
    descriptor.clear();
    if (!associationDescriptorGateEnable)
        return false;
    if (feats_undistort == nullptr || feats_undistort->empty())
        return false;
    if (associationDescriptorGridX <= 0 || associationDescriptorGridY <= 0)
        return false;

    const float half_l = 0.5f * measure_lwh.x();
    const float half_w = 0.5f * measure_lwh.y();
    const float half_h = 0.5f * measure_lwh.z();
    if (half_l <= 0.0f || half_w <= 0.0f || half_h <= 0.0f)
        return false;

    const int cell_num = associationDescriptorGridX * associationDescriptorGridY;
    std::vector<int> point_count(cell_num, 0);
    std::vector<float> max_height(cell_num, -std::numeric_limits<float>::max());
    Eigen::Matrix3f R_lidar =
        Eigen::AngleAxisf(yaw_lidar, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    for (const auto &pt : feats_undistort->points)
    {
        Eigen::Vector3f p_lidar(pt.x, pt.y, pt.z);
        Eigen::Vector3f p_local = R_lidar.transpose() * (p_lidar - center_lidar);
        if (std::fabs(p_local.x()) > half_l ||
            std::fabs(p_local.y()) > half_w ||
            std::fabs(p_local.z()) > half_h)
            continue;

        float u = (p_local.x() + half_l) / (2.0f * half_l);
        float v = (p_local.y() + half_w) / (2.0f * half_w);
        int ix = static_cast<int>(u * associationDescriptorGridX);
        int iy = static_cast<int>(v * associationDescriptorGridY);
        ix = std::max(0, std::min(associationDescriptorGridX - 1, ix));
        iy = std::max(0, std::min(associationDescriptorGridY - 1, iy));
        int cell_idx = iy * associationDescriptorGridX + ix;

        point_count[cell_idx]++;
        // 将局部 z 平移到 [0, h] 后再归一化，描述车体高度分布而不是绝对高度。
        float normalized_height = (p_local.z() + half_h) / (2.0f * half_h);
        max_height[cell_idx] = std::max(max_height[cell_idx], normalized_height);
    }

    descriptor.assign(cell_num, 0.0f);
    int valid_cells = 0;
    for (int i = 0; i < cell_num; ++i)
    {
        if (point_count[i] >= associationDescriptorMinPointsPerCell)
        {
            descriptor[i] = max_height[i];
            valid_cells++;
        }
    }

    if (valid_cells < associationDescriptorMinValidCells)
    {
        descriptor.clear();
        return false;
    }
    return true;
}

bool setFrame() {
    double timeLaserInfoCur = lidar_start_time;
    {
        std::lock_guard<std::mutex> lock(decLock);
        cout << "decQueue.size()" << decQueue.size() << endl;
    }
    const double TIMESTAMP_TOLERANCE = 0.001; // 时间戳匹配容差：0.001秒
    const int MAX_DETECTION_WAIT_MS = std::max(0, detection_wait_ms);
    std_msgs::Float64MultiArray thisDec;
    bool hasMatchedDetection = false;
    
    // 查找当前雷达帧对应的检测结果。TRT/TF 实时检测有计算延迟，
    // 主循环是单线程 spinOnce 模式，因此等待期间也要主动 spinOnce 让 /detect3d 回调入队。
    int wait_ms = 0;
    while (wait_ms < MAX_DETECTION_WAIT_MS) {
        bool need_wait = true;
        {
            std::lock_guard<std::mutex> lock(decLock);
            while (!decQueue.empty()) {
                std_msgs::Float64MultiArray candidateDec = decQueue.front();
                double timeDiffSigned = candidateDec.data[0] - timeLaserInfoCur;

                if (timeDiffSigned < (-1) * TIMESTAMP_TOLERANCE) {
                    ROS_WARN_STREAM("Skipping detection with timestamp " << std::fixed << std::setprecision(8) << candidateDec.data[0]
                                    << " (too early, current lidar: " << std::fixed << std::setprecision(8) << timeLaserInfoCur << ")");
                    decQueue.pop_front();
                    continue;
                }

                if (std::abs(timeDiffSigned) < TIMESTAMP_TOLERANCE) {
                    thisDec = candidateDec;
                    hasMatchedDetection = true;
                    need_wait = false;
                    break;
                }

                // 队首检测比当前雷达帧更晚，可能是当前帧检测还没到；先等一下，不立刻判失败。
                break;
            }
        }

        if (!need_wait) {
            break;
        }

        ros::spinOnce();
        ros::Duration(0.001).sleep();
        wait_ms++;
    }

    if (!hasMatchedDetection) {
        cout << "Waiting for test results more than " << MAX_DETECTION_WAIT_MS << "ms, no more waiting!" << endl;
        LidarSLAMFrame frame;
        frame.frame_id = flow;
        frame.objects.clear();
        frames.push_back(frame);
        return false;
    }
    
    // 最终检查时间戳是否匹配（必须在0.001秒以内）
    double timeDiff = abs(thisDec.data[0] - timeLaserInfoCur);
    if (timeDiff >= TIMESTAMP_TOLERANCE) {
        ROS_WARN_STREAM("Detection timestamp mismatch: detection=" << thisDec.data[0] 
                        << ", lidar=" << timeLaserInfoCur 
                        << ", diff=" << timeDiff << "s (threshold=" << TIMESTAMP_TOLERANCE << "s)");
        // 时间戳不匹配，返回空帧
        LidarSLAMFrame frame;
        frame.frame_id = flow;
        frame.objects.clear();
        frames.push_back(frame);
        // 如果检测数据太早，移除它；如果太晚，保留等待下一帧
        if (thisDec.data[0] < timeLaserInfoCur) {
            std::lock_guard<std::mutex> lock(decLock);
            decQueue.pop_front();
        }
        return false;
    }
    
    int ob_num = (thisDec.data.size() - 1) / 9;
    
    // 调试信息：打印数据大小和物体数量
    ROS_INFO_STREAM("Matched detection: timestamp=" << thisDec.data[0] 
                    << ", lidar=" << timeLaserInfoCur 
                    << ", diff=" << timeDiff << "s, ob_num=" << ob_num);
    if (ob_num > 0) {
        ROS_INFO_STREAM("First object: type=" << thisDec.data[1] 
                        << " x=" << thisDec.data[2] 
                        << " y=" << thisDec.data[3]
                        << " z=" << thisDec.data[4]
                        << " score=" << thisDec.data[9]);
    }
    
    if (abs(thisDec.data[0] - timeLaserInfoCur) < TIMESTAMP_TOLERANCE) {
        LidarSLAMFrame frame;
        frame.frame_id = flow;
        vector<LidarSLAMObject> objects;
        int filtered_by_type = 0;
        int filtered_by_position = 0;
        int filtered_by_score = 0;
        
        for (int i = 0; i < ob_num; i++) {
            LidarSLAMObject ob;
            ob.frame_id = flow;
            ob.obj_status = 1;
            ob.type = thisDec.data[1 + 9 * i];
            ob.score = thisDec.data[9 + 9 * i];
            if (ob.type != 0) {
                filtered_by_type++;
                continue;
            }
            Eigen::Vector3f local_xyz = Eigen::Vector3f(float(thisDec.data[2 + 9 * i]), float(thisDec.data[3 + 9 * i]), float(thisDec.data[4 + 9 * i]));

            if (local_xyz[0] <= -30.0 || abs(local_xyz[1]) > 24.0) {
                filtered_by_position++;
                continue;
            }
            if (ob.score < Scorethre) {
                filtered_by_score++;
                continue;
            }
            Eigen::Vector3f measure_lwh = Eigen::Vector3f(float(thisDec.data[5 + 9 * i]), float(thisDec.data[6 + 9 * i]), float(thisDec.data[7 + 9 * i]));
            Eigen::Vector3f rotation_zyx = Eigen::Vector3f(float(thisDec.data[8 + 9 * i]), 0.0, 0.0);
            for (int i = 0; i < 3; i++) {
                ob.local_xyz[i] = local_xyz[i];
                ob.detect_local_xyz[i] = local_xyz[i];
                ob.rotation_zyx[i] = rotation_zyx(i);
                ob.detect_rotation_zyx[i] = rotation_zyx(i);
                ob.measure_lwh[i] = measure_lwh(i);
            }
            ob.has_detect_pose = true;
            // 宽松误关联检测使用的局部高度描述子；默认参数关闭时不会做额外点云遍历。
            ob.has_height_descriptor = computeAssociationHeightDescriptor(
                local_xyz,
                measure_lwh,
                rotation_zyx[0],
                ob.height_descriptor);
            // Eigen::Affine3f local_t = Eigen::Affine3f::Identity();
            // local_t = pcl::getTransformation(local_xyz[0], local_xyz[1], local_xyz[2], 0, 0, rotation_zyx[0]);
            // ob.pose_inimu[0] = local_t.rotation().eulerAngles(0, 1, 2)[0];
            // ob.pose_inimu[1] = local_t.rotation().eulerAngles(0, 1, 2)[1];
            // ob.pose_inimu[2] = local_t.rotation().eulerAngles(0, 1, 2)[2];
            // ob.pose_inimu[3] = local_t.translation()[0];
            // ob.pose_inimu[4] = local_t.translation()[1];
            // ob.pose_inimu[5] = local_t.translation()[2];
            // LiDAR系下的局部变换
            Eigen::Affine3f local_t_lidar = pcl::getTransformation(local_xyz[0], local_xyz[1], local_xyz[2], 0, 0, rotation_zyx[0]);
            // 将 LiDAR 系位姿转换到 IMU/Body 系： p_imu = R_li * p_lidar + t_li
            Eigen::Matrix3f R_li = Lidar_R_wrt_IMU.cast<float>();            // 全局 LiDAR->IMU 旋转（double->float）
            Eigen::Vector3f t_li = Lidar_T_wrt_IMU.cast<float>();            // 全局 LiDAR->IMU 平移（double->float）
            Eigen::Matrix3f R_obj_imu = R_li * local_t_lidar.rotation();     // 目标在 IMU 系下的旋转
            Eigen::Vector3f t_obj_imu = R_li * local_t_lidar.translation() + t_li; // 目标在 IMU 系下的平移
            Eigen::Affine3f local_t_imu = Eigen::Affine3f::Identity();
            local_t_imu.linear() = R_obj_imu;
            local_t_imu.translation() = t_obj_imu;
            // 填充到 ob.pose_inimu（IMU 系下）
            ob.pose_inimu[0] = local_t_imu.rotation().eulerAngles(0, 1, 2)[0];
            ob.pose_inimu[1] = local_t_imu.rotation().eulerAngles(0, 1, 2)[1];
            ob.pose_inimu[2] = local_t_imu.rotation().eulerAngles(0, 1, 2)[2];
            ob.pose_inimu[3] = local_t_imu.translation()[0];
            ob.pose_inimu[4] = local_t_imu.translation()[1];
            ob.pose_inimu[5] = local_t_imu.translation()[2];
            objects.push_back(ob);
        }
        
        // 打印过滤统计信息
        ROS_INFO_STREAM("Filter stats - Total: " << ob_num 
                        << ", Passed: " << objects.size()
                        << ", Type filtered: " << filtered_by_type
                        << ", Position filtered: " << filtered_by_position
                        << ", Score filtered: " << filtered_by_score);
        
        frame.objects = objects;
        frames.push_back(frame);
        std::lock_guard<std::mutex> lock(decLock);
        decQueue.pop_front();
        return true;
    }
    else {
        cout << "something wrong!!!" << endl;
        return false;
    }
}

void saveKeyFramesAndFactor()
{
    current_mot_tracking_time_ms = 0.0;
    current_mot_graph_time_ms = 0.0;
    current_global_graph_time_ms = 0.0;
    current_backend_graph_time_ms = 0.0;
    bool local_graph_optimized_this_frame = false;
    int object_local_map_factor_count_this_frame = 0;
    Eigen::Affine3f fast_lio_pose_before_local = trans2Affine3f(transformTobeMapped);
    Eigen::Affine3f local_optimized_pose_this_frame = fast_lio_pose_before_local;

    if (if_dynamic){
        setFrame();//这一帧的object和flow
        // 当前帧在 body/world 下的位姿
        Eigen::Affine3f local_pose = pcl::getTransformation(
            (float)transformTobeMapped[3], (float)transformTobeMapped[4], (float)transformTobeMapped[5],
            (float)transformTobeMapped[0], (float)transformTobeMapped[1], (float)transformTobeMapped[2]);
        fast_lio_pose_before_local = local_pose;
        // 使用上一帧的 last_pos/last_rot 计算帧间增量：odom_incre = last_pose^{-1} * local_pose
        Eigen::Affine3f odom_incre = Eigen::Affine3f::Identity();
        if (flow > 0) {
            Eigen::Affine3f last_pose = pcl::getTransformation(
                static_cast<float>(last_pos(0)), static_cast<float>(last_pos(1)), static_cast<float>(last_pos(2)),
                static_cast<float>(last_rot(0)), static_cast<float>(last_rot(1)), static_cast<float>(last_rot(2)));
            odom_incre = last_pose.inverse() * local_pose;
        } else {
            odom_incre = Eigen::Affine3f::Identity();
        }
         frames[flow].optimize_pose = local_pose;
         
          // 1. Calculate the initial value of the object's pose
         for (int n = 0; n < frames[flow].objects.size(); n++) {
            Eigen::Affine3f global_t = Eigen::Affine3f::Identity();
            global_t = local_pose * trans2Affine3f(frames[flow].objects[n].pose_inimu);
            Affine3ftovec6f(global_t, frames[flow].objects[n].optimize_t);
         }
            // 2. Start tracking
            double mot_tracking_start = omp_get_wtime();
            tracker.AssociateObjects(frames, flow, vel_threshold, dynamBoxBuf,
                                     associationDescriptorGateEnable,
                                     associationDescriptorCosThreshold);
            current_mot_tracking_time_ms = (omp_get_wtime() - mot_tracking_start) * 1000.0;
            total_mot_tracking_time_ms += current_mot_tracking_time_ms;
            mot_tracking_frame_count++;

            // 3. Add factors
            double mot_graph_start = omp_get_wtime();
            // 3.1 Add LiDAR odometry factors
            frames[flow].vertex_local_id = local_graph.key_id;
            if (flow == 0) {
                priorfactor_id[flow].push_back(local_graph.f_id);
                priorfactor_egoP_id[flow].push_back(local_graph.f_id);
                local_graph.AddRobustPriorFactor(frames[flow].vertex_local_id, Affine3f2Pose3(local_pose), egoP_egoP, rubostNum);
                local_graph.key_id++;
                local_graph.setinitialEstimate(frames[flow].vertex_local_id, Affine3f2Pose3(local_pose));
            }
            if (flow > 0) {
                frames[flow].edge_local_id = local_graph.f_id;
                // 使用基于 last_pos/last_rot 与 local_pose 计算得到的 odom_incre
                local_graph.AddRobustBTW2factor(frames[flow - 1].vertex_local_id, frames[flow].vertex_local_id, Affine3f2Pose3(odom_incre), egoP_egoP, rubostNum);
                local_graph.key_id++;
                local_graph.setinitialEstimate(frames[flow].vertex_local_id, Affine3f2Pose3(local_pose));
            }

            // 3.2 Add object motion factors
            for (int k = 0; k < frames[flow].objects.size(); k++) {
                if (!frames[flow].objects[k].initialized) {
                    continue;
                }
                else {
                    int last_as_ob = frames[flow].objects[k].associated_object_last.second;
                    if (frames[flow].objects[k].dynamic) {
                        frames[flow].objects[k].vertex_id = local_graph.key_id;
                        if (if_priorfactor)
                            if (frames[flow - 1].objects[last_as_ob].flag1 == -1) {
                                priorfactor_id[flow].push_back(local_graph.f_id);
                                pair<int, int> p(local_graph.f_id, k);
                                priorfactor_objP_id[flow].push_back(p);
                                local_graph.AddRobustPriorFactor(frames[flow].objects[k].vertex_id, vec6ftoPose3(frames[flow].objects[k].optimize_t), egoP_objP, rubostNum);
                            }
                        frames[flow].objects[k].flag1 = 1;

                        frames[flow].objects[k].edge_id_link_frame_pose = local_graph.f_id;
                        local_graph.AddRobustBTW2factor(frames[flow].vertex_local_id, frames[flow].objects[k].vertex_id, trans2gtsamPose(frames[flow].objects[k].pose_inimu), egoP_objP, rubostNum);
                        local_graph.setinitialEstimate(frames[flow].objects[k].vertex_id, vec6ftoPose3(frames[flow].objects[k].optimize_t));
                        local_graph.key_id++;
                    }
                    else {
                        if (frames[flow - 1].objects[last_as_ob].vertex_id == -1 ||
                            (frames[flow - 1].objects[last_as_ob].vertex_id != -1 && frames[flow - 1].objects[last_as_ob].dynamic)) {
                            frames[flow].objects[k].vertex_id = local_graph.key_id;
                            if (if_priorfactor)
                                if (frames[flow - 1].objects[last_as_ob].flag1 == -1) {
                                    priorfactor_id[flow].push_back(local_graph.f_id);
                                    pair<int, int> p(local_graph.f_id, k);
                                    priorfactor_objP_id[flow].push_back(p);
                                    local_graph.AddRobustPriorFactor(frames[flow].objects[k].vertex_id, vec6ftoPose3(frames[flow].objects[k].optimize_t), egoP_objP, rubostNum);
                                }
                            frames[flow].objects[k].flag1 = 1;
                            frames[flow].objects[k].edge_id_link_frame_pose = local_graph.f_id;
                            local_graph.AddRobustBTW2factor(frames[flow].vertex_local_id, frames[flow].objects[k].vertex_id, trans2gtsamPose(frames[flow].objects[k].pose_inimu), ego_stationObjP, rubostNum);
                            local_graph.setinitialEstimate(frames[flow].objects[k].vertex_id, vec6ftoPose3(frames[flow].objects[k].optimize_t));
                            local_graph.key_id++;
                        }
                        else {
                            frames[flow].objects[k].vertex_id = frames[flow - 1].objects[last_as_ob].vertex_id;
                            frames[flow].objects[k].edge_id_link_frame_pose = local_graph.f_id;
                            local_graph.AddRobustBTW2factor(frames[flow].vertex_local_id, frames[flow].objects[k].vertex_id, trans2gtsamPose(frames[flow].objects[k].pose_inimu), ego_stationObjP, rubostNum);
                        }
                    }

                    // 3.3 Add object motion factors
                    if (frames[flow - 1].objects[last_as_ob].vertex_id != -1 &&
                        (frames[flow].objects[k].dynamic ||
                            (!frames[flow].objects[k].dynamic && frames[flow - 1].objects[last_as_ob].dynamic))) {
                        Eigen::Affine3f last_ob_t = gtsamPose3toAffine3f(vec6ftoPose3(frames[flow - 1].objects[last_as_ob].optimize_t));
                        Eigen::Affine3f change = last_ob_t.inverse() * gtsamPose3toAffine3f(vec6ftoPose3(frames[flow].objects[k].optimize_t));
                        frames[flow].objects[k].vertex_id_pose_change = local_graph.key_id;
                        if (if_priorfactor)
                            if (frames[flow - 1].objects[last_as_ob].flag2 == -1) {
                                priorfactor_id[flow - 1].push_back(local_graph.f_id);
                                local_graph.AddRobustPriorFactor(frames[flow].objects[k].vertex_id_pose_change, Affine3f2Pose3(change), objP_objP_chgP, rubostNum);
                            }
                        frames[flow].objects[k].flag2 == 1;
                        frames[flow].objects[k].edge_id_link_2object_with_pose_change = local_graph.f_id;

                        local_graph.AddRobustBTW3factor(frames[flow - 1].objects[last_as_ob].vertex_id, frames[flow].objects[k].vertex_id, frames[flow].objects[k].vertex_id_pose_change, objP_objP_chgP, rubostNum);
                        local_graph.key_id++;

                        local_graph.setinitialEstimate(frames[flow].objects[k].vertex_id_pose_change, Affine3f2Pose3(change));
                        // 3.4 Add smooth motion factors
                        if (frames[flow - 1].objects[last_as_ob].vertex_id_pose_change != -1) {
                            frames[flow].objects[k].edge_id_link_2pose_change = local_graph.f_id;
                            local_graph.AddRobustBTW2factor(frames[flow - 1].objects[last_as_ob].vertex_id_pose_change,
                                frames[flow].objects[k].vertex_id_pose_change, Affine3f2Pose3(Eigen::Affine3f::Identity()), chgP_chgP, rubostNum);
                        }
                    }
                }
            }

            object_local_map_factor_count_this_frame = addObjectLocalMapFactorsBeforeOptimization();

            // 4. Sliding window, remove old factors
            if (flow >= window_size) {
                gtsam::Marginals marginals(graph_temp, local_graph.result);
                int margin_frame = flow - window_size;

                for (auto it = priorfactor_id[margin_frame].begin(); it != priorfactor_id[margin_frame].end(); it++) {
                    local_graph.Removefactor(*it);
                }

                local_graph.Removefactor(frames[margin_frame + 1].edge_local_id);
                local_graph.Removekey(frames[margin_frame].vertex_local_id);
                priorfactor_id[margin_frame + 1].push_back(local_graph.f_id);
                priorfactor_egoP_id[margin_frame + 1].push_back(local_graph.f_id);

                local_graph.AddRobustMarginalPrior(frames[margin_frame + 1].vertex_local_id, local_graph.result.at<gtsam::Pose3>(frames[margin_frame + 1].vertex_local_id), marginals, rubostNum);

                for (int k = 0; k < frames[margin_frame].objects.size(); k++) {
                    int last_as_ob = frames[margin_frame].objects[k].associated_object_last.second;
                    int next_as_ob = frames[margin_frame].objects[k].associated_object_next.second;
                    if (frames[margin_frame].objects[k].vertex_id != -1) {
                        if (frames[margin_frame].objects[k].dynamic) {
                            local_graph.Removefactor(frames[margin_frame].objects[k].edge_id_link_frame_pose);
                            local_graph.Removekey(frames[margin_frame].objects[k].vertex_id);
                            if (next_as_ob == -1) {
                                continue;
                            }
                            if (if_priorfactor && frames[margin_frame + 1].objects[next_as_ob].vertex_id != -1) {
                                priorfactor_id[margin_frame + 1].push_back(local_graph.f_id);
                                pair<int, int> p(local_graph.f_id, next_as_ob);
                                priorfactor_objP_id[margin_frame + 1].push_back(p);
                                local_graph.AddRobustMarginalPrior(frames[margin_frame + 1].objects[next_as_ob].vertex_id, local_graph.result.at<gtsam::Pose3>(frames[margin_frame + 1].objects[next_as_ob].vertex_id), marginals, rubostNum);
                            }

                            if (frames[margin_frame + 1].objects[next_as_ob].vertex_id_pose_change != -1) {
                                local_graph.Removefactor(frames[margin_frame + 1].objects[next_as_ob].edge_id_link_2object_with_pose_change);
                                local_graph.Removekey(frames[margin_frame + 1].objects[next_as_ob].vertex_id_pose_change);
                                int n_next_ob = frames[margin_frame + 1].objects[next_as_ob].associated_object_next.second;
                                if (n_next_ob != -1)
                                    if (frames[margin_frame + 2].objects[n_next_ob].vertex_id_pose_change != -1) {
                                        if (if_priorfactor) {
                                            priorfactor_id[margin_frame + 1].push_back(local_graph.f_id);
                                            local_graph.AddRobustMarginalPrior(frames[margin_frame + 2].objects[n_next_ob].vertex_id_pose_change, local_graph.result.at<gtsam::Pose3>(frames[margin_frame + 2].objects[n_next_ob].vertex_id_pose_change), marginals, rubostNum);
                                        }
                                        if (frames[margin_frame + 2].objects[n_next_ob].edge_id_link_2pose_change != -1)
                                            local_graph.Removefactor(frames[margin_frame + 2].objects[n_next_ob].edge_id_link_2pose_change);
                                    }
                            }
                        }
                        else {
                            local_graph.Removefactor(frames[margin_frame].objects[k].edge_id_link_frame_pose);
                            if (next_as_ob == -1 || (next_as_ob != -1 && frames[margin_frame + 1].objects[next_as_ob].dynamic)) {
                                local_graph.Removekey(frames[margin_frame].objects[k].vertex_id);
                            }
                            if (next_as_ob == -1)
                                continue;
                            if (if_priorfactor && frames[margin_frame + 1].objects[next_as_ob].vertex_id != -1) {
                                priorfactor_id[margin_frame + 1].push_back(local_graph.f_id);
                                pair<int, int> p(local_graph.f_id, next_as_ob);
                                priorfactor_objP_id[margin_frame + 1].push_back(p);
                                local_graph.AddRobustMarginalPrior(frames[margin_frame + 1].objects[next_as_ob].vertex_id, local_graph.result.at<gtsam::Pose3>(frames[margin_frame + 1].objects[next_as_ob].vertex_id), marginals, rubostNum);
                            }
                            if (frames[margin_frame + 1].objects[next_as_ob].vertex_id_pose_change != -1) {
                                local_graph.Removefactor(frames[margin_frame + 1].objects[next_as_ob].edge_id_link_2object_with_pose_change);
                                local_graph.Removekey(frames[margin_frame + 1].objects[next_as_ob].vertex_id_pose_change);
                                int n_next_ob = frames[margin_frame + 1].objects[next_as_ob].associated_object_next.second;
                                if (n_next_ob != -1)
                                    if (frames[margin_frame + 2].objects[n_next_ob].vertex_id_pose_change != -1) {
                                        if (if_priorfactor) {
                                            priorfactor_id[margin_frame + 1].push_back(local_graph.f_id);
                                            local_graph.AddRobustMarginalPrior(frames[margin_frame + 2].objects[n_next_ob].vertex_id_pose_change, local_graph.result.at<gtsam::Pose3>(frames[margin_frame + 2].objects[n_next_ob].vertex_id_pose_change), marginals, rubostNum);
                                        }
                                        if (frames[margin_frame + 2].objects[n_next_ob].edge_id_link_2pose_change != -1)
                                            local_graph.Removefactor(frames[margin_frame + 2].objects[n_next_ob].edge_id_link_2pose_change);
                                    }
                            }
                        }
                    }
                }
            }

            // 5. Start optimization and updated results 
            cout << "Start optimize!   non-null factors size: " << local_graph.gtSAMgraph2.nrFactors();
            auto Time0 = std::chrono::steady_clock::now();
            local_graph.StartOptimiz(local_graph.gtSAMgraph2.nrFactors());
            auto Time1 = std::chrono::steady_clock::now();
            cout << "Optimize end! opt time = " << std::chrono::duration<double>(Time1 - Time0).count() * 1000 << " ms" << endl;

            gtsam::Pose3 this_pose = local_graph.result.at<gtsam::Pose3>(frames[flow].vertex_local_id);
            frames[flow].optimize_pose = gtsamPose3toAffine3f(local_graph.result.at<gtsam::Pose3>(frames[flow].vertex_local_id));
            local_optimized_pose_this_frame = frames[flow].optimize_pose;
            local_graph_optimized_this_frame = true;
            for (int k = 0; k < frames[flow].objects.size(); k++) {
                if (frames[flow].objects[k].vertex_id != -1) {
                    Pose3tovec6f(local_graph.result.at<gtsam::Pose3>(frames[flow].objects[k].vertex_id), frames[flow].objects[k].optimize_t);
                    Eigen::Affine3f local_t = frames[flow].optimize_pose.inverse() * gtsamPose3toAffine3f(local_graph.result.at<gtsam::Pose3>(frames[flow].objects[k].vertex_id));
                    for (int i = 0; i < 3; i++) {
                        frames[flow].objects[k].local_xyz[i] = local_t.translation()[i];
                    }
                    tracker.Update_TrackUtmXY(frames[flow].objects[k].object_id, frames[flow].objects[k].optimize_t[3], frames[flow].objects[k].optimize_t[4]);
                    if (frames[flow].objects[k].vertex_id_pose_change != -1) {
                        Pose3tovec6f(local_graph.result.at<gtsam::Pose3>(frames[flow].objects[k].vertex_id_pose_change), frames[flow].objects[k].v_t);
                        std::vector<float> v_t = frames[flow].objects[k].v_t;
                        double dd = sqrt(v_t[3] * v_t[3] + v_t[4] * v_t[4] + v_t[5] * v_t[5]);
                        frames[flow].objects[k].velocity = dd / 0.1 * 3.6;
                    }
                }
            }

            // 目标优化完成后，保存当前帧所有目标的优化位姿到文本文件
            saveObjectTrackingResult();
            accumulateTrackedObjectCloud();
            accumulateObjectLocalMaps();

            // 局部因子图优化后，不更新自身姿态，保持原来的transformTobeMapped
            // transformTobeMapped[0] = this_pose.rotation().roll();
            // transformTobeMapped[1] = this_pose.rotation().pitch();
            // transformTobeMapped[2] = this_pose.rotation().yaw();
            // transformTobeMapped[3] = this_pose.x();
            // transformTobeMapped[4] = this_pose.y();
            // transformTobeMapped[5] = this_pose.z();


            graph_temp = local_graph.gtSAMgraph2;
            current_mot_graph_time_ms = (omp_get_wtime() - mot_graph_start) * 1000.0;



        

    }
    if (backendResultMode == 0 && local_graph_optimized_this_frame)
    {
        applyPoseToEskfState(local_optimized_pose_this_frame);
    }

    //  计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
    if (saveFrame() == false){
        PointTypePose currentPose6D = trans2PointTypePose(transformTobeMapped);
        currentPose6D.time = lidar_end_time;
        recordIntermediateFrameForMap(currentPose6D);
        current_backend_graph_time_ms = current_mot_graph_time_ms + current_global_graph_time_ms;

        // 局部因子图优化后，不更新自身姿态，保持原来的ESKF状态
        // if (if_dynamic){
        //     //修改ESKF状态为当前局部因子图优化后的结果
        //     state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
        //     Eigen::Vector3d pos(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]);
        //     Eigen::Quaterniond q = EulerToQuat(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

        // //  更新状态量
        //     state_updated.pos = pos;
        //     state_updated.rot =  q;
        //     state_point = state_updated; // 对state_point进行更新，state_point可视化用到
        // // if(aLoopIsClosed == true )
        //     kf.change_x(state_updated);  //  对cur_pose 进行isam2优化后的修正 
        // }

         // 保存当前帧的平移和旋转到last_pose
        state_ikfom state_now = kf.get_x();
        last_pos = state_now.pos.cast<double>();
        Eigen::Vector3d rot_ang = state_now.rot.toRotationMatrix().eulerAngles(2,1,0); 
        last_rot(0) = rot_ang(2);//roll
        last_rot(1) = rot_ang(1);//pitch
        last_rot(2) = rot_ang(0);//yaw
        flow++;
        return;
    }

    if (backendResultMode == 0 && local_graph_optimized_this_frame)
    {
        double local_only_start = omp_get_wtime();
        appendLocalOnlyKeyFrame();

        state_ikfom state_now = kf.get_x();
        last_pos = state_now.pos.cast<double>();
        Eigen::Vector3d rot_ang = state_now.rot.toRotationMatrix().eulerAngles(2,1,0);
        last_rot(0) = rot_ang(2);//roll
        last_rot(1) = rot_ang(1);//pitch
        last_rot(2) = rot_ang(0);//yaw

        current_global_graph_time_ms = 0.0;
        current_backend_graph_time_ms = current_mot_graph_time_ms + (omp_get_wtime() - local_only_start) * 1000.0;
        flow++;
        return;
    }

    // 激光里程计因子(from fast-lio),  输入的是frame_relative pose  帧间位姿(body 系下)
    double global_graph_start = omp_get_wtime();
    int current_global_key = cloudKeyPoses3D->size();
    addOdomFactor();
    if (backendResultMode == 1 && local_graph_optimized_this_frame)
    {
        addLocalGraphFeedbackToGlobalIfNeeded(current_global_key,
                                              fast_lio_pose_before_local,
                                              local_optimized_pose_this_frame,
                                              object_local_map_factor_count_this_frame);
    }
    // GPS因子 (UTM -> WGS84)
    if (use_gnss){
        addGPSFactor();
    }    
    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();
    // 执行优化
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (aLoopIsClosed == true) // 有回环因子，多update几次
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }
    // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    // 优化结果
    isamCurrentEstimate = isam->calculateBestEstimate();
    // 当前帧位姿结果
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size() - 1);

    // cloudKeyPoses3D加入当前帧位置
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    // 索引
    thisPose3D.intensity = cloudKeyPoses3D->size(); //  使用intensity作为该帧点云的index
    cloudKeyPoses3D->push_back(thisPose3D);         //  新关键帧帧放入队列中

    // cloudKeyPoses6D加入当前帧位姿
    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity;
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);
    keyframeAnchorCapturePoses.push_back(thisPose6D);
    intermediateFramesPerKeyframe.emplace_back();
    if (saveMapFrameMode == 1 && saveMapRecordIntermediateFrames && saveMapIntermediateFrameAnchorMode == 1)
    {
        flushPendingIntermediateFramesToAnchor(static_cast<int>(cloudKeyPoses6D->size()) - 1);
    }

    // 位姿协方差
    poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size() - 1);

    // ESKF状态和方差  更新
    state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());

    //  更新状态量
    state_updated.pos = pos;
    state_updated.rot =  q;
    state_point = state_updated; // 对state_point进行更新，state_point可视化用到
    // if(aLoopIsClosed == true )
    kf.change_x(state_updated);  //  对cur_pose 进行isam2优化后的修正

    //保存这次的结果到last_pos,last_rot
    state_ikfom state_now = kf.get_x();
    last_pos = state_now.pos.cast<double>();
    Eigen::Vector3d rot_ang = state_now.rot.toRotationMatrix().eulerAngles(2,1,0); 
    last_rot(0) = rot_ang(2);//roll
    last_rot(1) = rot_ang(1);//pitch
    last_rot(2) = rot_ang(0);//yaw
    
    // 全局优化之后，把当前帧位姿的结果作为一个强的观测节点加入localgraph中
    if (if_dynamic && flow < frames.size() && frames[flow].vertex_local_id != -1) {
        // 使用全局优化后的协方差创建噪声模型
        gtsam::noiseModel::Base::shared_ptr robustGlobalPoseNoise;
        robustGlobalPoseNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(rubostNum),
            gtsam::noiseModel::Gaussian::Covariance(poseCovariance));
        
        // 添加先验因子到localgraph
        priorfactor_id[flow].push_back(local_graph.f_id);
        priorfactor_egoP_id[flow].push_back(local_graph.f_id);
        local_graph.gtSAMgraph2.add(gtsam::PriorFactor<gtsam::Pose3>(
            frames[flow].vertex_local_id, latestEstimate, robustGlobalPoseNoise));
        local_graph.f_id++;
    }
    if (backendResultMode == 1 && local_graph_optimized_this_frame)
    {
        updateLocalGraphKeyPoseForGlobal(local_optimized_pose_this_frame);
    }
    current_global_graph_time_ms = (omp_get_wtime() - global_graph_start) * 1000.0;
    current_backend_graph_time_ms = current_mot_graph_time_ms + current_global_graph_time_ms;
     
    // TODO:  P的修正有待考察，按照yanliangwang的做法，修改了p，会跑飞
    // esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P_updated = kf.get_P(); // 获取当前的状态估计的协方差矩阵
    // P_updated.setIdentity();
    // P_updated(6, 6) = P_updated(7, 7) = P_updated(8, 8) = 0.00001;
    // P_updated(9, 9) = P_updated(10, 10) = P_updated(11, 11) = 0.00001;
    // P_updated(15, 15) = P_updated(16, 16) = P_updated(17, 17) = 0.0001;
    // P_updated(18, 18) = P_updated(19, 19) = P_updated(20, 20) = 0.001;
    // P_updated(21, 21) = P_updated(22, 22) = 0.00001;
    // kf.change_P(P_updated);

    // 当前帧激光角点、平面点，降采样集合
    // pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    // pcl::copyPointCloud(*feats_undistort,  *thisCornerKeyFrame);
    // 根据dense_keyframe参数选择存储的点云类型
    if (dense_keyframe) {
        pcl::copyPointCloud(*feats_undistort, *thisSurfKeyFrame); // 存储关键帧,去畸变后的完整点云
    } else {
        pcl::copyPointCloud(*feats_down_body, *thisSurfKeyFrame); // 存储关键帧,降采样后的点云
    }

    // 保存特征点降采样集合
    // cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);

    updatePath(thisPose6D); //  可视化update后的path
    flow++;
}

void reconstructIKdTree(int construct_mode){
    if(reconstructKdTree){
        if (cloudKeyPoses3D->empty())
        {
            ROS_WARN("[reconstructIKdTree] cloudKeyPoses3D is empty, skip reconstruct");
            updateKdtreeCount++;
            return;
        }

        double t_recon_start = omp_get_wtime();
        /*** if path is too large, the rvis will crash ***/
        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMapPoses(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        double t_search_start = omp_get_wtime();
        if (construct_mode == 0)
        {
            mtx.lock();
            kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
            kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), reconstruct_ikdtree_search_radius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
            mtx.unlock();
        }
        else if (construct_mode == 1)
        {
            size_t total_keyframes = cloudKeyPoses3D->size();
            size_t frames_to_use = std::min(static_cast<size_t>(reconstruct_ikdtree_frame), total_keyframes);
            size_t start_idx = total_keyframes - frames_to_use;
            pointSearchIndGlobalMap.reserve(frames_to_use);
            for (size_t idx = start_idx; idx < total_keyframes; ++idx)
            {
                pointSearchIndGlobalMap.push_back(static_cast<int>(idx));
            }
        }
        else
        {
            ROS_WARN_THROTTLE(1.0, "[reconstructIKdTree] unsupported construct_mode=%d, fallback to mode 0", construct_mode);
            mtx.lock();
            kdtreeGlobalMapPoses->setInputCloud(cloudKeyPoses3D);
            kdtreeGlobalMapPoses->radiusSearch(cloudKeyPoses3D->back(), reconstruct_ikdtree_search_radius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
            mtx.unlock();
        }
        double t_search_end = omp_get_wtime();
        int max_frame = reconstruct_ikdtree_frame;
        int frame_available = std::min(max_frame, (int)pointSearchIndGlobalMap.size());
        for (int i = 0; i < frame_available; ++i){
            subMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);     //  subMap的pose集合
            
        }
        double t_ds_pose_start = omp_get_wtime();
        if (construct_mode == 0)
        {
            // 仅 mode 0 需要对关键帧位姿做体素降采样
            // pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyPoses;
            // downSizeFilterSubMapKeyPoses.setLeafSize(reconstruct_pose_density, reconstruct_pose_density, reconstruct_pose_density);
            // downSizeFilterSubMapKeyPoses.setInputCloud(subMapKeyPoses);
            // downSizeFilterSubMapKeyPoses.filter(*subMapKeyPosesDS);
            *subMapKeyPosesDS = *subMapKeyPoses;
            //因为其实在取关键帧的时候就规定了要间隔一定距离（取10）才能作为关键帧，这里也是对关键帧进行降采样，也是取差不多的值，效果不大
        }
        else
        {
            // mode 1 直接使用按时间选出的关键帧序列
            *subMapKeyPosesDS = *subMapKeyPoses;
        }
        double t_ds_pose_end = omp_get_wtime();
        // 提取局部相邻关键帧对应的特征点云
        double t_collect_start = omp_get_wtime();
        for (int i = 0; i < (int)subMapKeyPosesDS->size(); ++i)
        {
            
            // // 距离过滤
            // if (pointDistance(subMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > reconstruct_ikdtree_search_radius)
            //         continue;
            int thisKeyInd = (int)subMapKeyPosesDS->points[i].intensity;
            // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *subMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
        }
        double t_collect_end = omp_get_wtime();
        // 降采样，发布
        double t_ds_map_start = omp_get_wtime();
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(reconstruct_map_leafsize, reconstruct_map_leafsize, reconstruct_map_leafsize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(subMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*subMapKeyFramesDS);
        double t_ds_map_end = omp_get_wtime();

        std::cout << "subMapKeyFramesDS sizes  =  "   << subMapKeyFramesDS->points.size()  << std::endl;
        double t_rebuild_start = omp_get_wtime();
        ikdtree.reconstruct(subMapKeyFramesDS->points);
        double t_rebuild_end = omp_get_wtime();
        updateKdtreeCount = 0;
        ROS_INFO("Reconstructed ikdtree");
        int featsFromMapNum = ikdtree.validnum();
        kdtree_size_st = ikdtree.size();
        std::cout << "featsFromMapNum  =  "   << featsFromMapNum   <<  "\t" << " kdtree_size_st   =  "  <<  kdtree_size_st  << std::endl;

        double total_ms      = (t_recon_start   - t_recon_start) * 1000.0; // 占位，下面直接按子阶段相加
        double search_ms     = (t_search_end    - t_search_start) * 1000.0;
        double ds_pose_ms    = (t_ds_pose_end   - t_ds_pose_start) * 1000.0;
        double collect_ms    = (t_collect_end   - t_collect_start) * 1000.0;
        double ds_map_ms     = (t_ds_map_end    - t_ds_map_start) * 1000.0;
        double rebuild_ms    = (t_rebuild_end   - t_rebuild_start) * 1000.0;
        double recon_ms      = (t_rebuild_end   - t_recon_start) * 1000.0;

        ROS_INFO("[TIME][reconstructIKdTree] search=%.3f ms, ds_pose=%.3f ms, collect=%.3f ms, ds_map=%.3f ms, rebuild=%.3f ms, total=%.3f ms",
                 search_ms, ds_pose_ms, collect_ms, ds_map_ms, rebuild_ms, recon_ms);

    }
        updateKdtreeCount ++ ; 
}

/**
 * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
 */
void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true||gpsisfound == true)
    {
        // 清空里程计轨迹
        globalPath.poses.clear();
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D->points[i]);
        }
        // 清空局部map， reconstruct  ikdtree submap
        double t_correct_start = omp_get_wtime();
        if (aLoopIsClosed && if_reconstruct_ikdtree_map_loop_closure){
            updateKdtreeCount++;
            reconstructIKdTree(reconstruct_ikdtree_mode_loop_closure);
        }
        else if(gpsisfound && if_reconstruct_ikdtree_map_add_gps){
            updateKdtreeCount++;
            if (updateKdtreeCount >= add_gps_num_reconstruct){
            reconstructIKdTree(reconstruct_ikdtree_mode_add_gps);
            }
        }
        double t_correct_end = omp_get_wtime();
        ROS_INFO("ISMA2 Update");
        ROS_INFO("[TIME] reconstruct cost: %.3f ms", (t_correct_end - t_correct_start) * 1000.0);
        gpsisfound = false;
        aLoopIsClosed = false;
    }
}

//回环检测三大要素
// 1.设置最小时间差，太近没必要
// 2.控制回环的频率，避免频繁检测，每检测一次，就做一次等待
// 3.根据当前最小距离重新计算等待时间
bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    // 当前关键帧帧
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1; //  当前关键帧索引
    int loopKeyPre = -1;

    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    std::vector<int> pointSearchIndLoop;                        //  候选关键帧索引
    std::vector<float> pointSearchSqDisLoop;                    //  候选关键帧距离
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D); //  历史帧构建kdtree
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;
    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    ROS_INFO("Find loop clousre frame ");
    return true;
}

/**
 * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
 */
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    auto surfcloud_keyframes_size = surfCloudKeyFrames.size() ;
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize)
            continue;

        if (keyNear < 0 || keyNear >= surfcloud_keyframes_size)
            continue;

        // *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        // 注意：cloudKeyPoses6D 存储的是 T_w_b , 而点云是lidar系下的，构建icp的submap时，需要通过外参数T_b_lidar 转换 , 参考pointBodyToWorld 的转换
        *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]); //  fast-lio 没有进行特征提取，默认点云就是surf
    }

    if (nearKeyframes->empty())
        return;

    // 降采样
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

void performLoopClosure()
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";
    if (cloudKeyPoses3D->points.empty() == true)
    {
        return;
    }

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur;
    int loopKeyPre;
    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
    {
        return;
    }

    // 提取
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>()); //  cue keyframe
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>()); //   history keyframe submap
    {
        // 提取当前关键帧特征点集合，降采样
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0); //  将cur keyframe 转换到world系下
        // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum); //  选取historyKeyframeSearchNum个keyframe拼成submap
        // 如果特征点较少，返回
        // if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        //     return;
        // 发布闭环匹配关键帧局部map
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // scan-to-map，调用icp匹配
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    // 未收敛，或者匹配不够好
    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    std::cout << "icp  success  " << std::endl;

    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // 闭环优化前当前帧位姿
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); //  获取上一帧 相对 当前帧的 位姿
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    // float noiseScore = icp.getFitnessScore() ; 
    float noiseScore = loopClosureNoiseScore; //  loop_clousre  noise from config
    
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
    std::cout << "loopNoiseQueue   =   " << noiseScore << std::endl;

    // 添加闭环因子需要的数据
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    loopIndexContainer[loopKeyCur] = loopKeyPre; //   使用hash map 存储回环对
}

void loadDetectionsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open detection file: " << filename << std::endl;
        ROS_ERROR_STREAM("Failed to open detection file: " << filename);
        return;
    }
    
    // 先统计文件总行数用于显示进度
    std::cout << "[INFO] Counting total lines in file: " << filename << std::endl;
    std::cout.flush();
    int totalLines = 0;
    std::string tempLine;
    std::streampos filePos = file.tellg();
    while (std::getline(file, tempLine)) {
        // 跳过空行和注释行（以#开头）
        if (tempLine.empty() || tempLine.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }
        // 跳过注释行
        std::string trimmed = tempLine;
        trimmed.erase(0, trimmed.find_first_not_of(" \t")); // 去除前导空白
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        totalLines++;
    }
    file.clear(); // 清除EOF标志
    file.seekg(0, std::ios::beg); // 重置文件指针到开头
    std::cout << "[INFO] Total lines to process: " << totalLines << std::endl;
    std::cout.flush();
    
    std::string line;
    int count = 0;
    int skipped = 0;
    int processedLines = 0;
    int lastProgressPercent = -1;
    
    std::cout << "[INFO] Loading detection data from file: " << filename << std::endl;
    std::cout.flush();
    ROS_INFO_STREAM("Loading detection data from file: " << filename);
    
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) {
            continue;
        }
        
        // 跳过注释行（以#开头）
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t")); // 去除前导空白
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        
        processedLines++;
        
        // 显示进度（每10%或每1000行）
        if (totalLines > 0) {
            int currentPercent = (processedLines * 100) / totalLines;
            if (currentPercent != lastProgressPercent && (currentPercent % 10 == 0 || processedLines % 1000 == 0)) {
                std::cout << "[INFO] Loading progress: " << currentPercent << "% (" 
                         << processedLines << "/" << totalLines << " lines, "
                         << count << " valid frames loaded)" << std::endl;
                std::cout.flush();
                lastProgressPercent = currentPercent;
            }
        } else if (processedLines % 1000 == 0) {
            std::cout << "[INFO] Loading progress: " << processedLines << " lines processed, "
                     << count << " valid frames loaded" << std::endl;
            std::cout.flush();
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
            std::lock_guard<std::mutex> lock(decLock);
            decQueue.push_back(detectMsg);
            count++;
        } else {
            std::cerr << "[WARN] Skipped malformed line (line number: " << processedLines 
                     << "), data length: " << detectMsg.data.size() << std::endl;
            skipped++;
        }
    }
    
    file.close();
    std::cout << "[INFO] Loading complete! Successfully loaded " << count << " frames of detection data (skipped " << skipped << " lines)" << std::endl;
    std::cout << "[INFO] decQueue size: " << decQueue.size() << std::endl;
    std::cout.flush();
    ROS_INFO_STREAM("Successfully loaded " << count << " frames of detection data (skipped " << skipped << " lines)");
    ROS_INFO_STREAM("decQueue size: " << decQueue.size());
}


void performSCLoopClosure() {
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";
    if (cloudKeyPoses3D->points.empty() == true)
        return;

    // find keys
    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = detectResult.first;
    float yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)
    if (loopKeyPre == -1 /* No loop found */)
        return;

    std::cout << "SC loop found! between " << loopKeyCur << " and " << loopKeyPre << "." << std::endl; // giseop

    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
        // loopFindNearKeyframesWithRespectTo(cureKeyframeCloud, loopKeyCur, 0, loopKeyPre); // giseop
        // loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);

        int base_key = 0;
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
            publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore) {
        std::cout << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << historyKeyframeFitnessScore << "). Reject this SC loop." << std::endl;
        return;
    }
    else {
        std::cout << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << historyKeyframeFitnessScore << "). Add this SC loop." << std::endl;
    }

    // publish corrected cloud
    if (pubIcpKeyFrames.getNumSubscribers() != 0) {
        pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
        publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);


    // giseop, robust kernel for a SC loop
    float robustNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6);
    robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
    gtsam::noiseModel::Base::shared_ptr robustConstraintNoise;
    robustConstraintNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(std::make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(robustConstraintNoise);
    mtx.unlock();

    // add loop constriant
    // loopIndexContainer[loopKeyCur] = loopKeyPre;
    loopIndexContainer.insert(std::pair<int, int>(loopKeyCur, loopKeyPre)); // giseop for multimap
} // performSCLoopClosure


//回环检测线程
void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
    {
        std::cout << "loopClosureEnableFlag   ==  false " << endl;
        return;
    }

    ros::Rate rate(loopClosureFrequency); //   回环频率
    while (ros::ok() && startFlag)
    {
        rate.sleep();
        performLoopClosure();   //  回环检测
        // performSCLoopClosure(); //  基于scan context的回环检测
        visualizeLoopClosure(); // rviz展示闭环边
    }
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

// 将当前帧中所有已初始化目标的优化后位姿写入文本文件
// 格式: frame_id object_id X Y Z yaw score
void saveObjectTrackingResult()
{
    if (!saveObjectResult)
        return;

    if (!objectResultStream.is_open())
        return;

    if (flow < 0 || flow >= static_cast<int>(frames.size()))
        return;

    const LidarSLAMFrame &current_frame = frames[flow];

    for (const auto &obj : current_frame.objects)
    {
        if (!obj.initialized || obj.object_id < 0)
            continue;

        objectResultStream << std::fixed << std::setprecision(6)
                           << current_frame.frame_id << " "
                           << obj.object_id << " "
                           << obj.optimize_t[3] << " "
                           << obj.optimize_t[4] << " "
                           << obj.optimize_t[5] << " "
                           << obj.optimize_t[2] << " "
                           << obj.score << std::endl;
    }

    objectResultStream.flush();
}

void ensureFrontBoxLogOpen()
{
    if (frontBoxLogStream.is_open())
        return;

    std::string log_dir = root_dir + "/Log";
    boost::filesystem::create_directories(log_dir);
    std::string log_path = log_dir + "/tracked_object_cloud.log";
    frontBoxLogStream.open(log_path, std::ios::out | std::ios::app);
}

void ensureObjectLocalMapLogOpen()
{
    if (objectLocalMapLogStream.is_open())
        return;

    std::string log_dir = root_dir + "/Log";
    boost::filesystem::create_directories(log_dir);
    std::string log_path = log_dir + "/object_local_map.log";
    objectLocalMapLogStream.open(log_path, std::ios::out | std::ios::app);
}

bool extractObjectLocalCloud(const LidarSLAMObject &obj, PointCloudXYZI::Ptr cloudLocal)
{
    if (!cloudLocal || !feats_undistort || feats_undistort->empty())
        return false;

    if (!obj.has_detect_pose || obj.obj_status != 1)
        return false;

    Eigen::Vector3f center_lidar(obj.detect_local_xyz[0], obj.detect_local_xyz[1], obj.detect_local_xyz[2]);
    float hx = 0.5f * obj.measure_lwh[0];
    float hy = 0.5f * obj.measure_lwh[1];
    float hz = 0.5f * obj.measure_lwh[2];
    float yaw_lidar = obj.detect_rotation_zyx[0];
    Eigen::Matrix3f R_lidar =
        Eigen::AngleAxisf(yaw_lidar, Eigen::Vector3f::UnitZ()).toRotationMatrix();

    cloudLocal->clear();
    cloudLocal->reserve(feats_undistort->size());

    // 将检测框内点变换到目标局部坐标系，后续累计时各帧使用同一局部表达
    for (const auto &pt : feats_undistort->points)
    {
        Eigen::Vector3f p_lidar(pt.x, pt.y, pt.z);
        Eigen::Vector3f p_local = R_lidar.transpose() * (p_lidar - center_lidar);
        if (std::fabs(p_local.x()) <= hx &&
            std::fabs(p_local.y()) <= hy &&
            std::fabs(p_local.z()) <= hz)
        {
            PointType out;
            out.x = p_local.x();
            out.y = p_local.y();
            out.z = p_local.z();
            out.intensity = pt.intensity;
            cloudLocal->push_back(out);
        }
    }

    return !cloudLocal->empty();
}

Eigen::Affine3f objectOptimizedPoseToAffine(const LidarSLAMObject &obj)
{
    std::vector<float> pose_vec = obj.optimize_t;
    return gtsamPose3toAffine3f(vec6ftoPose3(pose_vec));
}

void transformObjectCloudToAnchor(const PointCloudXYZI::Ptr &cloudLocal,
                                  const Eigen::Affine3f &current_pose_world,
                                  const Eigen::Affine3f &anchor_pose_world,
                                  PointCloudXYZI::Ptr cloudAnchor)
{
    if (!cloudAnchor)
        return;

    cloudAnchor->clear();
    if (!cloudLocal || cloudLocal->empty())
        return;

    cloudAnchor->reserve(cloudLocal->size());
    const Eigen::Affine3f T_anchor_current = anchor_pose_world.inverse() * current_pose_world;

    // 将当前目标局部坐标系下的点投到首次稳定目标坐标系，形成论文里的目标独立局部地图
    for (const auto &pt : cloudLocal->points)
    {
        Eigen::Vector3f p_anchor = T_anchor_current * Eigen::Vector3f(pt.x, pt.y, pt.z);
        PointType out = pt;
        out.x = p_anchor.x();
        out.y = p_anchor.y();
        out.z = p_anchor.z();
        cloudAnchor->push_back(out);
    }
}

void checkObjectLocalMapIcp(const LidarSLAMObject &obj,
                            const PointCloudXYZI::Ptr &cloudLocal,
                            ObjectLocalMapState &state)
{
    if (!objectLocalMapIcpCheckEnable)
        return;

    if (!cloudLocal || static_cast<int>(cloudLocal->size()) < objectLocalMapIcpCheckMinScanPoints)
        return;

    if (state.valid_frames < objectLocalMapMinValidFrames)
        return;

    const PointCloudXYZI::Ptr &target_cloud =
        (objectLocalMapIcpMap && state.icp_map_cloud && !state.icp_map_cloud->empty())
            ? state.icp_map_cloud
            : state.accumulated_cloud;

    if (!target_cloud || static_cast<int>(target_cloud->size()) < objectLocalMapIcpCheckMinMapPoints)
        return;

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setInputSource(cloudLocal);
    icp.setInputTarget(target_cloud);
    icp.setMaximumIterations(objectLocalMapIcpCheckMaxIterations);
    icp.setMaxCorrespondenceDistance(objectLocalMapIcpCheckMaxCorrespondence);
    icp.setTransformationEpsilon(1e-6);

    PointCloudXYZI::Ptr aligned(new PointCloudXYZI());
    icp.align(*aligned);

    Eigen::Matrix4f correction = icp.getFinalTransformation();
    const double dx = correction(0, 3);
    const double dy = correction(1, 3);
    const double dz = correction(2, 3);
    const double yaw = std::atan2(correction(1, 0), correction(0, 0));
    const double fitness = icp.hasConverged() ? icp.getFitnessScore() : std::numeric_limits<double>::infinity();
    const double abs_dx = std::fabs(dx);
    const double abs_dy = std::fabs(dy);
    const double abs_dz = std::fabs(dz);
    const double abs_yaw = std::fabs(yaw);

    std::string reject_reason = "ok";
    bool accepted = true;
    if (!icp.hasConverged())
    {
        accepted = false;
        reject_reason = "not_converged";
    }
    else if (fitness > objectLocalMapIcpCheckFitnessThreshold)
    {
        accepted = false;
        reject_reason = "fitness";
    }
    else if (abs_dx > objectLocalMapIcpCheckMaxAbsX ||
             abs_dy > objectLocalMapIcpCheckMaxAbsY ||
             abs_dz > objectLocalMapIcpCheckMaxAbsZ)
    {
        accepted = false;
        reject_reason = "translation";
    }
    else if (abs_yaw > objectLocalMapIcpCheckMaxAbsYaw)
    {
        accepted = false;
        reject_reason = "yaw";
    }

    ++state.icp_check_count;
    if (accepted)
    {
        state.has_backend_candidate = true;
        state.latest_backend_candidate_correction = Eigen::Affine3f::Identity();
        state.latest_backend_candidate_correction.matrix() = correction;
        state.latest_backend_candidate_frame = obj.frame_id;
        state.latest_backend_candidate_fitness = fitness;
        ++state.icp_backend_candidate_count;
    }
    else
    {
        ++state.icp_reject_count;
    }

    ensureObjectLocalMapLogOpen();
    if (objectLocalMapLogStream.is_open())
    {
        // 这里只筛选后端候选观测，不改变当前因子图；真正加因子留到后续后端改造
        objectLocalMapLogStream << "[objectLocalMapIcpCheck] frame=" << obj.frame_id
                                << ", object_id=" << obj.object_id
                                << ", converged=" << (icp.hasConverged() ? 1 : 0)
                                << ", accepted=" << (accepted ? 1 : 0)
                                << ", backend_candidate=" << (accepted ? 1 : 0)
                                << ", reject_reason=" << reject_reason
                                << ", fitness=" << fitness
                                << ", dx=" << dx
                                << ", dy=" << dy
                                << ", dz=" << dz
                                << ", yaw=" << yaw
                                << ", scan_points=" << cloudLocal->size()
                                << ", map_points=" << target_cloud->size()
                                << ", map_valid_frames=" << state.valid_frames
                                << ", anchor_frame=" << state.anchor_frame_id
                                << ", candidate_count=" << state.icp_backend_candidate_count
                                << ", reject_count=" << state.icp_reject_count
                                << std::endl;
        objectLocalMapLogStream.flush();
    }
}

void addObjectLocalMapBackendFactorIfReady(const LidarSLAMObject &obj,
                                           ObjectLocalMapState &state,
                                           const Eigen::Affine3f &current_object_pose)
{
    if (!objectLocalMapBackendFactorEnable)
        return;

    if (!state.has_backend_candidate || state.latest_backend_candidate_frame != obj.frame_id)
        return;

    if (obj.vertex_id == -1)
        return;

    const Eigen::Affine3f T_anchor_current = state.anchor_pose_world.inverse() * current_object_pose;
    const Eigen::Affine3f corrected_object_pose =
        state.anchor_pose_world * state.latest_backend_candidate_correction * T_anchor_current;

    const int factor_id = local_graph.f_id;
    priorfactor_id[flow].push_back(factor_id);
    local_graph.AddRobustPriorFactor(obj.vertex_id,
                                     Affine3f2Pose3(corrected_object_pose),
                                     objectLocalMapBackendFactorNoise,
                                     rubostNum);

    ++state.backend_factor_count;

    ensureObjectLocalMapLogOpen();
    if (objectLocalMapLogStream.is_open())
    {
        // 使用弱先验承接目标局部地图 ICP 候选观测，保留原检测观测因子作为主约束
        objectLocalMapLogStream << "[objectLocalMapBackendFactor] frame=" << obj.frame_id
                                << ", object_id=" << obj.object_id
                                << ", vertex_id=" << obj.vertex_id
                                << ", factor_id=" << factor_id
                                << ", noise=" << objectLocalMapBackendFactorNoise
                                << ", fitness=" << state.latest_backend_candidate_fitness
                                << ", backend_factor_count=" << state.backend_factor_count
                                << std::endl;
        objectLocalMapLogStream.flush();
    }
}

int addObjectLocalMapFactorsBeforeOptimization()
{
    bool feature_enabled =
        objectLocalMapEnable &&
        objectLocalMapIcpCheckEnable &&
        objectLocalMapBackendFactorEnable;

    if (!feature_enabled)
        return 0;

    if (flow < 0 || flow >= static_cast<int>(frames.size()))
        return 0;

    if (!feats_undistort || feats_undistort->empty())
        return 0;

    LidarSLAMFrame &current_frame = frames[flow];
    int added_factor_count = 0;
    for (const auto &obj : current_frame.objects)
    {
        if (!obj.initialized || obj.object_id < 0 || obj.vertex_id == -1)
            continue;

        auto it = objectLocalMaps.find(obj.object_id);
        if (it == objectLocalMaps.end() || !it->second.has_anchor_pose)
            continue;

        ObjectLocalMapState &state = it->second;
        PointCloudXYZI::Ptr cloudLocal(new PointCloudXYZI());
        if (!extractObjectLocalCloud(obj, cloudLocal))
            continue;

        Eigen::Affine3f current_object_pose = objectOptimizedPoseToAffine(obj);
        PointCloudXYZI::Ptr cloudAnchor(new PointCloudXYZI());
        transformObjectCloudToAnchor(cloudLocal, current_object_pose, state.anchor_pose_world, cloudAnchor);
        if (cloudAnchor->empty())
            continue;

        const int factor_count_before = state.backend_factor_count;
        // 在 local_graph 优化前加入目标局部地图匹配因子，使检测观测和 ICP 观测同图优化。
        checkObjectLocalMapIcp(obj, cloudAnchor, state);
        addObjectLocalMapBackendFactorIfReady(obj, state, current_object_pose);
        if (state.backend_factor_count > factor_count_before)
            ++added_factor_count;
    }

    if (added_factor_count > 0)
    {
        ensureObjectLocalMapLogOpen();
        if (objectLocalMapLogStream.is_open())
        {
            objectLocalMapLogStream << "[objectLocalMapLocalGraph] frame=" << current_frame.frame_id
                                    << ", added_factors=" << added_factor_count
                                    << std::endl;
            objectLocalMapLogStream.flush();
        }
    }

    return added_factor_count;
}

void saveTrackedObjectCloudsIfNeeded(bool force_save)
{
    bool feature_enabled =
        frontBoxTargetObjectId >= 0 &&
        frontBoxPublishMaxFrames != 0 &&
        (frontBoxAccumulateForPcd || frontBoxIcpMap);

    if (!feature_enabled)
        return;

    bool reach_limit = frontBoxPublishMaxFrames > 0 && frontBoxPublishFrameCount >= frontBoxPublishMaxFrames;
    if (!force_save && !reach_limit)
        return;

    std::string save_dir = root_dir + "/PCD";
    boost::filesystem::create_directories(save_dir);
    ensureFrontBoxLogOpen();

    if (frontBoxAccumulateForPcd && frontBoxAccumulatedCloud && !frontBoxAccumulatedCloud->empty())
    {
        std::string pcd_path = save_dir + "/tracked_object_id_" + std::to_string(frontBoxTargetObjectId) + "_accumulated.pcd";
        if (force_save || !frontBoxAccumulatedCloudSaved)
        {
            int ret = pcl::io::savePCDFileBinary(pcd_path, *frontBoxAccumulatedCloud);
            if (frontBoxLogStream.is_open())
            {
                if (ret == 0)
                {
                    frontBoxLogStream << "[trackedObjectCloud] Saved accumulated PCD: object_id=" << frontBoxTargetObjectId
                                      << ", points=" << frontBoxAccumulatedCloud->size()
                                      << ", valid_frames=" << frontBoxPublishFrameCount
                                      << ", skips_no_detection=" << frontBoxSkipNoDetection
                                      << ", skips_empty_cloud=" << frontBoxSkipEmptyCloud
                                      << " -> " << pcd_path << std::endl;
                }
                else
                {
                    frontBoxLogStream << "[trackedObjectCloud] WARN: failed to save accumulated PCD -> "
                                      << pcd_path << std::endl;
                }
                frontBoxLogStream.flush();
            }
            if (ret == 0)
                frontBoxAccumulatedCloudSaved = true;
        }
    }

    if (frontBoxIcpMap && frontBoxIcpMapCloud && !frontBoxIcpMapCloud->empty())
    {
        std::string pcd_path = save_dir + "/tracked_object_id_" + std::to_string(frontBoxTargetObjectId) + "_icp_map.pcd";
        if (force_save || !frontBoxIcpMapSaved)
        {
            int ret = pcl::io::savePCDFileBinary(pcd_path, *frontBoxIcpMapCloud);
            if (frontBoxLogStream.is_open())
            {
                if (ret == 0)
                {
                    frontBoxLogStream << "[trackedObjectCloud] Saved ICP map PCD: object_id=" << frontBoxTargetObjectId
                                      << ", points=" << frontBoxIcpMapCloud->size()
                                      << ", valid_frames=" << frontBoxPublishFrameCount
                                      << " -> " << pcd_path << std::endl;
                }
                else
                {
                    frontBoxLogStream << "[trackedObjectCloud] WARN: failed to save ICP map PCD -> "
                                      << pcd_path << std::endl;
                }
                frontBoxLogStream.flush();
            }
            if (ret == 0)
                frontBoxIcpMapSaved = true;
        }
    }
}

void accumulateTrackedObjectCloud()
{
    bool feature_enabled =
        frontBoxTargetObjectId >= 0 &&
        frontBoxPublishMaxFrames != 0 &&
        (frontBoxAccumulateForPcd || frontBoxIcpMap);

    if (!feature_enabled)
        return;

    if (frontBoxPublishMaxFrames > 0 && frontBoxPublishFrameCount >= frontBoxPublishMaxFrames)
        return;

    if (flow < 0 || flow >= static_cast<int>(frames.size()))
        return;

    if (!feats_undistort || feats_undistort->empty())
        return;

    const LidarSLAMFrame &current_frame = frames[flow];
    const LidarSLAMObject *selected = nullptr;
    for (const auto &obj : current_frame.objects)
    {
        if (obj.object_id == frontBoxTargetObjectId && obj.has_detect_pose && obj.obj_status == 1)
        {
            selected = &obj;
            break;
        }
    }

    if (selected == nullptr)
    {
        ++frontBoxSkipNoDetection;
        return;
    }

    PointCloudXYZI::Ptr cloudLocal(new PointCloudXYZI());
    if (!extractObjectLocalCloud(*selected, cloudLocal))
    {
        ++frontBoxSkipEmptyCloud;
        return;
    }

    static std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());

    if (frontBoxAccumulateForPcd)
    {
        if (!frontBoxAccumulatedCloud)
            frontBoxAccumulatedCloud.reset(new PointCloudXYZI());

        auto sample_between = [&rng](double a, double b) -> float {
            const double lo = std::min(a, b);
            const double hi = std::max(a, b);
            std::uniform_real_distribution<float> dist(static_cast<float>(lo), static_cast<float>(hi));
            return dist(rng);
        };

        const float dx = sample_between(frontBoxAccumulateRandomOffsetXMin, frontBoxAccumulateRandomOffsetXMax);
        const float dy = sample_between(frontBoxAccumulateRandomOffsetYMin, frontBoxAccumulateRandomOffsetYMax);
        const float dz = sample_between(frontBoxAccumulateRandomOffsetZMin, frontBoxAccumulateRandomOffsetZMax);
        const float yaw_offset_deg = sample_between(frontBoxAccumulateRandomYawDegMin, frontBoxAccumulateRandomYawDegMax);
        const float yaw_offset_rad = yaw_offset_deg * static_cast<float>(M_PI) / 180.0f;
        const Eigen::Matrix3f random_rot =
            Eigen::AngleAxisf(yaw_offset_rad, Eigen::Vector3f::UnitZ()).toRotationMatrix();

        for (const auto &pt : cloudLocal->points)
        {
            PointType p = pt;
            const Eigen::Vector3f rotated = random_rot * Eigen::Vector3f(pt.x, pt.y, pt.z);
            p.x = rotated.x() + dx;
            p.y = rotated.y() + dy;
            p.z = rotated.z() + dz;
            frontBoxAccumulatedCloud->push_back(p);
        }
    }

    if (frontBoxIcpMap)
    {
        if (!frontBoxIcpMapCloud)
            frontBoxIcpMapCloud.reset(new PointCloudXYZI());

        if (frontBoxIcpMapCloud->empty())
        {
            *frontBoxIcpMapCloud += *cloudLocal;
        }
        else
        {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setInputSource(cloudLocal);
            icp.setInputTarget(frontBoxIcpMapCloud);
            icp.setMaximumIterations(50);
            icp.setMaxCorrespondenceDistance(1.0);
            icp.setTransformationEpsilon(1e-6);
            PointCloudXYZI::Ptr aligned(new PointCloudXYZI());
            icp.align(*aligned);
            if (icp.hasConverged())
                *frontBoxIcpMapCloud += *aligned;
            else
                *frontBoxIcpMapCloud += *cloudLocal;
        }
    }

    int nextCount = frontBoxPublishFrameCount + 1;

    if (nextCount % 10 == 0 || (frontBoxPublishMaxFrames > 0 && nextCount >= frontBoxPublishMaxFrames))
    {
        ensureFrontBoxLogOpen();
        if (frontBoxLogStream.is_open())
        {
            frontBoxLogStream << "[trackedObjectCloud] object_id=" << frontBoxTargetObjectId
                              << ", valid_frames=" << nextCount
                              << ", target_frames=" << frontBoxPublishMaxFrames
                              << ", skips_no_detection=" << frontBoxSkipNoDetection
                              << ", skips_empty_cloud=" << frontBoxSkipEmptyCloud << std::endl;
            frontBoxLogStream.flush();
        }
    }

    // 按有效帧逐帧发布当前目标的局部点云，便于在线查看当前提取效果
    publishCloud(&pubTrackedObjectLocalCloud, cloudLocal, ros::Time().fromSec(lidar_end_time), "camera_init");

    ++frontBoxPublishFrameCount;

    saveTrackedObjectCloudsIfNeeded(false);
}

void accumulateObjectLocalMaps()
{
    bool feature_enabled =
        objectLocalMapEnable &&
        (objectLocalMapAccumulateForPcd || objectLocalMapIcpMap);

    if (!feature_enabled)
        return;

    if (flow < 0 || flow >= static_cast<int>(frames.size()))
        return;

    if (!feats_undistort || feats_undistort->empty())
        return;

    const LidarSLAMFrame &current_frame = frames[flow];
    int candidate_count = 0;
    int valid_count = 0;

    for (const auto &obj : current_frame.objects)
    {
        // 只给已经稳定初始化的跟踪目标建立局部地图，避免临时检测框污染目标点云
        if (!obj.initialized || obj.object_id < 0)
            continue;

        ++candidate_count;

        ObjectLocalMapState &state = objectLocalMaps[obj.object_id];
        PointCloudXYZI::Ptr cloudLocal(new PointCloudXYZI());
        if (!extractObjectLocalCloud(obj, cloudLocal))
        {
            ++state.skipped_empty_cloud;
            continue;
        }

        Eigen::Affine3f current_object_pose = objectOptimizedPoseToAffine(obj);
        if (!state.has_anchor_pose)
        {
            state.anchor_pose_world = current_object_pose;
            state.anchor_frame_id = current_frame.frame_id;
            state.has_anchor_pose = true;
        }

        PointCloudXYZI::Ptr cloudAnchor(new PointCloudXYZI());
        transformObjectCloudToAnchor(cloudLocal, current_object_pose, state.anchor_pose_world, cloudAnchor);
        if (cloudAnchor->empty())
        {
            ++state.skipped_empty_cloud;
            continue;
        }

        // 旧路径：优化后再计算 ICP 并补弱因子。现在因子已在 StartOptimiz 前加入同一个 local_graph，这里只累计地图。
        // checkObjectLocalMapIcp(obj, cloudAnchor, state);
        // addObjectLocalMapBackendFactorIfReady(obj, state, current_object_pose);

        if (objectLocalMapAccumulateForPcd)
            *state.accumulated_cloud += *cloudAnchor;

        if (objectLocalMapIcpMap)
        {
            if (state.icp_map_cloud->empty())
            {
                *state.icp_map_cloud += *cloudAnchor;
            }
            else
            {
                pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setInputSource(cloudAnchor);
                icp.setInputTarget(state.icp_map_cloud);
                icp.setMaximumIterations(50);
                icp.setMaxCorrespondenceDistance(1.0);
                icp.setTransformationEpsilon(1e-6);
                PointCloudXYZI::Ptr aligned(new PointCloudXYZI());
                icp.align(*aligned);
                if (icp.hasConverged())
                    *state.icp_map_cloud += *aligned;
                else
                    *state.icp_map_cloud += *cloudAnchor;
            }
        }

        ++state.valid_frames;
        ++valid_count;
    }

    if (valid_count > 0 && objectLocalMapLogInterval > 0 &&
        current_frame.frame_id % objectLocalMapLogInterval == 0)
    {
        ensureObjectLocalMapLogOpen();
        if (objectLocalMapLogStream.is_open())
        {
            objectLocalMapLogStream << "[objectLocalMap] frame=" << current_frame.frame_id
                                    << ", candidates=" << candidate_count
                                    << ", valid_updates=" << valid_count
                                    << ", tracked_maps=" << objectLocalMaps.size()
                                    << std::endl;
            objectLocalMapLogStream.flush();
        }
    }

    publishObjectLocalMaps();
}

void publishObjectLocalMaps()
{
    if (!objectLocalMapEnable || !objectLocalMapPublish)
        return;

    if (objectLocalMapPublishInterval > 0 &&
        flow >= 0 && flow < static_cast<int>(frames.size()) &&
        frames[flow].frame_id % objectLocalMapPublishInterval != 0)
        return;

    PointCloudXYZI::Ptr mergedCloud(new PointCloudXYZI());
    for (const auto &item : objectLocalMaps)
    {
        const int object_id = item.first;
        const ObjectLocalMapState &state = item.second;
        if (state.valid_frames < objectLocalMapMinValidFrames)
            continue;

        const PointCloudXYZI::Ptr &source_cloud =
            (objectLocalMapIcpMap && state.icp_map_cloud && !state.icp_map_cloud->empty())
                ? state.icp_map_cloud
                : state.accumulated_cloud;

        if (!source_cloud || source_cloud->empty())
            continue;

        // 合并发布时用 intensity 标记 object_id，RViz 可按强度颜色快速区分不同目标
        for (const auto &pt : source_cloud->points)
        {
            PointType out = pt;
            out.intensity = static_cast<float>(object_id);
            mergedCloud->push_back(out);
        }
    }

    if (mergedCloud->empty())
        return;

    publishCloud(&pubObjectLocalMaps, mergedCloud, ros::Time().fromSec(lidar_end_time), "camera_init");
}

void saveObjectLocalMapsIfNeeded(bool force_save)
{
    bool feature_enabled =
        objectLocalMapEnable &&
        (objectLocalMapAccumulateForPcd || objectLocalMapIcpMap);

    if (!feature_enabled || !force_save)
        return;

    ensureObjectLocalMapLogOpen();

    std::string save_dir = root_dir + "/PCD/object_local_maps";
    boost::filesystem::create_directories(save_dir);

    int saved_objects = 0;
    for (auto &item : objectLocalMaps)
    {
        const int object_id = item.first;
        ObjectLocalMapState &state = item.second;
        if (state.valid_frames < objectLocalMapMinValidFrames)
            continue;

        bool object_saved = false;
        if (objectLocalMapAccumulateForPcd &&
            state.accumulated_cloud &&
            !state.accumulated_cloud->empty())
        {
            std::string pcd_path = save_dir + "/object_id_" + std::to_string(object_id) + "_accumulated.pcd";
            int ret = pcl::io::savePCDFileBinary(pcd_path, *state.accumulated_cloud);
            if (objectLocalMapLogStream.is_open())
            {
                if (ret == 0)
                {
                    objectLocalMapLogStream << "[objectLocalMap] Saved accumulated PCD: object_id=" << object_id
                                            << ", points=" << state.accumulated_cloud->size()
                                            << ", valid_frames=" << state.valid_frames
                                            << ", anchor_frame=" << state.anchor_frame_id
                                            << ", skipped_empty_cloud=" << state.skipped_empty_cloud
                                            << " -> " << pcd_path << std::endl;
                }
                else
                {
                    objectLocalMapLogStream << "[objectLocalMap] WARN: failed to save accumulated PCD -> "
                                            << pcd_path << std::endl;
                }
            }
            if (ret == 0)
            {
                state.accumulated_saved = true;
                object_saved = true;
            }
        }

        if (objectLocalMapIcpMap &&
            state.icp_map_cloud &&
            !state.icp_map_cloud->empty())
        {
            std::string pcd_path = save_dir + "/object_id_" + std::to_string(object_id) + "_icp_map.pcd";
            int ret = pcl::io::savePCDFileBinary(pcd_path, *state.icp_map_cloud);
            if (objectLocalMapLogStream.is_open())
            {
                if (ret == 0)
                {
                    objectLocalMapLogStream << "[objectLocalMap] Saved ICP map PCD: object_id=" << object_id
                                            << ", points=" << state.icp_map_cloud->size()
                                            << ", valid_frames=" << state.valid_frames
                                            << ", anchor_frame=" << state.anchor_frame_id
                                            << " -> " << pcd_path << std::endl;
                }
                else
                {
                    objectLocalMapLogStream << "[objectLocalMap] WARN: failed to save ICP map PCD -> "
                                            << pcd_path << std::endl;
                }
            }
            if (ret == 0)
            {
                state.icp_saved = true;
                object_saved = true;
            }
        }

        if (object_saved)
            ++saved_objects;
    }

    if (objectLocalMapLogStream.is_open())
    {
        objectLocalMapLogStream << "[objectLocalMap] save summary: saved_objects=" << saved_objects
                                << ", total_maps=" << objectLocalMaps.size()
                                << ", min_valid_frames=" << objectLocalMapMinValidFrames
                                << std::endl;
        objectLocalMapLogStream.flush();
    }
}

inline void dump_lio_state_to_log(FILE *fp)
{
    // TUM format: timestamp x y z qx qy qz qw
    const auto &p = odomAftMapped.pose.pose.position;
    const auto &q = odomAftMapped.pose.pose.orientation;
    double t = lidar_end_time; // or lidar_end_time
    fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            t, p.x, p.y, p.z, q.x, q.y, q.z, q.w);
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

//按当前body(lidar)的状态，将局部点转换到世界系下
void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // world <-- imu <-- lidar
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)        //  lidar2world
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void test_RGBpointBodyToWorld(PointType const *const pi, PointType *const po,Eigen::Vector3d pos, Eigen::Matrix3d rotation)        //  lidar2world
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(rotation * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    for (int i = 0; i < points_history.size(); i++)
        _featsArray->push_back(points_history[i]);
}

//根据lidar的FoV分割场景
BoxPointType LocalMap_Points; // ikd-tree中,局部地图的包围盒角点
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 清空需要移除的区域
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world); // X轴分界点转换到w系下
    V3D pos_LiD = pos_lid;                               // global系lidar位置

    //初始化局部地图包围盒角点，以为w系下lidar位置为中心
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    float dist_to_map_edge[3][2]; //各个方向与局部地图边界的距离
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);

        //与某个方向上的边界距离太小，标记需要移除need_move
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }

    //不需要移除则直接返回
    if (!need_move)
        return;

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points; // 新的局部地图角点
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        //与包围盒最小值角点距离
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 移除较远包围盒
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void DetectHandler(const std_msgs::Float64MultiArray::ConstPtr& detectMsg) {
    std::lock_guard<std::mutex> lock(decLock);
    decQueue.push_back(*detectMsg);
}


void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false; // 标记是否已经进行了时间补偿
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu; //????
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());

    // 特征提取或间隔采样
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr); //储存处理后的lidar特征
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // lidar 和 imu时间差过大，且开启 时间同步, 纠正当前输入imu的时间
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        // 对输入imu时间，纠正为 时间差 + 原始时间
        msg->header.stamp =
            ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp; // update imu time

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void gnss_cbk(const sensor_msgs::NavSatFixConstPtr& msg_in)
{
    //  ROS_INFO("GNSS DATA IN ");
    if (use_gnss== false)
        return;
    double timestamp = msg_in->header.stamp.toSec();

    mtx_buffer.lock();

    // 没有进行时间纠正
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec();
    gnss_data.status = msg_in->status.status;
    gnss_data.service = msg_in->status.service;
    gnss_data.pose_cov[0] = msg_in->position_covariance[0];
    gnss_data.pose_cov[1] = msg_in->position_covariance[4];
    gnss_data.pose_cov[2] = msg_in->position_covariance[8];

    mtx_buffer.unlock();
   
    if(!gnss_inited){           //  初始化位置
        gnss_data.InitOriginPosition(msg_in->latitude, msg_in->longitude, msg_in->altitude) ;
        sensor_msgs::Imu::ConstPtr first_late_imu;
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);  // 保护 imu_buffer
            for (const auto &imu_msg : imu_buffer) {
                double t_imu = imu_msg->header.stamp.toSec();
                if (t_imu >= timestamp) {
                    first_late_imu = imu_msg;
                    break;
                }
            }
        }

        if (!first_late_imu) {
            ROS_WARN("[GNSS] no IMU with t >= first GNSS yet, wait for next GNSS");
            return;                // 这次 GNSS 先结束，不置 gnss_inited
        }

        // 找到了第一帧晚于 GNSS 的 IMU，算旋转
        const auto &q = first_late_imu->orientation;
        Eigen::Quaterniond q_imu(q.w, q.x, q.y, q.z);
        R_world_imu = q_imu.toRotationMatrix();
        Eigen::Vector3d euler_zyx = R_world_imu.eulerAngles(2, 1, 0); // yaw, pitch, roll

        ROS_INFO_STREAM("[GNSS] first IMU after GNSS t=" << first_late_imu->header.stamp.toSec()
                        << " yaw="   << euler_zyx[0]
                        << " pitch=" << euler_zyx[1]
                        << " roll="  << euler_zyx[2]); 
        // 保存第一帧的IMU旋转（位置在第一次计算gnss_pose时保存）
        first_gnss_imu_rot = q_imu;
        gnss_inited = true ;
    }else{
        gnss_data.UpdateXYZ(msg_in->latitude, msg_in->longitude, msg_in->altitude) ;             //  WGS84 -> ENU  ???  调试结果好像是 NED 北东地

        Eigen::Vector3d gnss_pose = Eigen::Vector3d::Zero() ;
        gnss_pose(0) = gnss_data.local_E ;                 //    东
        gnss_pose(1) = gnss_data.local_N ;                 //    北
        gnss_pose(2) = gnss_data.local_U ;                 //    天

        // Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
        // gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
        // gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下, （当前Gnss_T_wrt_Lidar，只是一个大致的初值）
        gnss_pose  =  R_world_imu.transpose()*gnss_pose-t_imu_gnss ;        //  东北天系 转到 世界imu 系
        // cout<<" gnss_pose  = \n"<< gnss_pose <<endl;
        nav_msgs::Odometry gnss_data_enu ;
        // add new message to buffer:
        gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
        gnss_data_enu.pose.pose.position.x =  gnss_pose(0) ;  //gnss_data.local_E ;   北
        gnss_data_enu.pose.pose.position.y =  gnss_pose(1) ;  //gnss_data.local_N;    东
        gnss_data_enu.pose.pose.position.z =  gnss_pose(2) ;  //  地

        gnss_data_enu.pose.pose.orientation.x =  geoQuat.x ;                //  gnss 的姿态不可观，所以姿态只用于可视化，取自imu
        gnss_data_enu.pose.pose.orientation.y =  geoQuat.y;
        gnss_data_enu.pose.pose.orientation.z =  geoQuat.z;
        gnss_data_enu.pose.pose.orientation.w =  geoQuat.w;

        gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
        gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
        gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

        // 保持原有 GNSS 因子逻辑: 只有有效状态的 GNSS 才进入 gnss_buffer
        if (gnss_data.status >= 0)
            gnss_buffer.push_back(gnss_data_enu);

        // visial gnss path in rviz:
        msg_gnss_pose.header.frame_id = "camera_init";
        msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);

        msg_gnss_pose.pose.position.x = gnss_pose(0) ;  
        msg_gnss_pose.pose.position.y = gnss_pose(1) ;
        msg_gnss_pose.pose.position.z = gnss_pose(2) ;
        // cout<<" gnss_data.pose.pose.position.x  = "<< gnss_data_enu.pose.pose.position.x  <<endl;

        // 获取与GNSS时间最接近的IMU姿态
        sensor_msgs::Imu::ConstPtr closest_imu;
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            double min_time_diff = std::numeric_limits<double>::max();
            for (const auto &imu_msg : imu_buffer) {
                double t_imu = imu_msg->header.stamp.toSec();
                double time_diff = std::abs(t_imu - timestamp);
                if (time_diff < min_time_diff) {
                    min_time_diff = time_diff;
                    closest_imu = imu_msg;
                }
            }
        }

        Eigen::Quaterniond current_gnss_imu_rot;
        if (closest_imu) {
            const auto &q = closest_imu->orientation;
            current_gnss_imu_rot = Eigen::Quaterniond(q.w, q.x, q.y, q.z);
        } else {
            // 如果没有找到IMU，使用geoQuat（可能不是最新的）
            current_gnss_imu_rot = Eigen::Quaterniond(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);
        }

        // gnss_pose已经是相对于初始IMU的位置，直接使用
        // 计算当前IMU旋转相对于初始IMU旋转的变换
        Eigen::Quaterniond relative_rot;
        if (closest_imu) {
            relative_rot = first_gnss_imu_rot.inverse() * current_gnss_imu_rot;
        } else {
            // 如果没有找到IMU，使用geoQuat并计算相对旋转
            Eigen::Quaterniond current_geo_rot(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);
            relative_rot = current_geo_rot;
        }

        // 以TUM格式写入txt文件：timestamp x y z qx qy qz qw
        // 位置直接使用gnss_pose（已经是相对于初始IMU的位置）
        static FILE *fp_gnss_pose_gt = nullptr;
        static bool gnss_pose_gt_file_inited = false;
        
        if (!gnss_pose_gt_file_inited) {
            string gnss_pose_gt_dir = root_dir + "/Log";
            boost::filesystem::create_directories(gnss_pose_gt_dir);
            string gnss_pose_gt_path = gnss_pose_gt_dir + "/mypose_gt.txt";
            fp_gnss_pose_gt = fopen(gnss_pose_gt_path.c_str(), "w");
            if (!fp_gnss_pose_gt) {
                ROS_ERROR("Failed to open GNSS pose GT file: %s", gnss_pose_gt_path.c_str());
            } else {
                gnss_pose_gt_file_inited = true;
                ROS_INFO("Opened GNSS pose GT file: %s", gnss_pose_gt_path.c_str());
            }
        }
        
        if (fp_gnss_pose_gt) {
            fprintf(fp_gnss_pose_gt, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    gnss_data.time,
                    gnss_pose.x(), gnss_pose.y(), gnss_pose.z(),
                    relative_rot.x(), relative_rot.y(), relative_rot.z(), relative_rot.w());
            fflush(fp_gnss_pose_gt);
        }

        gps_path.poses.push_back(msg_gnss_pose);

        //  save_gnss path
        PointTypePose thisPose6D;  
        thisPose6D.x = msg_gnss_pose.pose.position.x ;
        thisPose6D.y = msg_gnss_pose.pose.position.y ;
        thisPose6D.z = msg_gnss_pose.pose.position.z ;
        thisPose6D.intensity = 0;
        thisPose6D.roll =0;
        thisPose6D.pitch = 0;
        thisPose6D.yaw = 0;
        thisPose6D.time = lidar_end_time;
        gnss_cloudKeyPoses6D->push_back(thisPose6D);   
    }



}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();         // lidar指针指向最旧的lidar数据
        meas.lidar_beg_time = time_buffer.front(); //记录最早时间
        lidar_start_time = meas.lidar_beg_time;
        //更新结束时刻的时间
        if (meas.lidar->points.size() <= 1) // time too little 时间太短，点数不足
        {   
            lidar_start_time = meas.lidar_beg_time;
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 记录lidar结束时间为 起始时间 + 单帧扫描时间
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) //最后一个点的时间 小于 单帧扫描时间的一半
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 记录lidar结束时间为 起始时间 + 单帧扫描时间
        }
        else
        {
            scan_num++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); //结束时间设置为 起始时间 + 最后一个点的时间（相对）
            // 动态更新每帧lidar数据平均扫描时间
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // 最旧IMU时间
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) //记录imu数据，imu时间小于当前帧lidar结束时间
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if (imu_time > lidar_end_time)
            break;
        meas.imu.push_back(imu_buffer.front()); //记录当前lidar帧内的imu数据到meas.imu
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;            //需要加入到ikd-tree中的点云
    PointVector PointNoNeedDownsample; //加入ikd-tree时，不需要降采样的点云
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    //根据点与所在包围盒中心点的距离，分类是否需要降采样
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point); //当前点与box中心的距离

            //判断最近点在x、y、z三个方向上，与中心的距离，判断是否加入时需要降采样
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); //若三个方向距离都大于地图珊格半轴长，无需降采样
                continue;
            }

            //判断当前点的 NUM_MATCH_POINTS 个邻近点与包围盒中心的范围
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist) // 如果邻近点到中心的距离 小于 当前点到中心的距离，则不需要添加当前点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true); //加入点时需要降采样
    ikdtree.Add_Points(PointNoNeedDownsample, false);      //加入点时不需要降采样
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

// 使用当前帧的检测结果(所有检测框), 在 lidar 系下直接从输入点云中剔除对应区域
// 仅在 dynamic_filter_mode == 1 时调用; 为避免影响 limot 的图优化, 本函数不会修改 decQueue 的内容
void filterAllDetectionBoxesOnScan()
{
    if (!if_dynamic || dynamic_filter_mode != 1)
        return;

    if (feats_undistort == nullptr || feats_undistort->empty())
        return;

    // 当前雷达帧时间
    double timeLaserInfoCur = lidar_start_time;
    const double TIMESTAMP_TOLERANCE = 0.001;

    // 查找时间戳匹配的检测数据 (仅读取, 不 pop)
    std_msgs::Float64MultiArray matchedDec;
    bool hasMatched = false;
    {
        std::lock_guard<std::mutex> lock(decLock);
        for (size_t idx = 0; idx < decQueue.size(); ++idx)
        {
            const auto &thisDec = decQueue[idx];
            if (thisDec.data.empty())
                continue;
            double timeDiff = std::fabs(thisDec.data[0] - timeLaserInfoCur);
            if (timeDiff < TIMESTAMP_TOLERANCE)
            {
                matchedDec = thisDec;
                hasMatched = true;
                break;
            }
            // decQueue 时间有序, 且后面的时间只会更晚, 超过容差就可以提前结束
            if (thisDec.data[0] - timeLaserInfoCur > TIMESTAMP_TOLERANCE)
                break;
        }
    }

    if (!hasMatched)
        return;

    if (matchedDec.data.size() < 1)
        return;

    // 检测数据格式: [timestamp, type, x, y, z, l, w, h, yaw, score, ...]
    if ((matchedDec.data.size() - 1) % 9 != 0)
        return;

    int ob_num = (matchedDec.data.size() - 1) / 9;
    if (ob_num <= 0)
        return;

    // 以检测框为单位, 在 lidar 系下对 feats_undistort 做负向 CropBox
    box_filter.setNegative(true);

    PointCloudXYZI::Ptr cloud_current(new PointCloudXYZI);
    pcl::copyPointCloud(*feats_undistort, *cloud_current);

    for (int i = 0; i < ob_num; ++i)
    {
        double type = matchedDec.data[1 + 9 * i];
        // 这里沿用 setFrame 的筛选习惯: 只处理 type == 0 的目标
        if (type != 0.0)
            continue;

        double x = matchedDec.data[2 + 9 * i];
        double y = matchedDec.data[3 + 9 * i];
        double z = matchedDec.data[4 + 9 * i];
        double l = matchedDec.data[5 + 9 * i];
        double w = matchedDec.data[6 + 9 * i];
        double h = matchedDec.data[7 + 9 * i];
        double yaw = matchedDec.data[8 + 9 * i];

        // 略微扩展盒子, 防止边界残留
        double expand_in = 0.1;
        double expand_out = 0.1;

        box_filter.setMin(Eigen::Vector4f(static_cast<float>(-l / 2.0 - expand_in),
                                          static_cast<float>(-w / 2.0 - expand_in),
                                          static_cast<float>(-h / 2.0 - expand_in),
                                          1.0f));
        box_filter.setMax(Eigen::Vector4f(static_cast<float>(l / 2.0 + expand_out),
                                          static_cast<float>(w / 2.0 + expand_out),
                                          static_cast<float>(h / 2.0 + expand_out),
                                          1.0f));
        box_filter.setTranslation(Eigen::Vector3f(static_cast<float>(x),
                                                  static_cast<float>(y),
                                                  static_cast<float>(z)));
        box_filter.setRotation(Eigen::Vector3f(0.0f, 0.0f, static_cast<float>(yaw)));

        PointCloudXYZI::Ptr cloud_filtered(new PointCloudXYZI);
        box_filter.setInputCloud(cloud_current);
        box_filter.filter(*cloud_filtered);
        cloud_current.swap(cloud_filtered);
    }

    // 将剔除后的点云写回 feats_undistort, 后续所有流程(下采样 / 建图 / 发布)都使用剔除后的点云
    feats_undistort->clear();
    pcl::copyPointCloud(*cloud_current, *feats_undistort);
}

// 使用上一帧 tracker 预测得到的全局动态框, 在当前帧局部系下对 feats_undistort 做剔除
// 仅在 dynamic_filter_mode == 2 时调用, 并在下采样前执行, 保证后续匹配/建图链路一致
void filterPredictedBoxesOnScan()
{
    if (!if_dynamic || dynamic_filter_mode != 2)
        return;

    if (feats_undistort == nullptr || feats_undistort->empty())
        return;

    if (dynamBoxBuf.empty())
        return;

    std::vector<std::vector<double>> dynamic_boxes;
    while (!dynamBoxBuf.empty())
    {
        dynamic_boxes.emplace_back(dynamBoxBuf.front());
        dynamBoxBuf.pop();
    }

    if (dynamic_boxes.empty())
        return;

    box_filter.setNegative(true); // 保留盒子外的点

    PointCloudXYZI::Ptr cloud_current(new PointCloudXYZI);
    pcl::copyPointCloud(*feats_undistort, *cloud_current);

    Eigen::Affine3f T_ego = trans2Affine3f(transformTobeMapped); // local -> world

    for (const auto &v : dynamic_boxes)
    {
        if (v.size() < 8)
            continue;

        Eigen::Vector3f global_trans(static_cast<float>(v[0]),
                                     static_cast<float>(v[1]),
                                     static_cast<float>(v[2]));
        Eigen::Vector3f local_trans = T_ego.inverse() * global_trans;

        float global_v[6] = {0.0f, 0.0f, static_cast<float>(v[7]),
                             static_cast<float>(v[0]),
                             static_cast<float>(v[1]),
                             static_cast<float>(v[2])};
        Eigen::Affine3f global_T = trans2Affine3f(global_v);
        Eigen::Affine3f local_T = T_ego.inverse() * global_T;

        float dx, dy, dz, droll, dpitch, yaw_l;
        pcl::getTranslationAndEulerAngles(local_T, dx, dy, dz, droll, dpitch, yaw_l);

        double l = v[4];
        double w = v[5];
        double h = v[6];
        double expand_in = 0.1;
        double expand_out = 0.1;

        box_filter.setMin(Eigen::Vector4f(static_cast<float>(-l / 2.0 - expand_in),
                                          static_cast<float>(-w / 2.0 - expand_in),
                                          static_cast<float>(-h / 2.0 - expand_in),
                                          1.0f));
        box_filter.setMax(Eigen::Vector4f(static_cast<float>(l / 2.0 + expand_out),
                                          static_cast<float>(w / 2.0 + expand_out),
                                          static_cast<float>(h / 2.0 + expand_out),
                                          1.0f));
        box_filter.setTranslation(local_trans);
        box_filter.setRotation(Eigen::Vector3f(0.0f, 0.0f, yaw_l));

        PointCloudXYZI::Ptr cloud_filtered(new PointCloudXYZI);
        box_filter.setInputCloud(cloud_current);
        box_filter.filter(*cloud_filtered);
        cloud_current.swap(cloud_filtered);
    }

    feats_undistort->clear();
    pcl::copyPointCloud(*cloud_current, *feats_undistort);
}

// 在 world 坐标系下, 根据 limot 检测到的动态框剔除点云中对应区域的点
// 仅作用于传入的临时点云, 不改变内部用于建图/优化的原始点云
void filterDynamicPointsInWorld(PointCloudXYZI::Ptr &cloud_world)
{
    if (dynamic_filter_mode != 2)
        return;

    if (cloud_world == nullptr || cloud_world->empty())
        return;

    if (dynamBoxBuf.empty())
        return;

    // negative=true: 保留 box 之外的点
    box_filter.setNegative(true);

    // 逐个检测框进行剔除
    while (!dynamBoxBuf.empty())
    {
        std::vector<double> v = dynamBoxBuf.front();
        dynamBoxBuf.pop();

        // 期望布局: [x, y, z, ?, l, w, h, yaw]
        if (v.size() < 8)
            continue;

        double cx = v[0];
        double cy = v[1];
        double cz = v[2];
        double l = v[4];
        double w = v[5];
        double h = v[6];
        double yaw = v[7];

        // 略微扩展盒子, 防止边界残留
        double l1 = 0.1;
        double l2 = 0.1;

        box_filter.setMin(Eigen::Vector4f(-l / 2.0 - l1, -w / 2.0 - l1, -h / 2.0 - l1, 1.0f));
        box_filter.setMax(Eigen::Vector4f(l / 2.0 + l2, w / 2.0 + l2, h / 2.0 + l2, 1.0f));
        box_filter.setTranslation(Eigen::Vector3f(static_cast<float>(cx),
                                                  static_cast<float>(cy),
                                                  static_cast<float>(cz)));
        box_filter.setRotation(Eigen::Vector3f(0.0f, 0.0f, static_cast<float>(yaw)));

        PointCloudXYZI::Ptr cloud_filtered(new PointCloudXYZI);
        box_filter.setInputCloud(cloud_world);
        box_filter.filter(*cloud_filtered);

        cloud_world.swap(cloud_filtered);
    }
}
void publish_frame_world(const ros::Publisher &pubLaserCloudFull)         //    将稠密点云从 imu convert to  world
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        float max_intensity = std::numeric_limits<float>::lowest();
        for (int i = 0; i < size; i++)
        {
            const auto &pt_body = laserCloudFullRes->points[i];
            RGBpointBodyToWorld(&pt_body, &laserCloudWorld->points[i]);
            if (pt_body.intensity > max_intensity)
                max_intensity = pt_body.intensity;
        }

        // 打印当前帧的最大强度值
        cout << "[scan] max intensity = " << max_intensity << endl;

        // 发布到 world 坐标系（不在这里做动态剔除，统一在后面根据 dynamic_filter_mode 处理）
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(
            new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i],
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)          //   发布body系(imu)下的点云
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
        new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = state_point.rot.coeffs()[0];
    out.pose.orientation.y = state_point.rot.coeffs()[1];
    out.pose.orientation.z = state_point.rot.coeffs()[2];
    out.pose.orientation.w = state_point.rot.coeffs()[3];
}
// void append_tum_pose();
void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "body"));

    // append current optimized pose in TUM format to file
    // append_tum_pose();
}

// append current pose in TUM format: t x y z qx qy qz qw
// pose source: optimized odometry (odomAftMapped)
// void append_tum_pose()
// {
//     static std::ofstream tum_ofs;
//     static bool tum_inited = false;

//     if (!tum_inited)
//     {
//         std::string home = std::getenv("HOME") ? std::getenv("HOME") : std::string(".");
//         std::string tum_path = home + std::string("/fast_lio_sam_pose_tum.txt");
//         tum_ofs.open(tum_path, std::ios::out);
//         if (!tum_ofs.is_open())
//         {
//             ROS_ERROR("Failed to open TUM pose file: %s", tum_path.c_str());
//             return;
//         }
//         tum_inited = true;
//     }

//     const auto &p = odomAftMapped.pose.pose.position;
//     const auto &q = odomAftMapped.pose.pose.orientation;

//     // use lidar_end_time as timestamp (double seconds)
//     double t = lidar_end_time;

//     tum_ofs << std::fixed << std::setprecision(6)
//             << t << " "
//             << p.x << " " << p.y << " " << p.z << " "
//             << q.x << " " << q.y << " " << q.z << " " << q.w
//             << std::endl;
// }

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0)
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
        
        //  save  unoptimized pose
        //  V3D rot_ang(Log( state_point.rot.toRotationMatrix())); //   旋转向量
        Eigen::Vector3d rot_ang = state_point.rot.toRotationMatrix().eulerAngles(2,1,0); // yaw pitch roll
        PointTypePose thisPose6D;  
        thisPose6D.x = msg_body_pose.pose.position.x ;
        thisPose6D.y = msg_body_pose.pose.position.y ;
        thisPose6D.z = msg_body_pose.pose.position.z ;
        thisPose6D.roll = rot_ang(2) ;
        thisPose6D.pitch = rot_ang(1) ;
        thisPose6D.yaw = rot_ang(0) ;
        fastlio_unoptimized_cloudKeyPoses6D->push_back(thisPose6D);   
    }
}

void publish_path_update(const ros::Publisher pubPath)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int kkk = 0;
        kkk++;
        if (kkk % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}

//  发布gnss 轨迹
void publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(lidar_end_time);
    gps_path.header.frame_id = "camera_init";

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        pubPath.publish(gps_path);
    }
}


/*定义pose结构体*/
struct pose
{
    Eigen::Vector3d  t ;
    Eigen::Matrix3d  R;
};

bool CreateFile(std::ofstream& ofs, std::string file_path) {
    ofs.open(file_path, std::ios::out);                          //  使用std::ios::out 可实现覆盖
    if(!ofs)
    {
        std::cout << "open csv file error " << std::endl;
        return  false;
    }
    return true;
}

/* write2txt   format  KITTI*/
void WriteText(std::ofstream& ofs, pose data){
    ofs << std::fixed  <<  data.R(0,0)  << " " << data.R(0,1)   << " "<<   data.R(0,2)  << " "  <<    data.t[0]  <<  " "
                                      <<  data.R(1,0)  << " "  << data.R(1,1)  <<" " <<   data.R(1,2)   << " "  <<   data.t[1]  <<  " "
                                      <<  data.R(2,0)  << " "  << data.R(2,1)  <<" " <<   data.R(2,2)   << " "  <<   data.t[2]  <<  std::endl;

}

bool savePoseService(fast_lio_sam::save_poseRequest& req, fast_lio_sam::save_poseResponse& res)
{
    pose pose_gnss ;
    pose pose_optimized ;
    pose pose_without_optimized ;

    std::ofstream  file_pose_gnss ;
    std::ofstream  file_pose_optimized ;
    std::ofstream  file_pose_without_optimized ;

    string savePoseDirectory;
    cout << "****************************************************" << endl;
    cout << "Saving poses to pose files ..." << endl;
    if(req.destination.empty()) {
        // 检查 savePCDDirectory 是否是绝对路径
        if(savePCDDirectory.empty() || savePCDDirectory[0] == '/') {
            savePoseDirectory = savePCDDirectory;
        } else {
            savePoseDirectory = std::getenv("HOME") + savePCDDirectory;
        }
    } else {
        // 检查 req.destination 是否是绝对路径
        if(req.destination[0] == '/') {
            savePoseDirectory = req.destination;
        } else {
            savePoseDirectory = std::getenv("HOME") + req.destination;
        }
    }
    cout << "Save destination: " << savePoseDirectory << endl;
    // 创建目录（如果不存在）
    int unused = system((std::string("mkdir -p ") + savePoseDirectory).c_str());

    // create file 
    CreateFile(file_pose_gnss, savePoseDirectory + "/gnss_pose.txt");
    CreateFile(file_pose_optimized, savePoseDirectory + "/optimized_pose.txt");
    CreateFile(file_pose_without_optimized, savePoseDirectory + "/without_optimized_pose.txt");

    //  save optimize data
    for(int i = 0; i  < cloudKeyPoses6D->size(); i++){  
        pose_optimized.t =  Eigen::Vector3d(cloudKeyPoses6D->points[i].x, cloudKeyPoses6D->points[i].y, cloudKeyPoses6D->points[i].z  );
        pose_optimized.R = Exp(double(cloudKeyPoses6D->points[i].roll), double(cloudKeyPoses6D->points[i].pitch), double(cloudKeyPoses6D->points[i].yaw) );
        WriteText(file_pose_optimized, pose_optimized);
    }
    cout << "Sucess global optimized  poses to pose files ..." << endl;

    for(int i = 0; i  < fastlio_unoptimized_cloudKeyPoses6D->size(); i++){  
        pose_without_optimized.t =  Eigen::Vector3d(fastlio_unoptimized_cloudKeyPoses6D->points[i].x, fastlio_unoptimized_cloudKeyPoses6D->points[i].y, fastlio_unoptimized_cloudKeyPoses6D->points[i].z  );
        pose_without_optimized.R = Exp(double(fastlio_unoptimized_cloudKeyPoses6D->points[i].roll), double(fastlio_unoptimized_cloudKeyPoses6D->points[i].pitch), double(fastlio_unoptimized_cloudKeyPoses6D->points[i].yaw) );
        WriteText(file_pose_without_optimized, pose_without_optimized);
    }
    cout << "Sucess unoptimized  poses to pose files ..." << endl;

    for(int i = 0; i  < gnss_cloudKeyPoses6D->size(); i++){  
        pose_gnss.t =  Eigen::Vector3d(gnss_cloudKeyPoses6D->points[i].x, gnss_cloudKeyPoses6D->points[i].y, gnss_cloudKeyPoses6D->points[i].z  );
        pose_gnss.R = Exp(double(gnss_cloudKeyPoses6D->points[i].roll), double(gnss_cloudKeyPoses6D->points[i].pitch), double(gnss_cloudKeyPoses6D->points[i].yaw) );
        WriteText(file_pose_gnss, pose_gnss);
    }
    cout << "Sucess gnss  poses to pose files ..." << endl;

    file_pose_gnss.close();
    file_pose_optimized.close();
    file_pose_without_optimized.close();
    return true  ;
}

/**
 * 保存全局关键帧特征点集合
*/
bool saveMapService(fast_lio_sam::save_mapRequest& req, fast_lio_sam::save_mapResponse& res)
{
      string saveMapDirectory;
      vector<IntermediateFrameRecord> trailingIntermediateRecords;
    
      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) {
          // 检查 savePCDDirectory 是否是绝对路径
          if(savePCDDirectory.empty() || savePCDDirectory[0] == '/') {
              saveMapDirectory = savePCDDirectory;
          } else {
              saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
          }
      } else {
          // 检查 req.destination 是否是绝对路径
          if(req.destination[0] == '/') {
              saveMapDirectory = req.destination;
          } else {
              saveMapDirectory = std::getenv("HOME") + req.destination;
          }
      }
      cout << "Save destination: " << saveMapDirectory << endl;
      // 创建目录（如果不存在）
      int unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());

      // 最近关键帧之后残留的中间帧，在导图前临时挂到最后一个关键帧上，仅用于本次导图
      if (saveMapFrameMode == 1 &&
          saveMapRecordIntermediateFrames &&
          saveMapIntermediateFrameAnchorMode == 1 &&
          !pendingIntermediateFrames.empty() &&
          !cloudKeyPoses6D->empty())
      {
          int last_keyframe_index = static_cast<int>(cloudKeyPoses6D->size()) - 1;
          const PointTypePose &last_anchor_pose = keyframeAnchorCapturePoses[last_keyframe_index];
          for (const auto &pending : pendingIntermediateFrames)
          {
              Eigen::Affine3f T_w_b_anchor = pclPointToAffine3f(last_anchor_pose);
              Eigen::Affine3f T_w_b_frame = pclPointToAffine3f(pending.absolute_pose_body);
              Eigen::Affine3f T_anchor_to_frame = T_w_b_anchor.inverse() * T_w_b_frame;

              IntermediateFrameRecord record;
              record.frame_id = pending.frame_id;
              record.time = pending.time;
              record.relative_pose_to_anchor = affine3fToPointTypePose(T_anchor_to_frame, pending.time);
              record.cloud = pending.cloud;
              trailingIntermediateRecords.push_back(record);
          }
      }

      // 保存历史关键帧位姿
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);                    // 关键帧位置
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);      // 关键帧位姿
      // 提取历史关键帧角点、平面点集合
    //   pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
    //   pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

      // 注意：拼接地图时，keyframe是lidar系，而fastlio更新后的存到的cloudKeyPoses6D 关键帧位姿是body系下的，需要把
      //cloudKeyPoses6D  转换为T_world_lidar 。 T_world_lidar = T_world_body * T_body_lidar , T_body_lidar 是外参
      for (int i = 0; i < (int)cloudKeyPoses6D->size(); i++) {
            //   *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);

            if (saveMapFrameMode == 1 &&
                saveMapRecordIntermediateFrames &&
                saveMapIncludeIntermediateFramesInGlobalMap &&
                i < static_cast<int>(intermediateFramesPerKeyframe.size()))
            {
                for (const auto &intermediate : intermediateFramesPerKeyframe[i])
                {
                    Eigen::Affine3f T_w_b_anchor = pclPointToAffine3f(cloudKeyPoses6D->points[i]);
                    Eigen::Affine3f T_anchor_to_frame = pclPointToAffine3f(intermediate.relative_pose_to_anchor);
                    Eigen::Affine3f T_w_b_frame = T_w_b_anchor * T_anchor_to_frame;
                    PointTypePose frame_pose_world = affine3fToPointTypePose(T_w_b_frame, intermediate.time);
                    *globalSurfCloud += *transformPointCloud(intermediate.cloud, &frame_pose_world);
                }

                if (i == static_cast<int>(cloudKeyPoses6D->size()) - 1)
                {
                    for (const auto &intermediate : trailingIntermediateRecords)
                    {
                        Eigen::Affine3f T_w_b_anchor = pclPointToAffine3f(cloudKeyPoses6D->points[i]);
                        Eigen::Affine3f T_anchor_to_frame = pclPointToAffine3f(intermediate.relative_pose_to_anchor);
                        Eigen::Affine3f T_w_b_frame = T_w_b_anchor * T_anchor_to_frame;
                        PointTypePose frame_pose_world = affine3fToPointTypePose(T_w_b_frame, intermediate.time);
                        *globalSurfCloud += *transformPointCloud(intermediate.cloud, &frame_pose_world);
                    }
                }
            }
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if (saveMapFrameMode == 1 && saveMapRecordIntermediateFrames)
      {
          std::ofstream relativePoseOut(saveMapDirectory + "/intermediate_frame_relative_poses.txt", std::ios::out);
          if (relativePoseOut.is_open())
          {
              relativePoseOut << "# anchor_keyframe_index frame_id time rel_x rel_y rel_z rel_roll rel_pitch rel_yaw\n";
              for (int anchor_idx = 0; anchor_idx < static_cast<int>(intermediateFramesPerKeyframe.size()); ++anchor_idx)
              {
                  for (const auto &intermediate : intermediateFramesPerKeyframe[anchor_idx])
                  {
                      const auto &rel = intermediate.relative_pose_to_anchor;
                      relativePoseOut << anchor_idx << " "
                                      << intermediate.frame_id << " "
                                      << std::fixed << std::setprecision(6)
                                      << intermediate.time << " "
                                      << rel.x << " " << rel.y << " " << rel.z << " "
                                      << rel.roll << " " << rel.pitch << " " << rel.yaw << "\n";
                  }
              }
              int trailing_anchor_idx = static_cast<int>(cloudKeyPoses6D->size()) - 1;
              for (const auto &intermediate : trailingIntermediateRecords)
              {
                  const auto &rel = intermediate.relative_pose_to_anchor;
                  relativePoseOut << trailing_anchor_idx << " "
                                  << intermediate.frame_id << " "
                                  << std::fixed << std::setprecision(6)
                                  << intermediate.time << " "
                                  << rel.x << " " << rel.y << " " << rel.z << " "
                                  << rel.roll << " " << rel.pitch << " " << rel.yaw << "\n";
              }
              relativePoseOut.close();
          }
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // 降采样
        // downSizeFilterCorner.setInputCloud(globalCornerCloud);
        // downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        // downSizeFilterCorner.filter(*globalCornerCloudDS);
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // 降采样
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        //   downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
         downSizeFilterSurf.setInputCloud(globalSurfCloud);
         downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
         downSizeFilterSurf.filter(*globalSurfCloudDS);
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);       
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);           //  稠密点云地图
      }

      // 保存到一起，全局关键帧特征点集合
    //   *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;
      pcl::io::savePCDFileBinary(saveMapDirectory + "/filterGlobalMap.pcd", *globalSurfCloudDS);       //  滤波后地图
      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);       //  稠密地图
      res.success = ret == 0;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      // visial optimize global map on viz
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    string odometryFrame = "camera_init";
    publishCloud(&pubOptimizedGlobalMap, globalSurfCloudDS, timeLaserInfoStamp, odometryFrame);

      return true;
}


void saveMap()
{
    fast_lio_sam::save_mapRequest  req;
    fast_lio_sam::save_mapResponse res;
    // 保存全局关键帧特征点集合
    if(!saveMapService(req, res)){
        cout << "Fail to save map" << endl;
    }
}

/**
 * 发布局部关键帧map的特征点云
 */
void publishGlobalMap()
{
    /*** if path is too large, the rvis will crash ***/
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);
    string odometryFrame = "camera_init";
    if (pubLaserCloudSurround.getNumSubscribers() == 0)
        return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
    ;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kdtree查找最近一帧关键帧相邻的关键帧集合
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    // 提取局部相邻关键帧对应的特征点云
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
    {
        // 距离过大
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        // *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
        *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]); //  fast_lio only use  surfCloud
    }
    // 降采样，发布
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
}

//构造H矩阵
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++) //判断每个点的对应邻域是否符合平面点的假设
    {
        PointType &point_body = feats_down_body->points[i];   // lidar系下坐标
        PointType &point_world = feats_down_world->points[i]; // lidar数据点在world系下坐标

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);                     // lidar系下坐标
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // w系下坐标
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)        //  如果收敛了
        {
            /** Find the closest surfaces in the map **/
            // world系下从ikdtree找5个最近点用于平面拟合
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            //最近点数大于NUM_MATCH_POINTS，且最大距离小于等于5,point_selected_surf设置为true
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                : true;
        }

        //不符合平面特征
        if (!point_selected_surf[i])
            continue;

        VF(4)  pabcd;          //  plane 参数  a b c d
        point_selected_surf[i] = false; //二次筛选平面点
        //拟合局部平面，返回：是否有内点大于距离阈值
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            // plane distance
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); //筛选条件 1 - 0.9 * （点到平面距离 / 点到lidar原点距离）

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; //以intensity记录点到面残差
                res_last[i] = abs(pd2);             // 残差，距离
            }
        }
    }

    effct_feat_num = 0; //有效匹配点数

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; // body系 平面特征点
            corr_normvect->points[effct_feat_num] = normvec->points[i];         // world系 平面参数
            total_residual += res_last[i];                                      // 残差和
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num; // 残差均值 （距离）
    match_time += omp_get_wtime() - match_start;
    double solve_start_ = omp_get_wtime();

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //定义H维度
    ekfom_data.h.resize(effct_feat_num);                 //有效方程个数

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i]; // lidar系 平面特征点
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // 当前状态imu系下 点坐标
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this); // 当前状态imu系下 点坐标反对称矩阵

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z); //对应局部法相量, world系下

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec); // 将对应局部法相量旋转到imu系下 corr_normal_I
        V3D A(point_crossmat * C);           //残差对角度求导系数 P(IMU)^ [R(imu <-- w) * normal_w]
        //添加数据到矩阵
        if (extrinsic_est_en)
        {
            // B = lidar_p^ R(L <-- I) * corr_normal_I
            // B = lidar_p^ R(L <-- I) * R(I <-- W) * normal_W
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char **argv)
{
    // allocateMemory();
    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }
    if (if_dynamic) {
        // cout << "get objects! and size is: " << frames.size() << endl;
        box_filter.setNegative(true);
    }
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<string>("map_file_path", map_file_path, "");
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
    nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
    nh.param<double>("cube_side_length", cube_len, 200);
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);
    nh.param<double>("mapping/fov_degree", fov_deg, 180);
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, 2); // 0-second, 1-milisecond, 2-microsecond, 3-nanosecond
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    nh.param<float>("odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
    nh.param<float>("mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
    nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

    nh.param<float>("z_tollerance", z_tollerance, FLT_MAX);
    nh.param<float>("rotation_tollerance", rotation_tollerance, FLT_MAX);

    nh.param<int>("numberOfCores", numberOfCores, 2);
    nh.param<double>("mappingProcessInterval", mappingProcessInterval, 0.15);

    // save keyframes
    nh.param<bool>("dense_keyframe", dense_keyframe, true);  // true: 关键帧存储feats_undistort, false: 存储feats_down_body
    nh.param<int>("frontend_odom_mode", frontend_odom_mode, 0);
    nh.param<std::string>("saulio_odom_topic", saulio_odom_topic, std::string("/aft_mapped_to_init"));
    nh.param<std::string>("saulio_filtered_cloud_topic", saulio_filtered_cloud_topic, std::string("/sat_slam/saulio_filtered_points"));
    nh.param<double>("saulio_odom_time_tolerance", saulio_odom_time_tolerance, 0.05);
    nh.param<double>("saulio_odom_wait_timeout", saulio_odom_wait_timeout, 0.08);
    nh.param<bool>("saulio_publish_filtered_cloud", saulio_publish_filtered_cloud, true);
    nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 20.0);
    nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
    nh.param<float>("surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
    nh.param<float>("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

    // loop clousre
    nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, false);
    nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);
    nh.param<int>("surroundingKeyframeSize", surroundingKeyframeSize, 50);
    nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
    nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
    nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
    nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);
    nh.param<float>("loopClosureNoiseScore", loopClosureNoiseScore, 0.5);  // 闭环噪声值，默认0.5

    // gnss
    nh.param<bool>("use_gnss", use_gnss, false);
    nh.param<string>("common/gnss_topic", gnss_topic,"/gps/fix");
    nh.param<vector<double>>("mapping/extrinR_Gnss2Lidar", extrinR_Gnss2Lidar, vector<double>());
    nh.param<vector<double>>("mapping/extrinT_Gnss2Lidar", extrinT_Gnss2Lidar, vector<double>());
    nh.param<vector<double>>("mapping/t_imu_gnss", t_imu_gnss_vec, vector<double>());
    nh.param<bool>("useImuHeadingInitialization", useImuHeadingInitialization, false);
    nh.param<bool>("useGpsElevation", useGpsElevation, false);
    nh.param<float>("gpsCovThreshold", gpsCovThreshold, 2.0);
    nh.param<float>("gpsNoiseMin", gpsNoiseMin, 1.0);
    nh.param<float>("poseCovThreshold", poseCovThreshold, 25.0);


    // Visualization
    nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
    nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

    // visual ikdtree map
    nh.param<bool>("visulize_IkdtreeMap", visulize_IkdtreeMap, false);

    // reconstruct ikdtree 
    nh.param<bool>("reconstructKdTree", reconstructKdTree, false);
    nh.param<bool>("if_reconstruct_ikdtree_map_add_gps", if_reconstruct_ikdtree_map_add_gps, false);
    nh.param<bool>("if_reconstruct_ikdtree_map_loop_closure", if_reconstruct_ikdtree_map_loop_closure, false);
    nh.param<int>("reconstruct_ikdtree_mode_add_gps", reconstruct_ikdtree_mode_add_gps, 1);
    nh.param<int>("reconstruct_ikdtree_mode_loop_closure", reconstruct_ikdtree_mode_loop_closure, 0);
    nh.param<int>("reconstruct_ikdtree_frame", reconstruct_ikdtree_frame, 5);
    nh.param<float>("reconstruct_ikdtree_search_radius", reconstruct_ikdtree_search_radius, 100.0);
    nh.param<float>("reconstruct_pose_density", reconstruct_pose_density, 10.0);
    nh.param<float>("reconstruct_map_leafsize", reconstruct_map_leafsize, 0.5);
    nh.param<int>("add_gps_num_reconstruct", add_gps_num_reconstruct, 3);

        // limot
    nh.param<float>("limot/Scorethre", Scorethre, 0.5);
    nh.param<int>("limot/detection_wait_ms", detection_wait_ms, 60);
    nh.param<bool>("limot/association_descriptor_gate_enable", associationDescriptorGateEnable, false);
    nh.param<double>("limot/association_descriptor_cos_threshold", associationDescriptorCosThreshold, 0.55);
    nh.param<int>("limot/association_descriptor_grid_x", associationDescriptorGridX, 4);
    nh.param<int>("limot/association_descriptor_grid_y", associationDescriptorGridY, 4);
    nh.param<int>("limot/association_descriptor_min_points_per_cell", associationDescriptorMinPointsPerCell, 2);
    nh.param<int>("limot/association_descriptor_min_valid_cells", associationDescriptorMinValidCells, 3);
    nh.param<int>("limot/laserCloudInfoHandler_size", laserCloudInfoHandler_size, 10);
    nh.param<bool>("limot/if_priorfactor", if_priorfactor, true);
    nh.param<int>("limot/window_size", window_size, 10);
    nh.param<double>("limot/egoP_egoP", egoP_egoP, 1e-10);
    nh.param<double>("limot/egoP_objP", egoP_objP, 1e-6);
    nh.param<double>("limot/ego_stationObjP", ego_stationObjP, 1e-2);
    nh.param<double>("limot/objP_objP_chgP", objP_objP_chgP, 1e-4);
    nh.param<double>("limot/chgP_chgP", chgP_chgP, 1e-8);
    nh.param<double>("limot/rubostNum", rubostNum, 1.0);
    nh.param<bool>("limot/if_dynamic", if_dynamic, false);
    nh.param<bool>("limot/publishOffsetDetect", publishOffsetDetect, false);
    nh.param<int>("limot/publishOffsetDetectSourceId", publishOffsetDetectSourceId, 1);
    nh.param<int>("limot/publishOffsetDetectOutputId", publishOffsetDetectOutputId, 2);
    nh.param<double>("limot/publishOffsetDetectShiftX", publishOffsetDetectShiftX, -0.4);
    nh.param<double>("limot/publishOffsetDetectShiftY", publishOffsetDetectShiftY, -0.5);
    nh.param<double>("limot/publishOffsetDetectYawDeg", publishOffsetDetectYawDeg, 2.0);
    nh.param<int>("limot/dynamic_filter_mode", dynamic_filter_mode, 0);
    nh.param<bool>("limot/saveObjectResult", saveObjectResult, false);
    nh.param<int>("limot/frontBoxTargetObjectId", frontBoxTargetObjectId, -1);
    nh.param<int>("limot/frontBoxPublishMaxFrames", frontBoxPublishMaxFrames, 0);
    nh.param<bool>("limot/frontBoxAccumulateForPcd", frontBoxAccumulateForPcd, false);
    nh.param<double>("limot/frontBoxAccumulateRandomOffsetXMin", frontBoxAccumulateRandomOffsetXMin, -0.2);
    nh.param<double>("limot/frontBoxAccumulateRandomOffsetXMax", frontBoxAccumulateRandomOffsetXMax, 0.2);
    nh.param<double>("limot/frontBoxAccumulateRandomOffsetYMin", frontBoxAccumulateRandomOffsetYMin, -0.3);
    nh.param<double>("limot/frontBoxAccumulateRandomOffsetYMax", frontBoxAccumulateRandomOffsetYMax, 0.3);
    nh.param<double>("limot/frontBoxAccumulateRandomOffsetZMin", frontBoxAccumulateRandomOffsetZMin, -0.3);
    nh.param<double>("limot/frontBoxAccumulateRandomOffsetZMax", frontBoxAccumulateRandomOffsetZMax, 0.3);
    nh.param<double>("limot/frontBoxAccumulateRandomYawDegMin", frontBoxAccumulateRandomYawDegMin, -10.0);
    nh.param<double>("limot/frontBoxAccumulateRandomYawDegMax", frontBoxAccumulateRandomYawDegMax, 10.0);
    nh.param<bool>("limot/frontBoxIcpMap", frontBoxIcpMap, false);
    nh.param<bool>("limot/objectLocalMapEnable", objectLocalMapEnable, false);
    nh.param<bool>("limot/objectLocalMapAccumulateForPcd", objectLocalMapAccumulateForPcd, true);
    nh.param<bool>("limot/objectLocalMapIcpMap", objectLocalMapIcpMap, false);
    nh.param<int>("limot/objectLocalMapMinValidFrames", objectLocalMapMinValidFrames, 5);
    nh.param<int>("limot/objectLocalMapLogInterval", objectLocalMapLogInterval, 20);
    nh.param<bool>("limot/objectLocalMapPublish", objectLocalMapPublish, false);
    nh.param<int>("limot/objectLocalMapPublishInterval", objectLocalMapPublishInterval, 5);
    nh.param<bool>("limot/objectLocalMapIcpCheckEnable", objectLocalMapIcpCheckEnable, false);
    nh.param<int>("limot/objectLocalMapIcpCheckMinMapPoints", objectLocalMapIcpCheckMinMapPoints, 300);
    nh.param<int>("limot/objectLocalMapIcpCheckMinScanPoints", objectLocalMapIcpCheckMinScanPoints, 50);
    nh.param<int>("limot/objectLocalMapIcpCheckMaxIterations", objectLocalMapIcpCheckMaxIterations, 30);
    nh.param<double>("limot/objectLocalMapIcpCheckMaxCorrespondence", objectLocalMapIcpCheckMaxCorrespondence, 0.8);
    nh.param<double>("limot/objectLocalMapIcpCheckFitnessThreshold", objectLocalMapIcpCheckFitnessThreshold, 0.08);
    nh.param<double>("limot/objectLocalMapIcpCheckMaxAbsX", objectLocalMapIcpCheckMaxAbsX, 1.0);
    nh.param<double>("limot/objectLocalMapIcpCheckMaxAbsY", objectLocalMapIcpCheckMaxAbsY, 1.0);
    nh.param<double>("limot/objectLocalMapIcpCheckMaxAbsZ", objectLocalMapIcpCheckMaxAbsZ, 1.0);
    nh.param<double>("limot/objectLocalMapIcpCheckMaxAbsYaw", objectLocalMapIcpCheckMaxAbsYaw, 0.2);
    nh.param<bool>("limot/objectLocalMapBackendFactorEnable", objectLocalMapBackendFactorEnable, false);
    nh.param<double>("limot/objectLocalMapBackendFactorNoise", objectLocalMapBackendFactorNoise, 1.0);
    nh.param<int>("limot/backendResultMode", backendResultMode, 1);
    nh.param<bool>("limot/localGraphFeedbackToGlobal", localGraphFeedbackToGlobal, false);
    nh.param<double>("limot/localGraphFeedbackNoise", localGraphFeedbackNoise, 10.0);
    nh.param<int>("limot/localGraphFeedbackMinObjectFactors", localGraphFeedbackMinObjectFactors, 1);
    nh.param<double>("limot/localGraphFeedbackMaxTranslation", localGraphFeedbackMaxTranslation, 1.0);
    nh.param<double>("limot/localGraphFeedbackMaxRotation", localGraphFeedbackMaxRotation, 0.3);
    // nh.param<bool>("limot/pubtrackedobjects", pubtrackedobjects, false);
    nh.param<std::string>("limot/sequence", sequence, "09");
    nh.param<float>("limot/vel_threshold", vel_threshold, 1.0);

    // savMap
    nh.param<bool>("savePCD", savePCD, false);
    nh.param<std::string>("savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");
    nh.param<int>("mapSaveInterval", mapSaveInterval, -1);  // -1: 只在结束时保存, 1: 每帧保存, N: 每N帧保存
    nh.param<int>("saveMapFrameMode", saveMapFrameMode, 0);
    nh.param<bool>("saveMapRecordIntermediateFrames", saveMapRecordIntermediateFrames, false);
    nh.param<int>("saveMapIntermediateFrameAnchorMode", saveMapIntermediateFrameAnchorMode, 0);
    nh.param<bool>("saveMapIncludeIntermediateFramesInGlobalMap", saveMapIncludeIntermediateFramesInGlobalMap, false);

    downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
    // downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

    // ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf)); //重复？
    memset(res_last, -1000.0f, sizeof(res_last));

    //设置imu和lidar外参和imu参数等
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov)); // 加速度协方差
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    //设置gnss外参数
    Gnss_T_wrt_Lidar<<VEC_FROM_ARRAY(extrinT_Gnss2Lidar);
    Gnss_R_wrt_Lidar<<MAT_FROM_ARRAY(extrinR_Gnss2Lidar);
    t_imu_gnss << VEC_FROM_ARRAY(t_imu_gnss_vec);

    double epsi[23] = {0.001};
    fill(epsi, epsi + 23, 0.001);
    ///初始化，其中h_share_model定义了·平面搜索和残差计算
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log";
    boost::filesystem::create_directories(pos_log_dir);
    pos_log_dir += "/pos.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    // 目标跟踪结果输出文件（每帧优化后的目标位姿）
    std::string object_result_path = root_dir + "/Log/object_tracking.txt";
    if (saveObjectResult)
    {
        objectResultStream.open(object_result_path, std::ios::out);
        if (!objectResultStream.is_open())
        {
            ROS_WARN("Failed to open object tracking result file: %s", object_result_path.c_str());
        }
    }

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_saulio_odom;
    if (frontend_odom_mode == 1)
    {
        sub_saulio_odom = nh.subscribe<nav_msgs::Odometry>(saulio_odom_topic, 2000, saulioOdomHandler, ros::TransportHints().tcpNoDelay());
        ROS_INFO_STREAM("[saulio_frontend] use external SAULIO odometry topic: " << saulio_odom_topic);
    }
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);        //  world系下稠密点云
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);      //  body系下稠密点云
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);         //  no used
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);                    //  no used
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 1e00000);
    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("fast_lio_sam/path_update", 100000);                   //  isam更新后的path
    pubTrackedObjects = nh.advertise<visualization_msgs::MarkerArray>("/tracked_objects", 100);
    pubExternalTrackedObject = nh.advertise<visualization_msgs::MarkerArray>("external_tracked_object", 100);
    pubObjectTrajectories = nh.advertise<visualization_msgs::MarkerArray>("object_trajectories", 100);
    pubTrackedObjectLocalCloud = nh.advertise<sensor_msgs::PointCloud2>("/tracked_object_local_cloud", 10);
    pubObjectLocalMaps = nh.advertise<sensor_msgs::PointCloud2>("/limot/object_local_maps", 1);
    pubSaulioFilteredCloud = nh.advertise<sensor_msgs::PointCloud2>(saulio_filtered_cloud_topic, 20);
    pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);
    pubGnssPoseGT = nh.advertise<std_msgs::String>("/mypose_gt", 1000); // 发布GNSS IMU位姿真值（TUM格式）
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/keyframe_submap", 1); // 发布局部关键帧map的特征点云
    pubOptimizedGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/map_global_optimized", 1); // 发布局部关键帧map的特征点云

    // 检测数据源：优先从文件加载，如果没有文件则订阅topic
    // 只有在if_dynamic为true时才订阅/加载检测数据
    if (if_dynamic) {
        std::string detect_file_path;
        if (nh.getParam("detect_file_path", detect_file_path) && !detect_file_path.empty()) {
            ROS_INFO_STREAM("========== Initializing detection file loading at startup ==========");
            ROS_INFO_STREAM("Using detection file mode, path: " << detect_file_path);
            std::cout.flush(); // 立即刷新输出
            loadDetectionsFromFile(detect_file_path);
        } else {
            ROS_INFO_STREAM("Using topic subscription mode: /detect3d");
            subDetect = nh.subscribe<std_msgs::Float64MultiArray>("/detect3d", 200, DetectHandler, ros::TransportHints().tcpNoDelay());
        }
    } else {
        ROS_INFO_STREAM("if_dynamic is false, skipping detection data subscription/loading");
    }

    // loop clousre
    // 发布闭环匹配关键帧局部map
    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_history_cloud", 1);
    // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("fast_lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
    // 发布闭环边，rviz中表现为闭环帧之间的连线
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);

    // gnss
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk);
    
    // saveMap  发布地图保存服务
    srvSaveMap  = nh.advertiseService("/save_map" ,  &saveMapService);

    // savePose  发布轨迹保存服务
    srvSavePose  = nh.advertiseService("/save_pose" ,  &savePoseService);

    // 回环检测线程
    std::thread loopthread(&loopClosureThread);

    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();

        /// 在Measure内，储存当前lidar数据及lidar扫描时间内对应的imu数据序列
        if (sync_packages(Measures))
        {
            //第一帧lidar数据
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time; //记录第一帧绝对时间
                p_imu->first_lidar_time = first_lidar_time; //记录第一帧绝对时间
                flg_first_scan = false;
                continue;
            }

            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time = 0;
            t0 = omp_get_wtime();

            //根据imu数据序列和lidar数据，向前传播纠正点云的畸变, 此前已经完成间隔采样或特征提取
            // feats_undistort 为畸变纠正之后的点云,lidar系
            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();                                               // 前向传播后body的状态预测值
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // global系 lidar位置

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            // 用当前预测位姿先更新一次 transformTobeMapped, 供下采样前的动态点剔除使用
            getCurPose(state_point);

            // 模式 1: 用当前帧检测框剔除; 模式 2: 用上一帧 tracker 预测框剔除
            // 两者都在下采样前执行, 保证后续匹配 / 建图 / 发布使用一致的点云
            filterAllDetectionBoxesOnScan();
            filterPredictedBoxesOnScan();
            publishFilteredCloudForSaulio();

            // 检查当前lidar数据时间，与最早lidar数据时间是否足够
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment(); // 根据lidar在W系下的位置，重新确定局部地图的包围盒角点，移除远端的点

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size(); //当前帧降采样后点数

            if (frontend_odom_mode == 1)
            {
                nav_msgs::Odometry saulio_odom;
                if (!waitSaulioOdom(lidar_end_time, saulio_odom))
                {
                    ROS_WARN_STREAM_THROTTLE(1.0, "[saulio_frontend] no matched odometry near lidar_end_time="
                                                  << std::fixed << std::setprecision(6) << lidar_end_time
                                                  << ", tolerance=" << saulio_odom_time_tolerance
                                                  << ", wait_timeout=" << saulio_odom_wait_timeout);
                    continue;
                }

                // SAULIO 输出作为当前帧前端里程计，后续动态目标跟踪和后端仍复用原 sat-slam 流程。
                applySaulioOdomToEskfState(saulio_odom);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                euler_cur = SO3ToEuler(state_point.rot);
            }

            /*** initialize the map kdtree ***/
            if (ikdtree.Root_Node == nullptr)
            {
                if (feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // point转到world系下
                    }
                    // world系下对当前帧降采样后的点云，初始化lkd-tree
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();

            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            // lidar --> imu
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                     << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << endl;

            if (visulize_IkdtreeMap) // If you need to see map point, change to "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
                publish_map(pubLaserCloudMap);
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            if (frontend_odom_mode == 1)
            {
                solve_H_time = 0.0;
            }
            else
            {
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); //预测、更新
                state_point = kf.get_x();
                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // world系下lidar坐标
                geoQuat.x = state_point.rot.coeffs()[0];                                // world系下当前imu的姿态四元数
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];
            }

            double t_update_end = omp_get_wtime();

            getCurPose(state_point); //   更新transformTobeMapped

            /*back end*/
            // 1.计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            // 2.添加激光里程计因子、GPS因子、闭环因子
            // 3.执行因子图优化
            // 4.得到当前帧优化后的位姿，位姿协方差
            // 5.添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            saveKeyFramesAndFactor();
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹， 重构ikdtree
            correctPoses();
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            visualizeTrackedObjects();
            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();

            // 打印当前帧各阶段耗时（单位：ms）
            double imu_propagate_time_ms   = (t1 - t0) * 1000.0;                 // IMU+去畸变+下采样前
            double t2_t1        = (t2-t1) * 1000.0;
            double ekf_update_time_ms      = (t_update_end - t2) * 1000.0; // 前端匹配+ESKF/迭代更新
            double backend_opt_time_ms     = (t3 - t_update_end) * 1000.0;                 // saveKeyFramesAndFactor + correctPoses + 发布里程计
            double kdtree_incremental_ms   = (t5 - t3) * 1000.0;                 // 点云加入 ikdtree
            double total_frame_time_ms     = (t5 - t0) * 1000.0;                 // 整帧总时间
            total_mot_graph_time_ms += current_mot_graph_time_ms;
            total_global_graph_time_ms += current_global_graph_time_ms;
            total_backend_graph_time_ms += current_backend_graph_time_ms;
            backend_graph_frame_count++;
            double avg_mot_tracking_time_ms = mot_tracking_frame_count > 0 ? (total_mot_tracking_time_ms / mot_tracking_frame_count) : 0.0;
            double avg_mot_graph_time_ms = backend_graph_frame_count > 0 ? (total_mot_graph_time_ms / backend_graph_frame_count) : 0.0;
            double avg_global_graph_time_ms = backend_graph_frame_count > 0 ? (total_global_graph_time_ms / backend_graph_frame_count) : 0.0;
            double avg_backend_graph_time_ms = backend_graph_frame_count > 0 ? (total_backend_graph_time_ms / backend_graph_frame_count) : 0.0;
            /******* Publish points *******/
            if (path_en){
                publish_path(pubPath);
                publish_gnss_path(pubGnssPath);                        //   发布gnss轨迹
                publish_path_update(pubPathUpdate);             //   发布经过isam2优化后的路径
                static int jjj = 0;
                jjj++;
                if (jjj % 100 == 0)
                {
                    publishGlobalMap();             //  发布局部点云特征地图
                }
            }
            if (scan_pub_en || pcd_save_en)
            {
                // 模式 0 / 1: 直接使用原始发布逻辑 (模式 1 在前面已经对 feats_undistort 做过剔除)
                // if (dynamic_filter_mode != 2 || dynamBoxBuf.empty())
                // {
                    publish_frame_world(pubLaserCloudFull);        //   发布world系下的点云
                // }
                // else
                // {
                //     // 模式 2: 在 world 系下用动态框剔除后再发布, 不影响内部建图
                //     PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
                //     int size = laserCloudFullRes->points.size();
                //     PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

                //     for (int i = 0; i < size; i++)
                //     {
                //         RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                //                             &laserCloudWorld->points[i]);
                //     }

                //     filterDynamicPointsInWorld(laserCloudWorld);

                //     sensor_msgs::PointCloud2 laserCloudmsg;
                //     pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
                //     laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
                //     laserCloudmsg.header.frame_id = "camera_init";
                //     pubLaserCloudFull.publish(laserCloudmsg);
                // }
            }
            if (scan_pub_en && scan_body_pub_en)
                publish_frame_body(pubLaserCloudFull_body);         //  发布imu系下的点云

            // 按间隔保存地图（如果启用）
            if (savePCD && mapSaveInterval > 0 && frame_num % mapSaveInterval == 0)
            {
                saveMap();
            }

            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f ave backend_graph(ms): %0.6f ave mot_graph(ms): %0.6f ave global_graph(ms): %0.6f ave mot_track(ms): %0.6f \n",
                       t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time,
                       avg_backend_graph_time_ms, avg_mot_graph_time_ms, avg_global_graph_time_ms, avg_mot_tracking_time_ms);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose() << " " << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " << state_point.vel.transpose()
                         << " " << state_point.bg.transpose() << " " << state_point.ba.transpose() << " " << state_point.grav << " " << feats_undistort->points.size() << endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    
    // 保存完整地图（如果启用，且未使用间隔保存或间隔保存未触发）
    if (savePCD && (mapSaveInterval <= 0 || frame_num % mapSaveInterval != 0))
    {
        cout << "Saving complete map at program exit..." << endl;
        saveMap();
    }

    saveTrackedObjectCloudsIfNeeded(true);
    saveObjectLocalMapsIfNeeded(true);

    fout_out.close();
    fout_pre.close();
    if (frontBoxLogStream.is_open())
        frontBoxLogStream.close();
    if (objectLocalMapLogStream.is_open())
        objectLocalMapLogStream.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0; i < time_log_counter; i++)
        {
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i], int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    startFlag = false;
    loopthread.join(); //  分离线程

    return 0;
}
