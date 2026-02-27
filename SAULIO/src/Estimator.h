#ifndef Estimator_H
#define Estimator_H
#include "map_manager.h"
#include "common_lib.h"
#include "parameters.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
// #include <ikd-Tree/ikd_Tree.h>
#include <pcl/io/pcd_io.h>
#include <unordered_set>
extern FILE *fp_debug_1;
extern bool a;
extern PointCloudXYZI::Ptr normvec; //(new PointCloudXYZI(100000, 1));
extern std::vector<int> time_seq;
extern std::vector<int> time_seq1;
extern PointCloudXYZI::Ptr feats_down_body; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_undistort; // 去畸变后的点云
extern PointCloudXYZI::Ptr feats_down_world_ground; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world_beam; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world_roof; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world_facade; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world_pillar; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_world_vertex; //(new PointCloudXYZI());

//bigupdate里面的HTH等参数


//lyb

extern int num_100;
extern double match_time;
extern double solve_time_1;
extern double solve_time_2;
extern double solve_time_3;
// 每帧中四个观测模型函数各自的 match 时间统计（同一帧内多次调用会累加）
extern double match_time_h_model_input;
extern double match_time_h_model_input_1;
extern double match_time_h_model_input_bigupdate;
extern double match_time_h_model_input_bigupdate_1;
extern std::vector<PointVector> Nearest_oneset_Points;
extern int effect_num_oneframe;
extern std::vector<M3D> crossmat_bigupdate_imu_list;
extern std::vector<M3D> crossmat_bigupdate_body_list;
extern std::vector<M3D> crossmat_effect_list;
extern std::vector<M3D> crossmat_oneset_list;
extern FILE *fp_debug;
extern FILE *ratio_debug;
extern int bigupdate_num;

extern int bigupdate_effect_num;
extern int bigupdate_effect_num;
extern PointCloudXYZI::Ptr feats_down_incre_world;
extern PointCloudXYZI::Ptr feats_down_oneset_body; //(new PointCloudXYZI());
extern PointCloudXYZI::Ptr feats_down_oneset_world;
extern PointCloudXYZI::Ptr feats_down_bigupdate_world;
extern PointCloudXYZI::Ptr feats_down_bigupdate_body;
extern std::vector<int> oneset_pointnumbers;
extern std::vector<int>bigupdate_pointnumbers;
// extern std::vector<int> point_effect_numbers;
// extern std::vector<V3D> pworld_oneset_list;
extern int total_incre_idx;
extern int total_incre_num; 

extern int total_oneset_num;//是指上一个set结束到下一个set结束的所有点数，也就是相当于一次更新的feats_down_body的数量
extern int effect_oneset_num;//是指一个set的有效点数
extern int oneset_idx;//指这个set所有点云的序号
extern int oneset_startupdate_idx;//因为在一个set中包含有效点和无效点，都要放进feats_down_oneset_body，所以需要记录每个时刻的idx
extern int oneset_num;//是指在开始有效之后到这个set结束的所有点数 
extern bool if_effect_start;
extern double publish_time;
extern double oneset_time;
extern double last_small_check_time; // 上次小更新的时间，用于每0.02秒检查一次特征值
extern double bigupdate_start_time; // 大更新开始的时间，用于判断是否达到0.1s
extern bool is_last_segment_in_frame; // 标识是否是本帧最后一个segment

extern std::vector<V3D> pbody_list;
extern std::vector<V3D> pbody_oneset_list;
extern std::vector<V3D> pbody_bigupdate_list;
extern std::vector<PointVector> Nearest_incre_Points;

extern std::vector<PointVector> Nearest_bigupdate_Points;
extern std::vector<PointVector> Nearest_Points; 
                  // localmap in ivox
extern std::shared_ptr<IVoxType> ivox_all;                     // unified ivox for all points without feature segmentation



extern std::vector<float> pointSearchSqDis;
// extern bool point_selected_oneset_surf[10000];
// extern bool point_selected_bigupdate_surf[30000];
// extern bool point_selected_surf[10000];
extern std::vector<char> point_selected_oneset_surf;
extern std::vector<char> point_selected_bigupdate_surf;
extern std::vector<char> point_selected_surf;
 // = {0};
//extern std::vector<M3D> crossmat_list;
extern int effct_feat_num;
extern int k;
extern int idx;
extern V3D angvel_avr, acc_avr, acc_avr_norm;
extern int feats_down_size;
// extern std::vector<Eigen::Vector3d> normvec_holder;
extern V3D Lidar_T_wrt_IMU; //(Zero3d);
extern M3D Lidar_R_wrt_IMU; //(Eye3d);
extern double G_m_s2;
extern input_ikfom input_in;

Eigen::Matrix<double, 24, 24> process_noise_cov_input();

Eigen::Matrix<double, 30, 30> process_noise_cov_output();

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 24, 24> df_dw_input(state_input &s, const input_ikfom &in);

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in);

// Eigen::Matrix<double, 30, 15> df_dw_output(state_output &s);

// void ensureBigupdateCapacity(std::size_t required_size);

void h_model_input(state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_input_1(state_input &s,  esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_input_bigupdate(state_input &s,  esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_input_bigupdate_1(state_input &s,  esekfom::dyn_share_modified<double> &ekfom_data);
void h_model_output(state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data);

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data);

void pointBodyToWorld(PointType const * const pi, PointType * const po);

extern Eigen::Matrix4d lidar_to_imu;

extern std::vector<pointwithfeature> sourcecloudwithfeature;
extern std::vector<pointwithfeature> sourcecloudwithfeaturedown;//这是要点云流遍历取点的
extern std::vector<int> oneset_effect_feature;
// extern std::vector<int> oneset_feature;
// extern std::vector<int> bigupdate_feature;
#endif