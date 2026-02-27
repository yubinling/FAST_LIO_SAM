// #ifndef PARAM_H
// #define PARAM_H
#pragma once
#include <ros/ros.h>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <cstring>
#include "preprocess.h"
#include "IMU_Processing.h"
#include <sensor_msgs/NavSatFix.h>
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>
#include <mutex>
#include <omp.h>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ivox/ivox3d.h>
#include <Python.h>
#include <condition_variable>
#include <sensor_msgs/Imu.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/Vector3.h>


// #define IVOX_NODE_TYPE_PHC

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = faster_lio::IVox<3, faster_lio::IVoxNodeType::DEFAULT, PointType>;
#endif
extern double pre_thresh_1;
extern double pre_thresh_2;
extern double pre_thresh_3;
extern double pre_thresh_4;
extern double pre_thresh_5;
extern double pre_thresh_6;



extern int small_update_max_iterations; // 小更新的最大迭代次数
extern int bigupdate_max_iterations; // 大更新的最大迭代次数
extern double small_update_interval; // 小更新时间间隔（秒）
extern double pre_thresh_smooth_coeff; // pre_thresh_1 指数平滑系数
extern bool is_first_frame;
extern double lidar_end_time, first_lidar_time, time_con;
extern double first_lidar_end_time;
extern double last_timestamp_lidar, last_timestamp_imu;
extern int pcd_index;
extern IVoxType::Options ivox_options_;
extern int ivox_nearby_type;
extern state_input state_in;
extern state_output state_out;
extern std::string lid_topic, imu_topic;
extern bool prop_at_freq_of_imu, check_satu, con_frame, cut_frame;
extern bool use_imu_as_input, space_down_sample;
extern bool extrinsic_est_en, publish_odometry_without_downsample;
extern int  init_map_size, con_frame_num;
extern double match_s, satu_acc, satu_gyro, cut_frame_time_interval;
extern float  plane_thr;
extern double linearity_thr, planarity_thr, line_residual_thr;
extern double filter_size_surf_min, filter_size_map_min, fov_deg;
// extern double cube_len; 
extern float  DET_RANGE;
extern double dis_thre; // 最近邻搜索的距离阈值
extern bool   imu_en;
extern double imu_time_inte;
extern double laser_point_cov, acc_norm;
extern double acc_cov_input, gyr_cov_input, vel_cov;
extern double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
extern double imu_meas_acc_cov, imu_meas_omg_cov; 
extern int    lidar_type, pcd_save_interval;
extern std::vector<double> gravity_init, gravity;
extern bool   runtime_pos_log, pcd_save_en, path_en;
extern bool   scan_pub_en, scan_body_pub_en;
extern shared_ptr<Preprocess> p_pre;
extern shared_ptr<ImuProcess> p_imu;
extern bool is_first_frame;

extern std::vector<double> extrinT;
extern std::vector<double> extrinR;
extern double time_diff_lidar_to_imu;
extern double lidar_time_inte, first_imu_time;
extern int cut_frame_num, orig_odom_freq;
extern double online_refine_time; //unit: s
extern bool cut_frame_init;
extern double time_update_last, time_current, time_predict_last_const, t_last;

extern MeasureGroup Measures;
//params for voxel mapping algorithm
extern double min_eigen_value ;
extern int max_layer ;

extern int max_cov_points_size ;
extern int max_points_size ;
extern double sigma_num ;
extern double max_voxel_size;
extern std::vector<int> layer_size;

extern double ranging_cov;
extern double angle_cov;
extern std::vector<double> layer_point_size;

extern bool publish_voxel_map;
extern int publish_max_voxel_layer;

//extern std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;

extern std::vector<M3D> var_down_body;
//
extern ofstream fout_out, fout_imu_pbp;
void readParameters(ros::NodeHandle &n);
void open_file();
Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 &orient);
void reset_cov(Eigen::Matrix<double, 24, 24> & P_init);
void reset_cov_output(Eigen::Matrix<double, 30, 30> & P_init_output);


//mulls分割时候用到的全局变量
//lidar odometry related
extern bool scan_to_scan_module_on;//"apply scan-to-scan registration or just scan-to-localmap matching";
extern int initial_scan2scan_frame_num; //"only conduct scan to scan registration for the first　${initial_scan2scan_frame_num} frames");
extern int motion_compensation_method;// "method for motion compensation of lidar (0: disabled, 1: uniform motion model (from point-wise timestamp), 2: from azimuth, 3: from azimuth (rotation-only), 4: imu-assisted)");
extern bool vertical_ang_calib_on;// "apply vertical intrinsic angle correction for sepecific lidar");
extern double vertical_ang_correction_deg;// "the intrinsic correction of the vertical angle of lidar");
extern bool zupt_on_or_not;//"enable zupt (zero velocity updating) or not");
extern bool apply_scanner_filter;// "enable scanner based distance filtering or not");
extern bool semantic_assist_on;// "apply semantic mask to assist the geometric feature points extraction");
extern double cloud_down_res;// "voxel size(m) of downsample for target point cloud");
extern int dist_inverse_sampling_method;// "use distance inverse sampling or not (0: disabled, 1: linear weight, 2: quadratic weight)");
extern double unit_dist;// "distance that correspoinding to unit weight in inverse distance downsampling");
extern bool adaptive_parameters_on;// "use self-adaptive parameters for different surroundings and road situation");
extern double cloud_pca_neigh_r;// "pca neighborhood searching radius(m) for target point cloud");
extern int cloud_pca_neigh_k;// "use only the k nearest neighbor in the r-neighborhood to do PCA");
extern int cloud_pca_neigh_k_min;// "the min number of points in the neighborhood for doing PCA");
extern int pca_down_rate;// "Downsampling rate of the pca querying points and the points used to calculate pca and build the kd-tree");
extern bool sharpen_with_nms_on;//"using non-maximum supression to get the sharpen feature points from unsharpen points or not (use higher threshold)");
extern bool fixed_num_downsampling_on;// "enable/disable the fixed point number downsampling (processing time's standard deviation would br smaller)");
extern int ground_down_fixed_num;// "fixed number of the detected ground feature points (for source)");
extern int pillar_down_fixed_num;// "fixed number of the detected pillar feature points (for source)");
extern int facade_down_fixed_num;// "fixed number of the detected facade feature points (for source)");
extern int beam_down_fixed_num;// "fixed number of the detected beam feature points (for source)");
extern int roof_down_fixed_num;// "fixed number of the detected roof feature points (for source)");
extern int unground_down_fixed_num;// "fixed number of the unground points used for PCA calculation");
extern double gf_grid_size;// "grid size(m) of ground segmentation");
extern double gf_in_grid_h_thre;// "height threshold(m) above the lowest point in a grid for ground segmentation");
extern double gf_neigh_grid_h_thre;// "height threshold(m) among neighbor grids for ground segmentation");
extern double gf_max_h;// "max height(m) allowed for ground point");
extern int ground_normal_method;//"method for estimating the ground points' normal vector ( 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid)");
extern double gf_normal_estimation_radius;// "neighborhood radius for local normal estimation of ground points (only enabled when ground_normal_method=1)");
extern int gf_ground_down_rate;// "downsampling decimation rate for target ground point cloud");
extern int gf_nonground_down_rate;// "downsampling decimation rate for non-ground point cloud");
extern double intensity_thre_nonground;// "Points whose intensity is larger than this value would be regarded as highly reflective object so that downsampling would not be applied.");
extern int gf_grid_min_pt_num;// "min number of points in a grid (if < this value, the grid would not be considered");
extern int gf_reliable_neighbor_grid_thre;// "min number of neighboring grid whose point number is larger than gf_grid_min_pt_num-1");
extern int gf_down_down_rate;// "downsampling rate based on the already downsampled ground point clouds used for source point cloud");
extern double feature_pts_ratio_guess;// "A guess of the percent of the geometric feature points in the neighborhood");
extern double linearity_thre;// "pca linearity threshold for target point cloud");
extern double planarity_thre;// "pca planarity threshold for target point cloud");
extern double linearity_thre_down;// "pca linearity threshold for source point cloud");
extern double planarity_thre_down;// "pca planarity threshold for source point cloud");
extern double curvature_thre;// "pca local curvature threshold");
extern int bsc_grid_num_per_side;// "numbder of grid per side in BSC feature");
extern double beam_direction_ang;// "the verticle angle threshold for the direction vector of beam-type feature points");
extern double pillar_direction_ang;// "the verticle angle threshold for the direction vector of pillar-type feature points");
extern double facade_normal_ang;// "the verticle angle threshold for the normal vector of facade-type feature points");
extern double roof_normal_ang;// "the verticle angle threshold for the normal vector roof-type feature points");
extern double beam_max_height;// "max bearable height for beam points");
extern int vertex_extraction_method;// "extraction method of vertex points (0: disabled, 1: maximum local curvature within stable points, 2: intersection points of pillar and beams)");
extern bool detect_curb_or_not;// "detect curb feature for urban scenarios or not");
extern bool apply_roi_filter;//"use the region of interest filter to remove dynamic objects or not");
extern double roi_min_y;// "region of interest (delete part): min_y");
extern double roi_max_y;// "region of interest (delete part): max_y");
extern string used_feature_type;// "used_feature_type (1: on, 0: off, order: ground, pillar, beam, facade, roof, vetrex)");
extern bool dense_publish_en; // 是否发布稠密去畸变点云
extern bool reg_intersection_filter_on;// "filter the points outside the intersection aera of two point cloud during registration");
extern bool normal_shooting_on;// "using normal shooting instead of nearest neighbor searching when determing correspondences");
extern double normal_bearing;// "the normal consistency checking angle threshold (unit: degree)");
extern double corr_dis_thre_init;// "distance threshold between correspondence points at begining");
extern double corr_dis_thre_min;// "minimum distance threshold between correspondence points at begining");
extern double dis_thre_update_rate;// "update rate (divided by this value at each iteration) for distance threshold between correspondence points");
extern string corr_weight_strategy;// "weighting strategy for correspondences (1: on, 0: off, order: x,y,z balanced weight, residual weight, distance weight, intensity weight)");
extern double z_xy_balance_ratio;// "the weight ratio of the error along z and x,y direction when balanced weight is enabled");
extern double pt2pt_res_window;// "residual window size for the residual robust kernel function of point to point correspondence");
extern double pt2pl_res_window;// "residual window size for the residual robust kernel function of point to plane correspondence");
extern double pt2li_res_window;// "residual window size for the residual robust kernel function of point to line correspondence");
extern int reg_max_iter_num_s2s;// "max iteration number for icp-based registration (scan to scan)");
extern int reg_max_iter_num_s2m;// "max iteration number for icp-based registration (scan to map)");
extern int reg_max_iter_num_m2m;// "max iteration number for icp-based registration (map to map)");
extern double converge_tran;// "convergence threshold for translation (in m)");
extern double converge_rot_d;// "convergence threshold for rotation (in degree)");
extern double post_sigma_thre;// "the maximum threshold for the posterior standard deviation of the least square adjustment during the registration.(unit:m)");
extern double local_map_radius;// "the radius of the local map (regarded as a sphere aera)");
extern int local_map_max_pt_num;// "max point number allowed for the local map");
extern int local_map_max_vertex_pt_num;// "max vertex point number allowed for the local map");
extern double append_frame_radius;// "the radius of the frame that used to append into the local map");
extern bool apply_map_based_dynamic_removal;// "use map based dynamic object removal or not");
extern double map_min_dist_within_feature;// "if the expanded feature point is too close to already exsit map points, it would not be added to the map");
extern double dynamic_removal_radius;// "the radius of the map based dynamic object removing");
extern double dynamic_dist_thre_min;// "the distance threshold to judge if a point is dynamic or not");
extern int local_map_recalculation_frequency;// "Recalculate the linear features in the local map each ${local_map_recalculation_frequency} frame");
extern int s2m_frequency;// "frequency of scan to map registration");
extern int initial_guess_mode;// "Use which kind of initial guess(0: no initial guess, 1: uniform motion(translation only), 2: uniform motion(translation+rotation), 3:imu based)");
//prior knowledage
extern double approx_scanner_height;// "approximate height of the scanner (m)");
extern double underground_height_thre;// "z-axis threshold for rejecting underground ghost points (lines)");
