#include "parameters.h"
#include <float.h>

bool is_first_frame = true;
double lidar_end_time = 0.0, first_lidar_time = 0.0, time_con = 0.0;
double pre_thresh_1 = 0.0, pre_thresh_2 = 0.0, pre_thresh_3 = 0.0,
       pre_thresh_4 = 0.0, pre_thresh_5 = 0.0, pre_thresh_6 = 0.0;
double first_lidar_end_time=0.0;//第一帧的末尾时间
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
int pcd_index = 0;
IVoxType::Options ivox_options_;
int ivox_nearby_type = 6;

std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);
state_input state_in;
state_output state_out;
std::string lid_topic, imu_topic;
bool prop_at_freq_of_imu = true, check_satu = true, con_frame = false, cut_frame = false;
bool use_imu_as_input = false, space_down_sample = true, publish_odometry_without_downsample = false;
int  init_map_size = 10, con_frame_num = 1;
double match_s = 81, satu_acc, satu_gyro, cut_frame_time_interval = 0.1;
float  plane_thr = 0.1f;
double linearity_thr = 0.7, planarity_thr = 0.5, line_residual_thr = 0.2;
double filter_size_surf_min = 0.5, filter_size_map_min = 0.5, fov_deg = 180;
double dis_thre = 5.0; // 最近邻搜索的距离阈值
// double cube_len = 2000; 
int small_update_max_iterations = 4; // 小更新的最大迭代次数
int bigupdate_max_iterations = 4; // 大更新的最大迭代次数
float  DET_RANGE = 450;
bool   imu_en = true;
double imu_time_inte = 0.005;
double laser_point_cov = 0.01, acc_norm;
double vel_cov, acc_cov_input, gyr_cov_input;
double gyr_cov_output, acc_cov_output, b_gyr_cov, b_acc_cov;
double imu_meas_acc_cov, imu_meas_omg_cov; 
int    lidar_type, pcd_save_interval;
std::vector<double> gravity_init, gravity;
bool   runtime_pos_log, pcd_save_en, path_en, extrinsic_est_en = true;
bool   scan_pub_en, scan_body_pub_en;
shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;
double time_diff_lidar_to_imu = 0.0;

double lidar_time_inte = 0.1, first_imu_time = 0.0;
int cut_frame_num = 1, orig_odom_freq = 10;
double online_refine_time = 20.0; //unit: s
bool cut_frame_init = false; // true;

MeasureGroup Measures;

//mulls分割时候用到的全局变量
//lidar odometry related
bool scan_to_scan_module_on=false;//"apply scan-to-scan registration or just scan-to-localmap matching";
int initial_scan2scan_frame_num=2; //"only conduct scan to scan registration for the first　${initial_scan2scan_frame_num} frames");
int motion_compensation_method= 0;// "method for motion compensation of lidar (0: disabled, 1: uniform motion model (from point-wise timestamp), 2: from azimuth, 3: from azimuth (rotation-only), 4: imu-assisted)");
bool vertical_ang_calib_on=false;// "apply vertical intrinsic angle correction for sepecific lidar");
double vertical_ang_correction_deg= 0.0;// "the intrinsic correction of the vertical angle of lidar");
bool zupt_on_or_not= false;//"enable zupt (zero velocity updating) or not");
bool apply_scanner_filter= false;// "enable scanner based distance filtering or not");
bool semantic_assist_on= false;// "apply semantic mask to assist the geometric feature points extraction");
double cloud_down_res= 0.0;// "voxel size(m) of downsample for target point cloud");
int dist_inverse_sampling_method= 2;// "use distance inverse sampling or not (0: disabled, 1: linear weight, 2: quadratic weight)");
double unit_dist= 15.0;// "distance that correspoinding to unit weight in inverse distance downsampling");
bool adaptive_parameters_on= false;// "use self-adaptive parameters for different surroundings and road situation");
double cloud_pca_neigh_r= 1.2;// "pca neighborhood searching radius(m) for target point cloud");
int cloud_pca_neigh_k= 25;// "use only the k nearest neighbor in the r-neighborhood to do PCA");
int cloud_pca_neigh_k_min= 8;// "the min number of points in the neighborhood for doing PCA");
int pca_down_rate= 2;// "Downsampling rate of the pca querying points and the points used to calculate pca and build the kd-tree");
bool sharpen_with_nms_on= false;//"using non-maximum supression to get the sharpen feature points from unsharpen points or not (use higher threshold)");
bool fixed_num_downsampling_on= true;// "enable/disable the fixed point number downsampling (processing time's standard deviation would br smaller)");
int ground_down_fixed_num= 1500;// "fixed number of the detected ground feature points (for source)");
int pillar_down_fixed_num= 600;// "fixed number of the detected pillar feature points (for source)");
int facade_down_fixed_num= 2000;// "fixed number of the detected facade feature points (for source)");
int beam_down_fixed_num= 500;// "fixed number of the detected beam feature points (for source)");
int roof_down_fixed_num= 100;// "fixed number of the detected roof feature points (for source)");
int unground_down_fixed_num= 10000;// "fixed number of the unground points used for PCA calculation");
double gf_grid_size= 2.5;// "grid size(m) of ground segmentation");
double gf_in_grid_h_thre= 0.25;// "height threshold(m) above the lowest point in a grid for ground segmentation");
double gf_neigh_grid_h_thre=1.5;// "height threshold(m) among neighbor grids for ground segmentation");
double gf_max_h= 2.0;// "max height(m) allowed for ground point");
int ground_normal_method= 3;//"method for estimating the ground points' normal vector ( 0: directly use (0,0,1), 1: estimate normal in fix radius neighborhood , 2: estimate normal in k nearest neighborhood, 3: use ransac to estimate plane coeffs in a grid)");
double gf_normal_estimation_radius=2.0;// "neighborhood radius for local normal estimation of ground points (only enabled when ground_normal_method=1)");
int gf_ground_down_rate= 15;// "downsampling decimation rate for target ground point cloud");
int gf_nonground_down_rate= 3;// "downsampling decimation rate for non-ground point cloud");
double intensity_thre_nonground= FLT_MAX;// "Points whose intensity is larger than this value would be regarded as highly reflective object so that downsampling would not be applied.");
int gf_grid_min_pt_num= 10;// "min number of points in a grid (if < this value, the grid would not be considered");
int gf_reliable_neighbor_grid_thre= 0;// "min number of neighboring grid whose point number is larger than gf_grid_min_pt_num-1");
int gf_down_down_rate= 2;// "downsampling rate based on the already downsampled ground point clouds used for source point cloud");
double feature_pts_ratio_guess=0.3;// "A guess of the percent of the geometric feature points in the neighborhood");
double linearity_thre= 0.65;// "pca linearity threshold for target point cloud");
double planarity_thre= 0.65;// "pca planarity threshold for target point cloud");
double linearity_thre_down= 0.75;// "pca linearity threshold for source point cloud");
double planarity_thre_down=0.75;// "pca planarity threshold for source point cloud");
double curvature_thre= 0.12;// "pca local curvature threshold");
int bsc_grid_num_per_side= 7;// "numbder of grid per side in BSC feature");
double beam_direction_ang= 25;// "the verticle angle threshold for the direction vector of beam-type feature points");
double pillar_direction_ang= 60;// "the verticle angle threshold for the direction vector of pillar-type feature points");
double facade_normal_ang= 30;// "the verticle angle threshold for the normal vector of facade-type feature points");
double roof_normal_ang= 75;// "the verticle angle threshold for the normal vector roof-type feature points");
double beam_max_height= FLT_MAX;// "max bearable height for beam points");
int vertex_extraction_method= 2;// "extraction method of vertex points (0: disabled, 1: maximum local curvature within stable points, 2: intersection points of pillar and beams)");
bool detect_curb_or_not=  false;// "detect curb feature for urban scenarios or not");
bool apply_roi_filter= false;//"use the region of interest filter to remove dynamic objects or not");
double roi_min_y= -FLT_MAX;// "region of interest (delete part): min_y");
double roi_max_y= FLT_MAX;// "region of interest (delete part): max_y");
string used_feature_type= "111100";// "used_feature_type (1: on, 0: off, order: ground, pillar, beam, facade, roof, vetrex)");
bool reg_intersection_filter_on= true;// "filter the points outside the intersection aera of two point cloud during registration");
bool normal_shooting_on= false;// "using normal shooting instead of nearest neighbor searching when determing correspondences");
double normal_bearing= 35.0;// "the normal consistency checking angle threshold (unit: degree)");
double corr_dis_thre_init= 1.5;// "distance threshold between correspondence points at begining");
double corr_dis_thre_min= 0.5;// "minimum distance threshold between correspondence points at begining");
double dis_thre_update_rate= 1.1;// "update rate (divided by this value at each iteration) for distance threshold between correspondence points");
string corr_weight_strategy= "1101";// "weighting strategy for correspondences (1: on, 0: off, order: x,y,z balanced weight, residual weight, distance weight, intensity weight)");
double z_xy_balance_ratio= 1.0;// "the weight ratio of the error along z and x,y direction when balanced weight is enabled");
double pt2pt_res_window= 0.1;// "residual window size for the residual robust kernel function of point to point correspondence");
double pt2pl_res_window= 0.1;// "residual window size for the residual robust kernel function of point to plane correspondence");
double pt2li_res_window= 0.1;// "residual window size for the residual robust kernel function of point to line correspondence");
int reg_max_iter_num_s2s= 1;// "max iteration number for icp-based registration (scan to scan)");
int reg_max_iter_num_s2m= 1;// "max iteration number for icp-based registration (scan to map)");
int reg_max_iter_num_m2m= 3;// "max iteration number for icp-based registration (map to map)");
double converge_tran= 0.001;// "convergence threshold for translation (in m)");
double converge_rot_d= 0.01;// "convergence threshold for rotation (in degree)");
double post_sigma_thre= 0.35;// "the maximum threshold for the posterior standard deviation of the least square adjustment during the registration.(unit:m)");
double local_map_radius= 50.0;// "the radius of the local map (regarded as a sphere aera)");
int local_map_max_pt_num= 8000;// "max point number allowed for the local map");
int local_map_max_vertex_pt_num= 1000;// "max vertex point number allowed for the local map");
double append_frame_radius= 60.0;// "the radius of the frame that used to append into the local map");
bool apply_map_based_dynamic_removal= false;// "use map based dynamic object removal or not");
double map_min_dist_within_feature= 0.03;// "if the expanded feature point is too close to already exsit map points, it would not be added to the map");
double dynamic_removal_radius= 30.0;// "the radius of the map based dynamic object removing");
double dynamic_dist_thre_min= 0.3;// "the distance threshold to judge if a point is dynamic or not");
int local_map_recalculation_frequency= 99999;// "Recalculate the linear features in the local map each ${local_map_recalculation_frequency} frame");
int s2m_frequency= 1;// "frequency of scan to map registration");
int initial_guess_mode= 2;// "Use which kind of initial guess(0: no initial guess, 1: uniform motion(translation only), 2: uniform motion(translation+rotation), 3:imu based)");
//prior knowledge
double approx_scanner_height=1.5;// "approximate height of the scanner (m)");
double underground_height_thre= -6.0;// "z-axis threshold for rejecting underground ghost points (lines)");

ofstream fout_out, fout_imu_pbp;
bool dense_publish_en = true;
double small_update_interval = 0.02;
double pre_thresh_smooth_coeff = 0.7; // pre_thresh_1 指数平滑系数

void readParameters(ros::NodeHandle &nh)
{
  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());




  



  //我设定的阈值，特征值，特征点个数，迭代次数
  nh.param<int>("common/small_update_max_iterations", small_update_max_iterations, 4); // 小更新的最大迭代次数
  nh.param<int>("common/bigupdate_max_iterations", bigupdate_max_iterations, 4); // 大更新的最大迭代次数
    nh.param<double>("common/pre_thresh_1",pre_thresh_1,0.01);
  nh.param<double>("common/pre_thresh_2",pre_thresh_2,0.01);
  nh.param<double>("common/pre_thresh_3",pre_thresh_3,0.01);
  nh.param<double>("common/pre_thresh_4",pre_thresh_4,0.01);
  nh.param<double>("common/pre_thresh_5",pre_thresh_5,0.01);
  nh.param<double>("common/pre_thresh_6",pre_thresh_6,0.01);


  nh.param<double>("common/small_update_interval", small_update_interval, 0.02);
  nh.param<double>("common/pre_thresh_smooth_coeff", pre_thresh_smooth_coeff, 0.7);
  //pvlio
  //nh.param<float>("mapping/det_range",DET_RANGE,300.f);
  nh.param<int>("mapping/max_points_size", max_points_size, 100);
  nh.param<int>("mapping/max_cov_points_size", max_cov_points_size, 100);
  nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
  nh.param<int>("mapping/max_layer", max_layer, 2);
  nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
  nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
  nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
  nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);

  //
  nh.param<bool>("prop_at_freq_of_imu", prop_at_freq_of_imu, 1);
  nh.param<bool>("use_imu_as_input", use_imu_as_input, 0);
  nh.param<bool>("check_satu", check_satu, 1);
  nh.param<int>("init_map_size", init_map_size, 100);
  nh.param<bool>("space_down_sample", space_down_sample, 1);
  nh.param<double>("mapping/satu_acc",satu_acc,3.0);
  nh.param<double>("mapping/satu_gyro",satu_gyro,35.0);
  nh.param<double>("mapping/acc_norm",acc_norm,1.0);
  nh.param<float>("mapping/plane_thr", plane_thr, 0.05f);
  nh.param<double>("mapping/linearity_thr", linearity_thr, 0.7);
  nh.param<double>("mapping/planarity_thr", planarity_thr, 0.5);
  nh.param<double>("mapping/line_residual_thr", line_residual_thr, 0.2);
  nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
  nh.param<std::string>("common/lid_topic",lid_topic,"/livox/lidar");
  nh.param<std::string>("common/imu_topic", imu_topic,"/livox/imu");
  nh.param<bool>("common/con_frame",con_frame,false);
  nh.param<int>("common/con_frame_num",con_frame_num,1);
  nh.param<bool>("common/cut_frame",cut_frame,false);
  nh.param<double>("common/cut_frame_time_interval",cut_frame_time_interval,0.1);
  nh.param<double>("common/time_diff_lidar_to_imu",time_diff_lidar_to_imu,0.0);
  nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
  nh.param<double>("filter_size_map",filter_size_map_min,0.5);
  // nh.param<double>("cube_side_length",cube_len,2000);
  nh.param<float>("mapping/det_range",DET_RANGE,300.f);
  nh.param<double>("mapping/fov_degree",fov_deg,180);
  nh.param<bool>("mapping/imu_en",imu_en,true);
  nh.param<bool>("mapping/extrinsic_est_en",extrinsic_est_en,true);
  nh.param<double>("mapping/imu_time_inte",imu_time_inte,0.005);
  nh.param<double>("mapping/lidar_meas_cov",laser_point_cov,0.1);
  nh.param<double>("mapping/acc_cov_input",acc_cov_input,0.1);
  nh.param<double>("mapping/vel_cov",vel_cov,20);
  nh.param<double>("mapping/gyr_cov_input",gyr_cov_input,0.1);
  nh.param<double>("mapping/gyr_cov_output",gyr_cov_output,0.1);
  nh.param<double>("mapping/acc_cov_output",acc_cov_output,0.1);
  nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
  nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
  nh.param<double>("mapping/imu_meas_acc_cov",imu_meas_acc_cov,0.1);
  nh.param<double>("mapping/imu_meas_omg_cov",imu_meas_omg_cov,0.1);
  nh.param<double>("preprocess/blind", p_pre->blind, 1.0);
  nh.param<int>("preprocess/lidar_type", lidar_type, 1);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
  nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
  nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, 1);
  nh.param<double>("mapping/match_s", match_s, 81);
  nh.param<std::vector<double>>("mapping/gravity", gravity, std::vector<double>());
  nh.param<std::vector<double>>("mapping/gravity_init", gravity_init, std::vector<double>());
  nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT, std::vector<double>());
  nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR, std::vector<double>());
  nh.param<bool>("odometry/publish_odometry_without_downsample", publish_odometry_without_downsample, false);
  nh.param<bool>("publish/path_en",path_en, true);
  nh.param<bool>("publish/dense_publish_en",dense_publish_en, true);
  nh.param<bool>("publish/scan_publish_en",scan_pub_en,1);
  nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en,1);
  nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);

  nh.param<double>("mapping/lidar_time_inte",lidar_time_inte,0.1);
  nh.param<double>("mapping/lidar_meas_cov",laser_point_cov,0.1);
  nh.param<double>("mapping/dis_thre", dis_thre, 5.0); // 最近邻搜索的距离阈值

  nh.param<float>("mapping/ivox_grid_resolution", ivox_options_.resolution_, 0.2);
  nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);
  if (ivox_nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (ivox_nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (ivox_nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (ivox_nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    // LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
    p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
}

Eigen::Matrix<double, 3, 1> SO3ToEuler(const SO3 &rot) 
{
    double sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if(!singular)
    {
        x = atan2(rot(2, 1), rot(2, 2));
        y = atan2(-rot(2, 0), sy);   
        z = atan2(rot(1, 0), rot(0, 0));  
    }
    else
    {    
        x = atan2(-rot(1, 2), rot(1, 1));    
        y = atan2(-rot(2, 0), sy);    
        z = 0;
    }
    Eigen::Matrix<double, 3, 1> ang(x, y, z);
    return ang;
}

void open_file()
{

    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"),ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

}

void reset_cov(Eigen::Matrix<double, 24, 24> & P_init)
{
    P_init = MD(24, 24)::Identity() * 0.1;
    P_init.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6,6)::Identity() * 0.001;


}

void reset_cov_output(Eigen::Matrix<double, 30, 30> & P_init_output)
{
    P_init_output = MD(30, 30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3,3)::Identity() * 0.0001;
    // P_init_output.block<6, 6>(6, 6) = MD(6,6)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6,6)::Identity() * 0.001;
}
