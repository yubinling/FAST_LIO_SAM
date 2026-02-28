// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"
#include <fstream>
#include <string>
#include <algorithm>
#include <cstddef>

namespace {
constexpr std::size_t kInitialBufferSize = 20000;
constexpr std::size_t kInitialBigupdateBufferSize = 30000;
constexpr std::size_t kInitialCrossmatBigupdateBodySize = 50000;

void ensurePointCloudSize(const PointCloudXYZI::Ptr &cloud, std::size_t required_size)
{
	if (cloud->size() < required_size)
	{
		cloud->resize(required_size);
	}
}

template <typename StateT>
M3D calcCurrentPointWorldCov(const StateT &s, const V3D &p_body)
{
	// 当前扫描点的测量协方差只由激光测量模型给出，再旋到世界系；
	// 状态先验协方差由 EKF 的 P 处理，避免在观测噪声里重复计算。
	const M3D body_cov = calc_lidar_point_cov(p_body, ranging_cov, angle_cov);
	const M3D rot_world_imu = s.rot;
	const M3D rot_lidar_imu = extrinsic_est_en ? M3D(s.offset_R_L_I) : Lidar_R_wrt_IMU;
	const M3D rot_world_lidar = rot_world_imu * rot_lidar_imu;
	return rot_world_lidar * body_cov * rot_world_lidar.transpose();
}
} // namespace

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(kInitialBufferSize, 1));
std::vector<int> time_seq;
std::vector<int> time_seq1;
std::vector<pointwithfeature> sourcecloudwithfeaturedown;
std::vector<pointwithfeature> sourcecloudwithfeature;//这是要点云流遍历取点的
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI(kInitialBufferSize, 1));
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI(kInitialBufferSize, 1));
// PointCloudXYZI::Ptr feats_down_world_ground(new PointCloudXYZI(20000, 1));
// PointCloudXYZI::Ptr feats_down_world_pillar(new PointCloudXYZI(20000, 1));
// PointCloudXYZI::Ptr feats_down_world_roof(new PointCloudXYZI(20000, 1));
// PointCloudXYZI::Ptr feats_down_world_beam(new PointCloudXYZI(20000, 1));
// PointCloudXYZI::Ptr feats_down_world_facade(new PointCloudXYZI(20000, 1));
// PointCloudXYZI::Ptr feats_down_world_vertex(new PointCloudXYZI(20000, 1));

PointCloudXYZI::Ptr feats_down_oneset_body(new PointCloudXYZI(kInitialBufferSize, 1));
PointCloudXYZI::Ptr feats_down_oneset_world(new PointCloudXYZI(kInitialBufferSize, 1));
PointCloudXYZI::Ptr feats_down_bigupdate_world(new PointCloudXYZI(kInitialBufferSize, 1));
PointCloudXYZI::Ptr feats_down_bigupdate_body(new PointCloudXYZI(kInitialBufferSize, 1));
PointCloudXYZI::Ptr feats_down_incre_world(new PointCloudXYZI(kInitialBufferSize, 1));//更新map需要很多组点积累到一起

//lyb
std::vector<int> oneset_pointnumbers(kInitialBufferSize);
std::vector<int> bigupdate_pointnumbers(kInitialBufferSize);//这是存放bigupdate中有效的点对应的全部点的序号
// std::vector<int> bigupdate_feature(kInitialBufferSize);

std::vector<M3D> crossmat_effect_list(kInitialBufferSize);
std::vector<M3D> crossmat_oneset_list(kInitialBufferSize);
std::vector<M3D> crossmat_bigupdate_imu_list(kInitialBigupdateBufferSize);
std::vector<M3D> crossmat_bigupdate_body_list(kInitialCrossmatBigupdateBodySize);
std::vector<PointVector> Nearest_oneset_Points(kInitialBufferSize); 
std::vector<PointVector> Nearest_incre_Points(kInitialBufferSize);


std::vector<PointVector> Nearest_bigupdate_Points(kInitialBufferSize);
std::vector<LidarMatchResult> lidar_match_oneset(kInitialBufferSize);
std::vector<LidarMatchResult> lidar_match_bigupdate(kInitialBigupdateBufferSize);
std::vector<LidarMatchResult> lidar_match_frame(kInitialBufferSize);

double solve_time_1=0;
double solve_time_2=0;
double solve_time_3=0;

// 统计小更新之间的时间间隔，用于后续更新频率分析
double observed_small_update_interval_sum = 0.0;
int observed_small_update_interval_count = 0;
double last_small_stats_time = 0.0;
const double small_update_stats_interval = 5.0; // 统计窗口时长（秒）
double small_update_stats_time_origin = -1.0;    // 统计起点，对应日志中的 0 秒
bool small_update_log_initialized = false;
extern std::string root_dir;

// 将一次统计结果写入 txt 文件
void log_small_update_stats_to_file(double start_time, double end_time,
                                    int count, double avg_interval)
{
	if (!small_update_log_initialized)
	{
		// 第一次使用时，按 root_dir/Log 创建/清空文件
		std::string log_dir = root_dir + "/Log";
		// 不强制创建目录，这里假设和 laserMapping 一样已存在；如需可用 boost::filesystem 创建
		std::string log_path_init = root_dir + "/Log/small_update_stats.txt";
		std::ofstream ofs_init(log_path_init.c_str(), std::ios::out | std::ios::trunc);
		if (ofs_init.is_open())
		{
			ofs_init << "# small update stats log" << std::endl;
		}
		small_update_log_initialized = true;
	}

	std::string log_path = root_dir + "/Log/small_update_stats.txt";
	std::ofstream ofs(log_path.c_str(), std::ios::app);
	if (!ofs.is_open())
	{
		return;
	}
	ofs << "[small_update_stats] from " << start_time << " s to " << end_time
	    << " s, count=" << count
	    << ", avg_interval=" << avg_interval << " s" << std::endl;
}

// 将每次 pre_thresh_1 更新后的值写入同一文件
void log_pre_thresh_to_file(double t, double thresh)
{
	if (!small_update_log_initialized)
	{
		std::string log_path_init = root_dir + "/Log/small_update_stats.txt";
		std::ofstream ofs_init(log_path_init.c_str(), std::ios::out | std::ios::trunc);
		if (ofs_init.is_open())
		{
			ofs_init << "# small update stats log" << std::endl;
		}
		small_update_log_initialized = true;
	}

	std::string log_path = root_dir + "/Log/small_update_stats.txt";
	std::ofstream ofs(log_path.c_str(), std::ios::app);
	if (!ofs.is_open())
	{
		return;
	}
	ofs << "[pre_thresh] t=" << t << " s, pre_thresh_1=" << thresh << std::endl;
}

// void log_effect_oneset_before_rebuild(double t, int effect_count, int total_count)
// {
// 	std::string log_path = root_dir + "/Log/effect_oneset_before_rebuild.txt";
// 	std::ofstream ofs(log_path.c_str(), std::ios::app);
// 	if (!ofs.is_open())
// 	{
// 		return;
// 	}
// 	ofs << "[effect_oneset_before_rebuild] t=" << t
// 	    << " s, effect_oneset_num=" << effect_count
// 	    << ", total_oneset_num=" << total_count << std::endl;
// }

void log_bigupdate_capacity_debug(const char *stage, std::size_t required_size)
{
	// Debug file logging disabled.
	(void)stage;
	(void)required_size;
}

void abort_bigupdate_bounds(const char *context, std::size_t index, std::size_t size_limit)
{
	// Debug bounds abort disabled.
	(void)context;
	(void)index;
	(void)size_limit;
}

// 每当新增一次 time_since_last_update 采样后调用，若达到统计周期则输出并清零
void check_and_reset_small_update_stats()
{
	if (last_small_stats_time == 0.0)
	{
		last_small_stats_time = time_current;
		return;
	}

	double stats_window = time_current - last_small_stats_time;
	if (stats_window >= small_update_stats_interval)
	{
		if (observed_small_update_interval_count > 0)
		{
			// 初始化统计起点，使日志时间从 0 开始
			if (small_update_stats_time_origin < 0.0)
			{
				small_update_stats_time_origin = last_small_stats_time;
			}

			double avg_interval = observed_small_update_interval_sum /
			                      static_cast<double>(observed_small_update_interval_count);

			// 将绝对时间转为相对起点的时间（秒）
			double start_time = last_small_stats_time - small_update_stats_time_origin;
			double end_time   = time_current - small_update_stats_time_origin;

			log_small_update_stats_to_file(start_time, end_time,
			                               observed_small_update_interval_count,
			                               avg_interval);
		}

		// 清零重新统计
		observed_small_update_interval_sum = 0.0;
		observed_small_update_interval_count = 0;
		last_small_stats_time = time_current;
	}
}

int total_incre_idx=0;//积累一组之后的序列号


int total_incre_num=0;
int bigupdate_num=0;

int bigupdate_effect_num=0;//这是每次计算满了一定数字以后实际重新算出来的effect_num的数字
int sets_effect_num=0;//这是几个set累计下来的有效数字，用来看有没有满足数量的阈值，满足了以后要重新算有效点个数
int total_oneset_num = 0;//是指上一个set结束到下一个set结束的所有点数，也就是相当于一次更新的feats_down_body的数量
int effect_oneset_num =0;//是指一个set的有效点数
int oneset_idx=0;//指这个set所有点云的序号
int oneset_startupdate_idx=0;//因为在一个set中包含有效点和无效点，都要放进feats_down_oneset_body，所以需要记录每个时刻的idx
int oneset_num=0;//是指在开始有效之后到这个set结束的所有点数 
bool if_effect_start = false;
int effect_num_oneframe=0;
std::vector<V3D> pbody_oneset_list(kInitialBufferSize);
std::vector<V3D> pbody_bigupdate_list(kInitialBufferSize);
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points; 
                 // localmap in ivox
std::shared_ptr<IVoxType> ivox_all = nullptr;                     // unified ivox for all points without feature segmentation

std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
std::vector<char> point_selected_surf(kInitialBufferSize, 0);
std::vector<char> point_selected_oneset_surf(kInitialBufferSize, 0);
std::vector<int> oneset_effect_feature(5000,0);
// std::vector<int> oneset_feature(10000,0);
// bool point_selected_bigupdate_surf[30000] ={0};
std::vector<char> point_selected_bigupdate_surf(kInitialBigupdateBufferSize, 0);
//std::vector<M3D> crossmat_list;
int effct_feat_num = 0;
int k = 0;
int idx = -1;
esekfom::esekf<state_input, 24, input_ikfom> kf_input;
esekfom::esekf<state_input, 24, input_ikfom> kf_input_undistort;
esekfom::esekf<state_output, 30, input_ikfom> kf_output;
input_ikfom input_in;
V3D angvel_avr, acc_avr, acc_avr_norm;
int feats_down_size = 0;  
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
double G_m_s2 = 9.81;

namespace {
void ensureOnesetCapacity(std::size_t required_size)
{
	ensurePointCloudSize(normvec, required_size);
	ensurePointCloudSize(feats_down_oneset_body, required_size);
	ensurePointCloudSize(feats_down_oneset_world, required_size);

	if (oneset_pointnumbers.size() < required_size)
	{
		oneset_pointnumbers.resize(required_size);
	}
	if (crossmat_oneset_list.size() < required_size)
	{
		crossmat_oneset_list.resize(required_size);
	}
	if (Nearest_oneset_Points.size() < required_size)
	{
		Nearest_oneset_Points.resize(required_size);
	}
	if (pbody_oneset_list.size() < required_size)
	{
		pbody_oneset_list.resize(required_size);
	}
	if (point_selected_oneset_surf.size() < required_size)
	{
		point_selected_oneset_surf.resize(required_size, 0);
	}
	if (lidar_match_oneset.size() < required_size)
	{
		lidar_match_oneset.resize(required_size);
	}
}

void ensureFrameSelectionCapacity(std::size_t required_size)
{
	if (point_selected_surf.size() < required_size)
	{
		point_selected_surf.resize(required_size, 0);
	}
	if (lidar_match_frame.size() < required_size)
	{
		lidar_match_frame.resize(required_size);
	}
}
 // namespace

void ensureBigupdateCapacity(std::size_t required_size)
{
	// log_bigupdate_capacity_debug("before_ensureBigupdateCapacity", required_size);
	ensurePointCloudSize(normvec, required_size);
	ensurePointCloudSize(feats_down_bigupdate_world, required_size);
	ensurePointCloudSize(feats_down_bigupdate_body, required_size);

	if (bigupdate_pointnumbers.size() < required_size)
	{
		bigupdate_pointnumbers.resize(required_size);
	}
	if (Nearest_bigupdate_Points.size() < required_size)
	{
		Nearest_bigupdate_Points.resize(required_size);
	}
	if (pbody_bigupdate_list.size() < required_size)
	{
		pbody_bigupdate_list.resize(required_size);
	}
	if (crossmat_bigupdate_imu_list.size() < required_size)
	{
		crossmat_bigupdate_imu_list.resize(required_size);
	}
	if (crossmat_bigupdate_body_list.size() < required_size)
	{
		crossmat_bigupdate_body_list.resize(required_size);
	}
	if (point_selected_bigupdate_surf.size() < required_size)
	{
		point_selected_bigupdate_surf.resize(required_size, 0);
	}
	if (lidar_match_bigupdate.size() < required_size)
	{
		lidar_match_bigupdate.resize(required_size);
	}
	// log_bigupdate_capacity_debug("after_ensureBigupdateCapacity", required_size);
}

// 统一处理一次 LiDAR 观测匹配：先做 PCA 分类，面特征走点面残差，线特征走两维点线残差。
bool buildLidarObservationMatch(const PointType &point_world,
                                double point_body_norm,
                                const PointVector &points_near,
                                LidarMatchResult &match,
                                LidarResidualType &classified_type)
{
    classified_type = LidarResidualType::Invalid;
    match = LidarMatchResult();

    LocalFeaturePCA local_feature;
    if (!compute_local_pca(points_near, local_feature))
    {
        return false;
    }

    classified_type = classify_lidar_feature(local_feature, linearity_thr, planarity_thr);
    if (classified_type == LidarResidualType::PointLine)
    {
        return build_point_line_match(point_world, local_feature, points_near,
                                      line_residual_thr, match);
    }
    if (classified_type != LidarResidualType::PointPlane)
    {
        return false;
    }

    return build_point_plane_match(point_world, point_body_norm, local_feature, points_near,
                                   plane_thr, match);
}

// 将一个匹配结果写入 EKF 矩阵；点面写 1 行，点线按两个垂直方向写 2 行。
int appendInputObservationRows(state_input &s,
                               const LidarMatchResult &match,
                               const V3D &p_body,
                               const M3D &p_imu_crossmat,
                               const M3D &p_body_crossmat,
                               const M3D &point_cov_world,
                               Eigen::MatrixXd &h_x,
                               Eigen::VectorXd &z,
                               int row)
{
    for (int r = 0; r < match.residual_count; ++r)
    {
        const V3D &norm_vec = match.residual_normals[r];
        if (extrinsic_est_en)
        {
            V3D C(s.rot.transpose() * norm_vec);
            V3D A(p_imu_crossmat * C);
            V3D B(p_body_crossmat * s.offset_R_L_I.transpose() * C);
            h_x.block<1, 12>(row, 0) << norm_vec(0), norm_vec(1), norm_vec(2),
                                            VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            V3D C(s.rot.transpose() * norm_vec);
            V3D A(p_imu_crossmat * C);
            h_x.block<1, 12>(row, 0) << norm_vec(0), norm_vec(1), norm_vec(2),
                                            VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }
        z(row) = -match.residuals[r];
        const double point_variance =
            (norm_vec.transpose() * point_cov_world * norm_vec)(0, 0);
        const double residual_variance = std::max(
            min_observation_variance(), match.feature_variances[r] + point_variance);
        // 逐残差白化：把每条观测自己的 R_i 吸收到 H 和 z 里，
        // 后端即可继续使用标量 M_Noise = 1。
        const double scale = 1.0 / std::sqrt(residual_variance);
        h_x.row(row) *= scale;
        z(row) *= scale;
        ++row;
    }
    return row;
}
}
Eigen::Matrix<double, 24, 24> process_noise_cov_input()
{
	Eigen::Matrix<double, 24, 24> cov;
	cov.setZero();
	cov.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	cov.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	cov.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	// MTK::get_cov<process_noise_input>::type cov = MTK::get_cov<process_noise_input>::type::Zero();
	// MTK::setDiagonal<process_noise_input, vect3, 0>(cov, &process_noise_input::ng, gyr_cov_input);// 0.03
	// MTK::setDiagonal<process_noise_input, vect3, 3>(cov, &process_noise_input::na, acc_cov_input); // *dt 0.01 0.01 * dt * dt 0.05
	// MTK::setDiagonal<process_noise_input, vect3, 6>(cov, &process_noise_input::nbg, b_gyr_cov); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	// MTK::setDiagonal<process_noise_input, vect3, 9>(cov, &process_noise_input::nba, b_acc_cov);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

Eigen::Matrix<double, 30, 30> process_noise_cov_output()
{
	Eigen::Matrix<double, 30, 30> cov;
	cov.setZero();
	cov.block<3, 3>(12, 12).diagonal() << vel_cov, vel_cov, vel_cov;
	cov.block<3, 3>(15, 15).diagonal() << gyr_cov_output, gyr_cov_output, gyr_cov_output;
	cov.block<3, 3>(18, 18).diagonal() << acc_cov_output, acc_cov_output, acc_cov_output;
	cov.block<3, 3>(24, 24).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	cov.block<3, 3>(27, 27).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
	return cov;
}

Eigen::Matrix<double, 24, 1> get_f_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero();
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	vect3 a_inertial = s.rot * (in.acc-s.ba); // .normalized()
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = omega[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i];//实际的加速度要用显示值加上重力的值； 
	}
	return res;
}

Eigen::Matrix<double, 30, 1> get_f_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 1> res = Eigen::Matrix<double, 30, 1>::Zero();
	vect3 a_inertial = s.rot * s.acc; // .normalized()
	for(int i = 0; i < 3; i++ ){
		res(i) = s.vel[i];
		res(i + 3) = s.omg[i]; 
		res(i + 12) = a_inertial[i] + s.gravity[i]; 
	}
	return res;
}
Eigen::Matrix<double, 24, 24> df_dw_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(12, 12) = -s.rot;
	cov.template block<3, 3>(3, 3) = -Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(15, 15) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(18, 18) = Eigen::Matrix3d::Identity();
	return cov;
}

Eigen::Matrix<double, 24, 24> df_dx_input(state_input &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 24> cov = Eigen::Matrix<double, 24, 24>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(acc_); // .normalized().toRotationMatrix()
	cov.template block<3, 3>(12, 18) = -s.rot; //.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}

Eigen::Matrix<double, 30, 30> df_dx_output(state_output &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 30, 30> cov = Eigen::Matrix<double, 30, 30>::Zero();
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	cov.template block<3, 3>(12, 3) = -s.rot*MTK::hat(s.acc); // .normalized().toRotationMatrix()
	cov.template block<3, 3>(12, 18) = s.rot; //.normalized().toRotationMatrix();
	// Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	// Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	// s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 3>(12, 21) = Eigen::Matrix3d::Identity(); // grav_matrix; 
	cov.template block<3, 3>(3, 15) = Eigen::Matrix3d::Identity(); 
	return cov;
}

void h_model_input(state_input &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)//最初积累点的函数
{
	double match_start = omp_get_wtime();
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	int effect_num_k = 0;
	// 记录当前 set 在 oneset 序列中的起始下标，方便并行访问
	int base_oneset_idx = oneset_idx;
	//lyb
	total_oneset_num+=time_seq[k];//积累点数
	
	// 确保 normvec 有足够大小（在更新阶段会用到）
	// if (normvec->size() < total_oneset_num) {
	// 	normvec->resize(total_oneset_num);
	// }
	std::size_t required_oneset_size = static_cast<std::size_t>(std::max(total_oneset_num, base_oneset_idx + time_seq[k]));
	ensureOnesetCapacity(required_oneset_size);    

	// 第一步：积累阶段 - 单线程快速积累点，只做轻量操作（坐标转换、存储），不做耗时的 KNN 和平面拟合
	// 这样每个 segment 进来时只做轻量操作，快速积累
	for (int j = 0; j < time_seq[k]; j++)
	{
		int cur_idx = base_oneset_idx + j;

		// Modified: 直接使用feats_down_body，不需要sourcecloudwithfeature
		// 需要先找到对应的点，feats_down_body已经排序过了
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		feats_down_oneset_body->points[cur_idx] = point_body_j;

		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); //把sourcecluddown的转换为全局坐标,还需要转换normal
		feats_down_oneset_world->points[cur_idx] = point_world_j;
		
		V3D p_body = pbody_list[idx+j+1];//这里实际没去畸变，但此时不知道去畸变到什么位置，应该在累计点完成后再重新求，但没有做这一步，感觉没太大必要
		pbody_oneset_list[cur_idx]=p_body;
		
		// 积累阶段：只存储点，不做 KNN 和平面拟合（这些耗时操作留到更新阶段并行处理）
		point_selected_oneset_surf[cur_idx] = false;  // 初始化为 false，等更新阶段再计算
	}

	// 更新全局 oneset_idx
	oneset_idx = base_oneset_idx + time_seq[k];
	
	

	//lyb
	// if (effect_oneset_num>0){
	// 	//oneset_num+=time_seq[k];//是指需要更新的总点数，和oneset的总点数区分
	// 	if_effect_start=true;
	
	// }
	// else{
	// 	if (!if_effect_start){
	// 		oneset_startupdate_idx+=time_seq[k];//开始更新的点数的序列
	// 	}
	// }
	// if (if_effect_start)
	// {
	// 	oneset_num+=time_seq[k];//set里所有可以更新的点的数量
	// }
	double t2=omp_get_wtime();

	
	// Modified: 小更新机制 - 每 small_update_interval 秒检查一次特征值（由配置文件设定）
	if (last_small_check_time == 0.0) {
		// 第一次初始化
		last_small_check_time = time_current;
	}
	
	double time_since_last_update = time_current - last_small_check_time;
	bool should_update = false;
	if (time_since_last_update >= small_update_interval) {
		// 每0.02秒计算一次特征值，看与pre_thresh的关系
		MatrixXd p_both=kf_input.P_.topLeftCorner(6,6);
		EigenSolver<Eigen::MatrixXd> solver_3(p_both);
		VectorXd eigenvalues_3= solver_3.eigenvalues().real();
		double max_eigenvalue = eigenvalues_3.maxCoeff();
		double stand_coeff=1.1;//容忍指数
		if (max_eigenvalue < stand_coeff*pre_thresh_1)
		{
			// 特征值都小于阈值，不需要更新，继续积累点
			// 但是如果是本帧最后一个segment，即使不满足小更新条件，也要去畸变后加入大更新
			if (is_last_segment_in_frame) {
				should_update = false; // 不进行小更新，但会继续执行去畸变
				// 统计本次与上次检查之间的时间差
				observed_small_update_interval_sum += time_since_last_update;
				observed_small_update_interval_count++;
				check_and_reset_small_update_stats();
				last_small_check_time = time_current; // 更新检查时间
			} else {
				ekfom_data.valid=false;
				observed_small_update_interval_sum += time_since_last_update;
				last_small_check_time = time_current; // 更新检查时间
				pre_thresh_1 = pre_thresh_smooth_coeff * pre_thresh_1 + (1.0 - pre_thresh_smooth_coeff) * max_eigenvalue;
				log_pre_thresh_to_file(time_current, pre_thresh_1);
				return;
			}
		} else {
			// 需要更新，更新时间戳
			should_update = true;
			// 统计本次与上次检查之间的时间差
			observed_small_update_interval_sum += time_since_last_update;
			observed_small_update_interval_count++;
			check_and_reset_small_update_stats();
			last_small_check_time = time_current;
		}
		// 动态更新 pre_thresh_1，供下一次使用
		pre_thresh_1 = pre_thresh_smooth_coeff * pre_thresh_1 + (1.0 - pre_thresh_smooth_coeff) * max_eigenvalue;
		log_pre_thresh_to_file(time_current, pre_thresh_1);
	} else {
		// 时间还没到0.02秒，继续积累点，不更新
		// 但是如果是本帧最后一个segment，即使时间没到，也要去畸变后加入大更新
		if (is_last_segment_in_frame) {
			should_update = false; // 不进行小更新，但会继续执行去畸变
			// 统计本次与上次检查之间的时间差
			// observed_small_update_interval_sum += time_since_last_update;
			// observed_small_update_interval_count++;
			// check_and_reset_small_update_stats();
			last_small_check_time = time_current;
		} else {
			ekfom_data.valid = false;
			return;
		}
	}
	
	// 如果是本帧最后一个segment但不满足小更新条件，设置valid为false，但继续执行去畸变
	if (is_last_segment_in_frame && !should_update) {
		ekfom_data.valid = false;
		// 不return，继续执行去畸变代码
	} else if (!should_update) {
		ekfom_data.valid = false;
		return;
	}

		int residual_oneset_num = 0;
	    if(should_update)
    {
		// 第二步：更新阶段 - 当特征值检查通过需要更新时，对积累的所有点并行做 KNN + 平面拟合
			// 这时候 total_oneset_num 通常有几百个点，并行收益大
			double knn_plane_start = omp_get_wtime();
			int plane_count = 0;
			int line_count = 0;
			int invalid_count = 0;
			
			#ifdef MP_EN
			omp_set_num_threads(MP_PROC_NUM);
			#pragma omp parallel for  
		#endif
		for (int i = 0; i < total_oneset_num; i++)
		{
			PointType &point_world_i = feats_down_oneset_world->points[i];
			V3D p_body = pbody_oneset_list[i];
			double p_norm = p_body.norm();
			V3D p_world;
			p_world << point_world_i.x, point_world_i.y, point_world_i.z;

			double t1 = omp_get_wtime();
			PointVector points_near;
			// Modified: 不使用特征分割，统一从ivox_all中查找最近点
			ivox_all->GetClosestPoint(point_world_i, points_near, NUM_MATCH_POINTS, dis_thre);
			solve_time_1 += omp_get_wtime() - t1;

			Nearest_oneset_Points[i] = points_near;
			
				if ((points_near.size() < NUM_MATCH_POINTS))
				{
					point_selected_oneset_surf[i] = false;//表示匹配失败
					#pragma omp atomic
					invalid_count++;
				}
				else
				{   
					point_selected_oneset_surf[i] = false;
					LidarMatchResult match;
					LidarResidualType classified_type;
					if (buildLidarObservationMatch(point_world_i, p_norm, points_near,
					                               match, classified_type))
					{
						if (classified_type == LidarResidualType::PointPlane)
						{
							#pragma omp atomic
							plane_count++;
							normvec->points[i].x = match.plane_coeff(0);
							normvec->points[i].y = match.plane_coeff(1);
							normvec->points[i].z = match.plane_coeff(2);
							normvec->points[i].intensity = match.residual;
						}
						else if (classified_type == LidarResidualType::PointLine)
						{
							#pragma omp atomic
							line_count++;
						}
						lidar_match_oneset[i] = match;
						point_selected_oneset_surf[i] = true;
					}
					else
					{
						#pragma omp atomic
						invalid_count++;
					}
				}
			}
			std::cout << "[h_model_input feature] plane=" << plane_count
			          << ", line=" << line_count
			          << ", invalid=" << invalid_count << std::endl;
		
		// 第三步：串行统计有效点，填充 oneset_pointnumbers 和各种计数
		// 先把进入重建前的状态打到单独日志里，便于确认是否带着旧计数进来
		// log_effect_oneset_before_rebuild(time_current, effect_oneset_num, total_oneset_num);
		// 这里会重新扫描当前累计的全部 total_oneset_num 个点，因此先清零再重建索引
		
			// 统计有效观测点和实际 EKF 残差行数：面特征 1 行，线特征 2 行。
				for (int i = 0; i < total_oneset_num; ++i)
			{
				if (point_selected_oneset_surf[i])
				{
					oneset_pointnumbers[effect_oneset_num] = i;
					residual_oneset_num += lidar_match_oneset[i].residual_count;
					effect_num_k ++;
					effect_oneset_num ++;
					effect_num_oneframe++;
				}
			}
	}
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input += match_cost;
	double solve_start_  = omp_get_wtime();


   
	//需要把feats_down_body统一到现在的点的坐标系下，通过world下的点算回来,去畸变
	#ifdef MP_EN
	omp_set_num_threads(MP_PROC_NUM);
	#pragma omp parallel for  
	#endif  
	for (int j = 0;j<total_oneset_num;j++)
	{    
		V3D p_body;
		V3D p_imu;
		V3D p_normal_body;
		if (extrinsic_est_en)
		{
			V3D p_world(feats_down_oneset_world->points[j].x, feats_down_oneset_world->points[j].y, feats_down_oneset_world->points[j].z);
			V3D p_normal_world(feats_down_oneset_world->points[j].normal_x,feats_down_oneset_world->points[j].normal_y,feats_down_oneset_world->points[j].normal_z);
			p_imu = s.rot.inverse()*(p_world-s.pos);
			p_body= s.offset_R_L_I.inverse()*(p_imu-s.offset_T_L_I);
			p_normal_body=s.offset_R_L_I.inverse()*s.rot.inverse()*p_normal_world;
			
			// M3D point_body_crossmat;
			// point_body_crossmat << SKEW_SYM_MATRX(p_body);
			// crossmat_oneset_list[j]=point_body_crossmat;
		}
		else
		{
			V3D p_world(feats_down_oneset_world->points[j].x, feats_down_oneset_world->points[j].y, feats_down_oneset_world->points[j].z);
			V3D p_normal_world(feats_down_oneset_world->points[j].normal_x,feats_down_oneset_world->points[j].normal_y,feats_down_oneset_world->points[j].normal_z);
			p_imu = s.rot.inverse()*(p_world-s.pos);
			p_body= Lidar_R_wrt_IMU.inverse()*(p_imu-Lidar_T_wrt_IMU);
			p_normal_body=Lidar_R_wrt_IMU.inverse()*s.rot.inverse()*p_normal_world;
			// cout<<"p_normal_world"<<p_normal_world(0)<<"<<"<<p_normal_world(1)<<"<"<<p_normal_world(2)<<endl;
			// cout<<"p_normal_body"<<p_normal_body.x()<<"<<"<<p_normal_body.y()<<"<"<<p_normal_body.z()<<endl;
			
		}
		pbody_oneset_list[j]=p_body;//去畸变了
		M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(p_imu);
		crossmat_oneset_list[j]=point_crossmat;//imu系下面的反对成矩阵
		feats_down_oneset_body->points[j].x = p_body(0);
		feats_down_oneset_body->points[j].y = p_body(1);
		feats_down_oneset_body->points[j].z = p_body(2);
		feats_down_oneset_body->points[j].normal_x=p_normal_body(0);
		feats_down_oneset_body->points[j].normal_y=p_normal_body(1);
		feats_down_oneset_body->points[j].normal_z=p_normal_body(2);

	}
	if (!should_update)
	{
		return;
	}
	ekfom_data.M_Noise = 1.0;
	ekfom_data.h_x.resize(residual_oneset_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(residual_oneset_num, 12);
	ekfom_data.z.resize(residual_oneset_num);
	//fprintf(fp_debug,"effect_num: %d\n",effect_oneset_num);
	cout<<"effect_oneset"<<effect_oneset_num<<endl;
	cout<<"total_oneset_num"<<total_oneset_num<<endl;

	int residual_row = 0;
	for (int j = 0; j < effect_oneset_num; j++)
	{
		const int point_idx = oneset_pointnumbers[j];
		const LidarMatchResult &match = lidar_match_oneset[point_idx];
		V3D p_body = pbody_oneset_list[point_idx];
		M3D p_body_crossmat;
		p_body_crossmat << SKEW_SYM_MATRX(p_body);
		M3D p_imu_crossmat = crossmat_oneset_list[point_idx];
		const M3D point_cov_world = calcCurrentPointWorldCov(s, p_body);
		residual_row = appendInputObservationRows(s, match, p_body, p_imu_crossmat,
		                                          p_body_crossmat, point_cov_world,
		                                          ekfom_data.h_x,
		                                          ekfom_data.z, residual_row);
	}


	solve_time_3 += omp_get_wtime() - solve_start_;

	effct_feat_num += effect_num_k;
}
void h_model_input_1(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data)//
{
	double match_start=omp_get_wtime();
	bool match_in_map = false;
	// VF(4) pabcd;
	// pabcd.setZero();
	// if (normvec->size()<total_oneset_num){
	// normvec->resize(total_oneset_num);}
	ensureOnesetCapacity(static_cast<std::size_t>(total_oneset_num));
		int effect_num_k = 0;
		// int effect_num_k = 0;
		effect_oneset_num=0;
		int plane_count = 0;
		int line_count = 0;
		int invalid_count = 0;
		//lyb
	//total_oneset_num+=time_seq[k];
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for  
	#endif  
	for (int j = 0; j < total_oneset_num; j++)
	{
		PointType &point_body_j  = feats_down_oneset_body->points[j];
		//feats_down_oneset_body->points[oneset_idx] = point_body_j;
		//lyb
		//pointnumbers[oneset_idx]=idx+j+1;//set中feat对应与feats_down_body的序号

		PointType &point_world_j = feats_down_oneset_world->points[j];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		//feats_down_oneset_world->points[oneset_idx] = point_world_j;
		V3D p_body = pbody_oneset_list[j];
		double p_norm = p_body.norm();
		//fprintf(fp_debug,"pnorm:%lf",p_norm);
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{  //fprintf(fp_debug,"p_world.x:%lf",point_world_j.x);
			// auto &points_near =Nearest_oneset_Points[j];
            PointVector points_near;
			if (ekfom_data.converge){//如果上一次收敛，才重新观测
				ivox_all->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS, dis_thre);
			    Nearest_oneset_Points[j]=points_near;
            	//ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);
			    if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
				{
					point_selected_oneset_surf[j] = false;
				}
				else{
					point_selected_oneset_surf[j] = true;
				}
			} //
				if (!point_selected_oneset_surf[j]){
					#pragma omp atomic
					invalid_count++;
					continue;
				}
				else
				{
					points_near=Nearest_oneset_Points[j];
					point_selected_oneset_surf[j] = false;
					LidarMatchResult match;
					LidarResidualType classified_type;
					if (buildLidarObservationMatch(point_world_j, p_norm, points_near,
					                               match, classified_type))
					{
						if (classified_type == LidarResidualType::PointPlane)
						{
							#pragma omp atomic
							plane_count++;
							normvec->points[j].x = match.plane_coeff(0);
							normvec->points[j].y = match.plane_coeff(1);
							normvec->points[j].z = match.plane_coeff(2);
							normvec->points[j].intensity = match.residual;
						}
						else if (classified_type == LidarResidualType::PointLine)
						{
							#pragma omp atomic
							line_count++;
						}
						lidar_match_oneset[j] = match;
						point_selected_oneset_surf[j] = true;
					}
					else
					{
						#pragma omp atomic
						invalid_count++;
					}
				}

		}
			//oneset_idx++;
		}
		std::cout << "[h_model_input_1 feature] plane=" << plane_count
		          << ", line=" << line_count
		          << ", invalid=" << invalid_count << std::endl;
	    int residual_oneset_num = 0;
	    for (int i = 0; i <total_oneset_num; i++)
    {
        if (point_selected_oneset_surf[i])
        {
            // laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            // corr_normvect->points[effct_feat_num] = normvec->points[i];
            effect_num_k++;
			oneset_pointnumbers[effect_oneset_num]=i;
			residual_oneset_num += lidar_match_oneset[i].residual_count;
            effect_oneset_num ++;
        }
    }
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input_1 += match_cost;

	if (effect_oneset_num ==0) 
	{
		ekfom_data.valid = false;
		return;
	}
	double solve_start_  = omp_get_wtime();
	ekfom_data.M_Noise = 1.0;
	ekfom_data.h_x.resize(residual_oneset_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(residual_oneset_num, 12);
	ekfom_data.z.resize(residual_oneset_num);
		// 按匹配类型写入观测雅可比；线特征会连续写入两行点到线残差。
		int residual_row = 0;
		for (int j = 0; j < effect_oneset_num; j++)
		{
			const int point_idx = oneset_pointnumbers[j];
			const LidarMatchResult &match = lidar_match_oneset[point_idx];
			V3D p_body = pbody_list[point_idx];
			M3D p_body_crossmat;
			p_body_crossmat << SKEW_SYM_MATRX(p_body);
			M3D p_imu_crossmat;
			if (extrinsic_est_en)
			{
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
			}
			else
			{
				p_imu_crossmat = crossmat_oneset_list[point_idx];
			}
			const M3D point_cov_world = calcCurrentPointWorldCov(s, p_body);
			residual_row = appendInputObservationRows(s, match, p_body, p_imu_crossmat,
			                                          p_body_crossmat, point_cov_world,
			                                          ekfom_data.h_x,
			                                          ekfom_data.z, residual_row);
		}

    solve_time_3 += omp_get_wtime() - solve_start_;
	effct_feat_num += effect_num_k;
}

void h_model_input_bigupdate(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	double match_start=omp_get_wtime();
	bool match_in_map = false;
	// VF(4) pabcd;
    // pabcd.setZero();
	// if (normvec->size()< bigupdate_num)//现在是一帧进行一次大更新
	// 	{normvec->resize(bigupdate_num);}
	ensureBigupdateCapacity(static_cast<std::size_t>(bigupdate_num));
	int effect_num_k = 0;
	// 这里会重新统计当前 bigupdate 中的全部有效点，必须从 0 开始重建索引
	bigupdate_effect_num = 0;

    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for  
	#endif  
	for (int j = 0; j < bigupdate_num; j++)//这是需要把所有的的set去畸变到现在的lidar坐标下，并计算crossmat，供后面用
	{

		

		V3D p_body;
		V3D p_imu;
		V3D p_normal_body;
		if (extrinsic_est_en)
		{
			V3D p_world(  feats_down_bigupdate_world->points[j].x, feats_down_bigupdate_world->points[j].y, feats_down_bigupdate_world->points[j].z);
			V3D p_normal_world(feats_down_bigupdate_world->points[j].normal_x,feats_down_bigupdate_world->points[j].normal_y,feats_down_bigupdate_world->points[j].normal_z);
			p_imu = s.rot.inverse()*(p_world-s.pos);
			p_body= s.offset_R_L_I.inverse()*(p_imu-s.offset_T_L_I);
			p_normal_body=s.offset_R_L_I.inverse()*s.rot.inverse()*p_normal_world;

			
			M3D point_body_crossmat;
			point_body_crossmat << SKEW_SYM_MATRX(p_body);
			crossmat_bigupdate_body_list[j]=point_body_crossmat;
		}
		else
		{
			V3D p_world(feats_down_bigupdate_world->points[j].x, feats_down_bigupdate_world->points[j].y, feats_down_bigupdate_world->points[j].z);
			V3D p_normal_world(feats_down_bigupdate_world->points[j].normal_x,feats_down_bigupdate_world->points[j].normal_y,feats_down_bigupdate_world->points[j].normal_z);
			p_body= Lidar_R_wrt_IMU.inverse()*(s.rot.inverse()*(p_world-s.pos)-Lidar_T_wrt_IMU);
			p_imu = s.rot.inverse()*(p_world-s.pos);
			p_normal_body=Lidar_R_wrt_IMU.inverse()*s.rot.inverse()*p_normal_world;
		}
		M3D point_crossmat;
		
		point_crossmat << SKEW_SYM_MATRX(p_imu);
		
		crossmat_bigupdate_imu_list[j]=point_crossmat;
		feats_down_bigupdate_body->points[j].x = p_body(0);
		feats_down_bigupdate_body->points[j].y = p_body(1);
		feats_down_bigupdate_body->points[j].z = p_body(2);
		feats_down_bigupdate_body->points[j].normal_x=p_normal_body(0);
		feats_down_bigupdate_body->points[j].normal_y=p_normal_body(1);
		feats_down_bigupdate_body->points[j].normal_z=p_normal_body(2);
		pbody_bigupdate_list[j]=p_body;
			//fprintf(fp_debug,"down_bodypointx:%lf\n",feats_down_body->points[pointnumbers[j]].x);
			// feats_down_body->points[pointnumbers[j]].x=p_body(0);
			// feats_down_body->points[pointnumbers[j]].y=p_body(1);
			// feats_down_body->points[pointnumbers[j]].z=p_body(2);

		//fprintf(fp_debug,"down_bodypointxxx:%lf\n",feats_down_body->points[pointnumbers[j]].x);

	}
	    // 统计 bigupdate 中 KNN + 平面拟合循环的耗时
	    double bigupdate_knn_plane_start = omp_get_wtime();
	    int plane_count = 0;
	    int line_count = 0;
	    int invalid_count = 0;
	    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for  
	#endif  
	for (int j = 0; j < bigupdate_num; j++)//这里只需要算最后一次加进bigupdate_num的,因为前面的没每次经过这个函数都算好了,虽然pbody值变了，应该全部重新算，但是pd2没变，前面的点只需要重新算pnorm，就算新的就行了
	{
		PointType &point_world_j = feats_down_bigupdate_world->points[j];//在lasermapping已经转换好了
		// pointBodyToWorld(&point_body_j, &point_world_j); 
		//feats_down_oneset_world->points[oneset_idx] = point_world_j;
		V3D p_body = pbody_bigupdate_list[j];
		double p_norm = p_body.norm();
		//fprintf(fp_debug,"pnorm:%lf",p_norm);
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		PointVector points_near;
		// Modified: 不使用特征分割，统一从ivox_all中查找最近点
		ivox_all->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS, dis_thre);
		Nearest_bigupdate_Points[j]=points_near;
		// ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_bigupdate_surf[j] = false;
				#pragma omp atomic
				invalid_count++;
				continue;
			}
			point_selected_bigupdate_surf[j] = false;
			LidarMatchResult match;
			LidarResidualType classified_type;
			if (buildLidarObservationMatch(point_world_j, p_norm, points_near,
			                               match, classified_type))
			{
				if (classified_type == LidarResidualType::PointPlane)
				{
					#pragma omp atomic
					plane_count++;
					normvec->points[j].x = match.plane_coeff(0);
					normvec->points[j].y = match.plane_coeff(1);
					normvec->points[j].z = match.plane_coeff(2);
					normvec->points[j].intensity = match.residual;
				}
				else if (classified_type == LidarResidualType::PointLine)
				{
					#pragma omp atomic
					line_count++;
				}
				lidar_match_bigupdate[j] = match;
				point_selected_bigupdate_surf[j] = true;
			}
			else
			{
				#pragma omp atomic
				invalid_count++;
			}

		
	}
		double bigupdate_knn_plane_time = omp_get_wtime() - bigupdate_knn_plane_start;
		std::cout << "[h_model_input_bigupdate] KNN+plane fitting loop time: " << bigupdate_knn_plane_time 
		          << "s, bigupdate_num: " << bigupdate_num << std::endl;
		std::cout << "[h_model_input_bigupdate feature] plane=" << plane_count
		          << ", line=" << line_count
		          << ", invalid=" << invalid_count << std::endl;
		//oneset_idx++;
	
	    // 统计大更新的有效点和残差行数；线特征按两维线残差进入 EKF。
	    int residual_bigupdate_num = 0;
	    for (int i = 0; i<bigupdate_num; i++)
	    {
	        if (point_selected_bigupdate_surf[i])
	        {
	            // laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
	            // corr_normvect->points[effct_feat_num] = normvec->points[i];
	            effect_num_k++;
				bigupdate_pointnumbers[bigupdate_effect_num]=i;//我这里是采用的一个vector把有效点对应到原来的点，voxelmap是有一个单独的数据结构，把有效点之后计算需要的数据push进去
				residual_bigupdate_num += lidar_match_bigupdate[i].residual_count;
	            bigupdate_effect_num++;
				
	        }
	    }
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input_bigupdate += match_cost;

	if (bigupdate_effect_num == 0 || residual_bigupdate_num <= 0)
	{
		// 大更新没有有效残差时直接回退，让 IKFoM 跳过本次观测更新。
		ekfom_data.valid = false;
		return;
	}
	double solve_start_  = omp_get_wtime();
	ekfom_data.M_Noise = 1.0;
	ekfom_data.h_x.resize(residual_bigupdate_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(residual_bigupdate_num, 12);
	// ekfom_data.h_x.bottomRows(bigupdate_effect_num-bigupdate_effect_start_idx).setZero();//后几行置零
	// ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_oneset_num, 12);
	ekfom_data.z.resize(residual_bigupdate_num);
		// 大更新同样按匹配结果写入；线特征的两个垂直残差占两行。
		int residual_row = 0;
		for (int j = 0; j < bigupdate_effect_num; j++)
		{
			const int point_idx = bigupdate_pointnumbers[j];
			const LidarMatchResult &match = lidar_match_bigupdate[point_idx];
			V3D p_body = pbody_bigupdate_list[point_idx];
			M3D p_body_crossmat;
			p_body_crossmat << SKEW_SYM_MATRX(p_body);
			M3D p_imu_crossmat = crossmat_bigupdate_imu_list[point_idx];
			const M3D point_cov_world = calcCurrentPointWorldCov(s, p_body);
			residual_row = appendInputObservationRows(s, match, p_body, p_imu_crossmat,
			                                          p_body_crossmat, point_cov_world,
			                                          ekfom_data.h_x,
			                                          ekfom_data.z, residual_row);
		}
		if (residual_row != residual_bigupdate_num)
		{
			std::cout << "[h_model_input_bigupdate] residual row mismatch, expect="
			          << residual_bigupdate_num << ", actual=" << residual_row << std::endl;
			if (residual_row <= 0)
			{
				ekfom_data.valid = false;
				return;
			}
			// 线/面残差行数出现轻微不一致时，只保留已经安全写入的行。
			ekfom_data.h_x.conservativeResize(residual_row, Eigen::NoChange);
			ekfom_data.z.conservativeResize(residual_row);
		}



    solve_time_3 += omp_get_wtime() - solve_start_;
	effct_feat_num += effect_num_k;
}
void h_model_input_bigupdate_1(state_input &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
	double match_start=omp_get_wtime();
	bool match_in_map = false;
	// VF(4) pabcd;
    // pabcd.setZero();
	// if (normvec->size()< bigupdate_num)
	// 	{normvec->resize(bigupdate_num);}
		ensureBigupdateCapacity(static_cast<std::size_t>(bigupdate_num));
		int effect_num_k = 0;
		bigupdate_effect_num=0;
		int plane_count = 0;
		int line_count = 0;
		int invalid_count = 0;
	
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for  
	#endif  
	for (int j = 0; j < bigupdate_num; j++)//这里只需要算最后一次加进bigupdate_num的,因为前面的没每次经过这个函数都算好了,虽然pbody值变了，应该全部重新算，但是这只是确定哪个点有效，没必要重新算
	{
		PointType &point_body_j  = feats_down_bigupdate_body->points[j];
		PointType &point_world_j = feats_down_bigupdate_world->points[j];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		//feats_down_oneset_world->points[oneset_idx] = point_world_j;
		V3D p_body = pbody_bigupdate_list[j];
		double p_norm = p_body.norm();
		//fprintf(fp_debug,"pnorm:%lf",p_norm);
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		
		// auto &points_near =Nearest_bigupdate_Points[j];
		PointVector points_near;
		if (ekfom_data.converge){//如果上一次收敛，才重新观测
			// Modified: 不使用特征分割，统一从ivox_all中查找最近点
			ivox_all->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS, dis_thre);
		    Nearest_bigupdate_Points[j]=points_near;
		// ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS);
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_bigupdate_surf[j] = false;
			}
			else{
				point_selected_bigupdate_surf[j] = true;
			}
		}
		else{//用之前的匹配结果
		    points_near=Nearest_bigupdate_Points[j];
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5) // 5)
			{
				point_selected_bigupdate_surf[j] = false;
			}
			else{
				point_selected_bigupdate_surf[j] = true;
			}

			} //
			if (!point_selected_bigupdate_surf[j]){
				#pragma omp atomic
				invalid_count++;
				continue;
			} 
			point_selected_bigupdate_surf[j] = false;
			LidarMatchResult match;
			LidarResidualType classified_type;
			if (buildLidarObservationMatch(point_world_j, p_norm, points_near,
			                               match, classified_type))
			{
				if (classified_type == LidarResidualType::PointPlane)
				{
					#pragma omp atomic
					plane_count++;
					normvec->points[j].x = match.plane_coeff(0);
					normvec->points[j].y = match.plane_coeff(1);
					normvec->points[j].z = match.plane_coeff(2);
					normvec->points[j].intensity = match.residual;
				}
				else if (classified_type == LidarResidualType::PointLine)
				{
					#pragma omp atomic
					line_count++;
				}
				lidar_match_bigupdate[j] = match;
				point_selected_bigupdate_surf[j] = true;
			}
			else
			{
				#pragma omp atomic
				invalid_count++;
			}

		
		}
		std::cout << "[h_model_input_bigupdate_1 feature] plane=" << plane_count
		          << ", line=" << line_count
		          << ", invalid=" << invalid_count << std::endl;
			//oneset_idx++;
	
	    // 统计大更新迭代时的残差行数，保证线特征两行不会覆盖矩阵。
	    int residual_bigupdate_num = 0;
	    for (int i = 0; i<bigupdate_num; i++)
	    {
	        if (point_selected_bigupdate_surf[i])
	        {
	            // laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
	            // corr_normvect->points[effct_feat_num] = normvec->points[i];
	            effect_num_k++;
				bigupdate_pointnumbers[bigupdate_effect_num]=i;//我这里是采用的一个vector把有效点对应到原来的点，voxelmap是有一个单独的数据结构，把有效点之后计算需要的数据push进去
				residual_bigupdate_num += lidar_match_bigupdate[i].residual_count;
	            bigupdate_effect_num++;
				
	        }
	    }
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input_bigupdate_1 += match_cost;
	//lyb

	if (bigupdate_effect_num == 0 || residual_bigupdate_num <= 0)
	{
		// 后续迭代也可能因为匹配质量下降没有有效观测，此时保持上一轮状态。
		ekfom_data.valid = false;
		return;
	}

	double solve_start_  = omp_get_wtime();
	ekfom_data.M_Noise = 1.0;
	ekfom_data.h_x.resize(residual_bigupdate_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(residual_bigupdate_num, 12);
	// ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_oneset_num, 12);
	ekfom_data.z.resize(residual_bigupdate_num);
		// 大更新迭代同样按匹配结果写入；线特征的两个垂直残差占两行。
		int residual_row = 0;
		for (int j = 0; j < bigupdate_effect_num; j++)
		{
			const int point_idx = bigupdate_pointnumbers[j];
			const LidarMatchResult &match = lidar_match_bigupdate[point_idx];
			V3D p_body = pbody_bigupdate_list[point_idx];
			M3D p_body_crossmat;
			p_body_crossmat << SKEW_SYM_MATRX(p_body);
			M3D p_imu_crossmat = crossmat_bigupdate_imu_list[point_idx];
			const M3D point_cov_world = calcCurrentPointWorldCov(s, p_body);
			residual_row = appendInputObservationRows(s, match, p_body, p_imu_crossmat,
			                                          p_body_crossmat, point_cov_world,
			                                          ekfom_data.h_x,
			                                          ekfom_data.z, residual_row);
		}
		if (residual_row != residual_bigupdate_num)
		{
			std::cout << "[h_model_input_bigupdate_1] residual row mismatch, expect="
			          << residual_bigupdate_num << ", actual=" << residual_row << std::endl;
			if (residual_row <= 0)
			{
				ekfom_data.valid = false;
				return;
			}
			// 避免把未填充的观测行交给矩阵求逆。
			ekfom_data.h_x.conservativeResize(residual_row, Eigen::NoChange);
			ekfom_data.z.conservativeResize(residual_row);
		}

	
    solve_time_3 += omp_get_wtime() - solve_start_;
	effct_feat_num += effect_num_k;
}
void h_model_output(state_output &s, Eigen::Matrix3d cov_p, Eigen::Matrix3d cov_R, esekfom::dyn_share_modified<double> &ekfom_data)
{
	bool match_in_map = false;
	VF(4) pabcd;
	pabcd.setZero();
	normvec->resize(time_seq[k]);
	std::vector<double> frame_feature_variances(time_seq[k],
	                                            min_observation_variance());
	ensureFrameSelectionCapacity(static_cast<std::size_t>(idx + time_seq[k] + 1));
	int effect_num_k = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		PointType &point_body_j  = feats_down_body->points[idx+j+1];
		PointType &point_world_j = feats_down_world->points[idx+j+1];
		pointBodyToWorld(&point_body_j, &point_world_j); 
		V3D p_body = pbody_list[idx+j+1];
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;
		{
			auto &points_near = Nearest_Points[idx+j+1];
			
            // ivox_->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS); // 
			
			if ((points_near.size() < NUM_MATCH_POINTS)) // || pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5)
			{
				point_selected_surf[idx+j+1] = false;
			}
				else
				{
					point_selected_surf[idx+j+1] = false;
					LocalFeaturePCA local_feature;
					LidarResidualType classified_type = LidarResidualType::Invalid;
					if (compute_local_pca(points_near, local_feature))
					{
						classified_type = classify_lidar_feature(local_feature, linearity_thr, planarity_thr);
					}
					if (classified_type == LidarResidualType::PointPlane &&
					    esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
					{
						float pd2 = fabs(pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3));

					if (p_norm > match_s * pd2 * pd2)
					{
						// point_selected_surf[i] = true;
						point_selected_surf[idx+j+1] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pabcd(3);
						const V3D point(point_world_j.x, point_world_j.y, point_world_j.z);
						const V3D normal(pabcd(0), pabcd(1), pabcd(2));
						const Eigen::Matrix<double, 6, 6> feature_cov =
						    compute_pca_direction_center_covariance(points_near,
						                                            local_feature, 0);
						frame_feature_variances[j] =
						    calc_plane_feature_observation_variance(
						        point, normal, local_feature.centroid, feature_cov);
						
						effect_num_k ++;
					}
				}  
			}
		}
	}
	if (effect_num_k == 0) 
	{
		ekfom_data.valid = false;
		return;
	}
	ekfom_data.M_Noise = 1.0;
	ekfom_data.h_x.resize(effect_num_k, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_num_k, 12);
	ekfom_data.z.resize(effect_num_k);
	int m = 0;
	for (int j = 0; j < time_seq[k]; j++)
	{
		// ekfom_data.converge = false;
		if(point_selected_surf[idx+j+1])
		{
			V3D norm_vec(normvec->points[j].x, normvec->points[j].y, normvec->points[j].z);
			if (extrinsic_est_en)
			{
				V3D p_body = pbody_list[idx+j+1];
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{   
				M3D point_crossmat = crossmat_oneset_list[idx+j+1];
				V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(m, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
			ekfom_data.z(m) = -norm_vec(0) * feats_down_world->points[idx+j+1].x -norm_vec(1) * feats_down_world->points[idx+j+1].y -norm_vec(2) * feats_down_world->points[idx+j+1].z-normvec->points[j].intensity;
			const V3D p_body = pbody_list[idx+j+1];
			const M3D point_cov_world = calcCurrentPointWorldCov(s, p_body);
			const double point_variance =
			    (norm_vec.transpose() * point_cov_world * norm_vec)(0, 0);
			const double residual_variance = std::max(
			    min_observation_variance(), frame_feature_variances[j] + point_variance);
			const double scale = 1.0 / std::sqrt(residual_variance);
			ekfom_data.h_x.row(m) *= scale;
			ekfom_data.z(m) *= scale;
			
			m++;
		}
	}
	effct_feat_num += effect_num_k;
}

void h_model_IMU_output(state_output &s, esekfom::dyn_share_modified<double> &ekfom_data)
{
    std::memset(ekfom_data.satu_check, false, 6);
	ekfom_data.z_IMU.block<3,1>(0, 0) = angvel_avr - s.omg - s.bg;
	ekfom_data.z_IMU.block<3,1>(3, 0) = acc_avr * G_m_s2 / acc_norm - s.acc - s.ba;
    ekfom_data.R_IMU << imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_omg_cov, imu_meas_acc_cov, imu_meas_acc_cov, imu_meas_acc_cov;
	if(check_satu)
	{
		if(fabs(angvel_avr(0)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[0] = true; 
			ekfom_data.z_IMU(0) = 0.0;
		}
		
		if(fabs(angvel_avr(1)) >= 0.99 * satu_gyro) 
		{
			ekfom_data.satu_check[1] = true;
			ekfom_data.z_IMU(1) = 0.0;
		}
		
		if(fabs(angvel_avr(2)) >= 0.99 * satu_gyro)
		{
			ekfom_data.satu_check[2] = true;
			ekfom_data.z_IMU(2) = 0.0;
		}
		
		if(fabs(acc_avr(0)) >= 0.99 * satu_acc)
		{
			ekfom_data.satu_check[3] = true;
			ekfom_data.z_IMU(3) = 0.0;
		}

		if(fabs(acc_avr(1)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[4] = true;
			ekfom_data.z_IMU(4) = 0.0;
		}

		if(fabs(acc_avr(2)) >= 0.99 * satu_acc) 
		{
			ekfom_data.satu_check[5] = true;
			ekfom_data.z_IMU(5) = 0.0;
		}
	}
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{    
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_normal(pi->normal_x,pi->normal_y,pi->normal_z);
    V3D p_global;
	V3D p_normal_global;//这是法向量
	if (extrinsic_est_en)
	{	
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (kf_output.x_.offset_R_L_I * p_body + kf_output.x_.offset_T_L_I) + kf_output.x_.pos;
			p_normal_global=kf_output.x_.rot * kf_output.x_.offset_R_L_I * p_normal;
		}
		else
		{
			p_global = kf_input.x_.rot * (kf_input.x_.offset_R_L_I * p_body + kf_input.x_.offset_T_L_I) + kf_input.x_.pos;
			p_normal_global = kf_input.x_.rot * kf_input.x_.offset_R_L_I * p_normal;
		} 
	}
	else
	{
		if (!use_imu_as_input)
		{
			p_global = kf_output.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_output.x_.pos; // .normalized()
			p_normal_global = kf_output.x_.rot * Lidar_R_wrt_IMU * p_normal;
		}
		else
		{
			p_global = kf_input.x_.rot * (Lidar_R_wrt_IMU * p_body + Lidar_T_wrt_IMU) + kf_input.x_.pos; // .normalized()
			Eigen::Matrix3d rotation_matrix = kf_input.x_.rot;
			// cout<<"rot"<<rotation_matrix<<"pos"<<kf_input.x_.pos<<"lidar_r_wre_imu"<<Lidar_R_wrt_IMU<<"lidar_T"<<Lidar_T_wrt_IMU<<"pglobal"<<p_global(2)<<endl;
			p_normal_global = kf_input.x_.rot * Lidar_R_wrt_IMU * p_normal;
			// cout<<"p_normal"<<p_normal(0)<<" "<<p_normal(1)<<" "<<p_normal(2)<<"p_normalglobal"<<p_normal_global(0)<<" "<<p_normal_global(1)<<" "<<p_normal_global(2)<<endl;
		}
	}

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
	po->normal_x=p_normal_global(0);
	po->normal_y=p_normal_global(1);
	po->normal_z=p_normal_global(2);



    po->intensity = pi->intensity;
}
