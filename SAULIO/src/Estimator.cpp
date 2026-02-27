// #include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include "Estimator.h"
#include <fstream>
#include <sys/stat.h>
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(20000, 1));
std::vector<int> time_seq;
std::vector<int> time_seq1;
std::vector<pointwithfeature> sourcecloudwithfeaturedown;
std::vector<pointwithfeature> sourcecloudwithfeature;//这是要点云流遍历取点的
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world_ground(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world_pillar(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world_roof(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world_beam(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world_facade(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_world_vertex(new PointCloudXYZI(20000, 1));

PointCloudXYZI::Ptr feats_down_oneset_body(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_oneset_world(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_bigupdate_world(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_bigupdate_body(new PointCloudXYZI(20000, 1));
PointCloudXYZI::Ptr feats_down_incre_world(new PointCloudXYZI(20000, 1));//更新map需要很多组点积累到一起

//lyb
std::vector<int> oneset_pointnumbers(20000);
std::vector<int> bigupdate_pointnumbers(20000);//这是存放bigupdate中有效的点对应的全部点的序号
std::vector<int> bigupdate_feature(20000);

std::vector<M3D> crossmat_effect_list(20000);
std::vector<M3D> crossmat_oneset_list(20000);
std::vector<M3D> crossmat_bigupdate_imu_list(30000);
std::vector<M3D> crossmat_bigupdate_body_list(50000);
std::vector<PointVector> Nearest_oneset_Points(20000); 
std::vector<PointVector> Nearest_incre_Points(20000);


std::vector<PointVector> Nearest_bigupdate_Points(20000);

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
static const char* kSmallUpdateStatsLogFile = "Log/small_update_stats.txt";

// 初始化 Log 目录和统计文件（在启动或首次使用时调用）
void init_small_update_stats_log()
{
	if (small_update_log_initialized)
	{
		return;
	}

	// 创建 Log 目录（若已存在则忽略错误）
	mkdir("Log", 0755);

	// 先清空/创建文件
	std::ofstream ofs(kSmallUpdateStatsLogFile, std::ios::out | std::ios::trunc);
	if (!ofs.is_open())
	{
		return;
	}
	ofs << "# small update stats log" << std::endl;
	small_update_log_initialized = true;
}

// 将一次统计结果写入 txt 文件
void log_small_update_stats_to_file(double start_time, double end_time,
                                    int count, double avg_interval)
{
	init_small_update_stats_log();
	std::ofstream ofs(kSmallUpdateStatsLogFile, std::ios::app);
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
	init_small_update_stats_log();
	std::ofstream ofs(kSmallUpdateStatsLogFile, std::ios::app);
	if (!ofs.is_open())
	{
		return;
	}
	ofs << "[pre_thresh] t=" << t << " s, pre_thresh_1=" << thresh << std::endl;
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
std::vector<V3D> pbody_oneset_list(20000);
std::vector<V3D> pbody_bigupdate_list(20000);
std::vector<V3D> pbody_list;
std::vector<PointVector> Nearest_Points; 
                 // localmap in ivox
std::shared_ptr<IVoxType> ivox_all = nullptr;                     // unified ivox for all points without feature segmentation

std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
bool point_selected_surf[10000] = {0};
bool point_selected_oneset_surf[10000]= {0};
std::vector<int> oneset_effect_feature(5000,0);
// std::vector<int> oneset_feature(10000,0);
bool point_selected_bigupdate_surf[30000] ={0};
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
	// normvec->resize(time_seq[k]);
	int effect_num_k = 0;
	//lyb
	total_oneset_num+=time_seq[k];//积累点数
    
	// 记录当前 set 在 oneset 序列中的起始下标，方便并行访问
	int base_oneset_idx = oneset_idx;

	// 第一步：并行计算最近邻和拟合平面，仅根据 cur_idx 写入数组，不修改计数器
    V3D prev_query_p_world;
    bool has_prev_query = false;

    // #ifdef MP_EN
    //     omp_set_num_threads(MP_PROC_NUM);
    //     #pragma omp parallel for  
    // #endif 
    // #ifdef MP_EN
    //     omp_set_num_threads(MP_PROC_NUM);
    //     #pragma omp parallel for  
	// #endif   
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
		
		double p_norm = p_body.norm();
		V3D p_world;
		p_world << point_world_j.x, point_world_j.y, point_world_j.z;

	    double t1=omp_get_wtime();
		PointVector points_near;
		// Modified: 不使用特征分割，统一从ivox_all中查找最近点
		ivox_all->GetClosestPoint(point_world_j, points_near, NUM_MATCH_POINTS, dis_thre);
        solve_time_1+=omp_get_wtime()-t1;

        // 统计相邻两次查询点之间的距离（在 world 坐标系下）
        if (has_prev_query) {
            double query_dist = (p_world - prev_query_p_world).norm();
            // 这里可以根据需要选择是直接打印还是累加到某个全局变量
            // 目前先简单打印，后续如果你需要做平均/直方图再改成累加
            std::cout << "query_dist: " << query_dist << std::endl;
        }
        prev_query_p_world = p_world;
        has_prev_query = true;

		Nearest_Points[idx+j+1]=points_near;
		Nearest_oneset_Points[cur_idx]=points_near;//
			
		if ((points_near.size() < NUM_MATCH_POINTS))
		{
			point_selected_oneset_surf[cur_idx] = false;//表示匹配失败
		}
		else
		{   
			// Modified: 使用esti_plane拟合平面，不再依赖点的法向量
			VF(4) pabcd_local;
			point_selected_oneset_surf[cur_idx] = false;
			if (esti_plane(pabcd_local, points_near, plane_thr))
			{
				float pd2 = pabcd_local(0) * point_world_j.x + pabcd_local(1) * point_world_j.y + pabcd_local(2) * point_world_j.z + pabcd_local(3);
				float s = 1 - 0.9 * fabs(pd2) / sqrt(p_norm);

				if (s > 0.9)
				{
					point_selected_oneset_surf[cur_idx] = true;
					normvec->points[cur_idx].x = pabcd_local(0);
					normvec->points[cur_idx].y = pabcd_local(1);
					normvec->points[cur_idx].z = pabcd_local(2);
					normvec->points[cur_idx].intensity = pd2;
				}
			}
		}
	}

	// 更新全局 oneset_idx
	oneset_idx = base_oneset_idx + time_seq[k];

	// 第二步：串行统计有效点，填充 oneset_pointnumbers 和各种计数
	for (int i = base_oneset_idx; i < base_oneset_idx + time_seq[k]; ++i)
	{
		if (point_selected_oneset_surf[i])
		{
			oneset_pointnumbers[effect_oneset_num] = i;
			effect_num_k ++;
			effect_oneset_num ++;
			effect_num_oneframe++;
		}
	}
	
	
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input += match_cost;
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
		double stand_coeff=1.2;//容忍指数
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
				last_small_check_time = time_current; // 更新检查时间
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
		const double pre_thresh_coeff = 0.9; // 平滑系数，可根据需要调整到(0,1)
		pre_thresh_1 = pre_thresh_coeff * pre_thresh_1 + (1.0 - pre_thresh_coeff) * max_eigenvalue;
	} else {
		// 时间还没到0.02秒，继续积累点，不更新
		// 但是如果是本帧最后一个segment，即使时间没到，也要去畸变后加入大更新
		if (is_last_segment_in_frame) {
			should_update = false; // 不进行小更新，但会继续执行去畸变
			// 统计本次与上次检查之间的时间差
			observed_small_update_interval_sum += time_since_last_update;
			observed_small_update_interval_count++;
			check_and_reset_small_update_stats();
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
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(effect_oneset_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_oneset_num, 12);
	ekfom_data.z.resize(effect_oneset_num);
	//fprintf(fp_debug,"effect_num: %d\n",effect_oneset_num);
	cout<<"effect_oneset"<<effect_oneset_num<<endl;
	cout<<"total_oneset_num"<<total_oneset_num<<endl;

	#ifdef MP_EN
	omp_set_num_threads(MP_PROC_NUM);
	#pragma omp parallel for  
	#endif  
	for (int j = 0; j < effect_oneset_num; j++)
	{
		// ekfom_data.converge = false;
		
		// if(point_selected_oneset_surf[j])
		// {
		
		V3D norm_vec(normvec->points[oneset_pointnumbers[j]].x, normvec->points[oneset_pointnumbers[j]].y, normvec->points[oneset_pointnumbers[j]].z);
		// if(oneset_effect_feature[j]==1){
			if (extrinsic_est_en)
			{
					V3D p_body = pbody_oneset_list[oneset_pointnumbers[j]];//
					M3D p_crossmat, p_imu_crossmat;
					p_crossmat << SKEW_SYM_MATRX(p_body);
					V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
					p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
					V3D C(s.rot.transpose() * norm_vec);
					V3D A(p_imu_crossmat * C);
					V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
					ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
			}
			else
			{       			
					// V3D p_imu =s.rot.inverse()*(pworld_oneset_list[j]-s.pos);
					// M3D point_crossmat;
					// point_crossmat<< SKEW_SYM_MATRX(p_imu);//这里导致的问题,这里是自己的imu坐标系下面,不要去畸变

					M3D point_crossmat = crossmat_oneset_list[oneset_pointnumbers[j]];
					V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
					V3D A(point_crossmat * C);
					ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			}
				ekfom_data.z(j) = -normvec->points[oneset_pointnumbers[j]].intensity;
				// cout<<"z(j)"<<ekfom_data.z(j)<<endl;
		// }
		
			// m++;
	// }
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
	if (normvec->size()<total_oneset_num){
	normvec->resize(total_oneset_num);}
	int effect_num_k = 0;
	effect_oneset_num=0;
	//lyb
	//total_oneset_num+=time_seq[k];
    // #ifdef MP_EN
    //     omp_set_num_threads(MP_PROC_NUM);
    //     #pragma omp parallel for  
	// #endif  
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
				continue;
			}
			else
			{
				points_near=Nearest_oneset_Points[j];
				// Modified: 使用esti_plane拟合平面，不再依赖点的法向量
				VF(4) pabcd;
				point_selected_oneset_surf[j] = false;
				if (esti_plane(pabcd, points_near, plane_thr))
				{
					float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);
					float s = 1 - 0.9 * fabs(pd2) / sqrt(p_norm);

					if (s > 0.9)
					{
						point_selected_oneset_surf[j] = true;
						normvec->points[j].x = pabcd(0);
						normvec->points[j].y = pabcd(1);
						normvec->points[j].z = pabcd(2);
						normvec->points[j].intensity = pd2;
					}
				}
			}

		}
		//oneset_idx++;
	}
    for (int i = 0; i <total_oneset_num; i++)
    {
        if (point_selected_oneset_surf[i])
        {
            // laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            // corr_normvect->points[effct_feat_num] = normvec->points[i];
            effect_num_k++;
			oneset_pointnumbers[effect_oneset_num]=i;
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
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(effect_oneset_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_oneset_num, 12);
	ekfom_data.z.resize(effect_oneset_num);
	//fprintf(fp_debug,"effect_num: %d\n",effect_oneset_num);
	// int m = 0;
	#ifdef MP_EN
	omp_set_num_threads(MP_PROC_NUM);
	#pragma omp parallel for  
	#endif  
	for (int j = 0; j < effect_oneset_num; j++)
	{
		// ekfom_data.converge = false;

		V3D norm_vec(normvec->points[oneset_pointnumbers[j]].x, normvec->points[oneset_pointnumbers[j]].y, normvec->points[oneset_pointnumbers[j]].z);
			
		if (extrinsic_est_en)
		{
				V3D p_body = pbody_list[oneset_pointnumbers[j]];//有
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat << SKEW_SYM_MATRX(p_body);
				V3D point_imu = s.offset_R_L_I * p_body + s.offset_T_L_I;
				p_imu_crossmat << SKEW_SYM_MATRX(point_imu);
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
		}
		else
		{       			
				// V3D p_imu =s.rot.inverse()*(pworld_oneset_list[j]-s.pos);
				// M3D point_crossmat;
				// point_crossmat<< SKEW_SYM_MATRX(p_imu);//这里导致的问题,这里是自己的imu坐标系下面,不要去畸变

				M3D point_crossmat = crossmat_oneset_list[oneset_pointnumbers[j]];
				V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		}
		ekfom_data.z(j) = -normvec->points[oneset_pointnumbers[j]].intensity;					
				
			
		
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
	if (normvec->size()< bigupdate_num)//现在是一帧进行一次大更新
		{normvec->resize(bigupdate_num);}
	int effect_num_k = 0;

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
			continue;
		}
		// Modified: 使用esti_plane拟合平面，不再依赖点的法向量
		VF(4) pabcd;
		point_selected_bigupdate_surf[j] = false;
		if (esti_plane(pabcd, points_near, plane_thr))
		{
			float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);
			float s = 1 - 0.9 * fabs(pd2) / sqrt(p_norm);

			if (s > 0.9)
			{
				point_selected_bigupdate_surf[j] = true;
				normvec->points[j].x = pabcd(0);
				normvec->points[j].y = pabcd(1);
				normvec->points[j].z = pabcd(2);
				normvec->points[j].intensity = pd2;
			}
		}

		
	}
		//oneset_idx++;
	
    for (int i = 0; i<bigupdate_num; i++)
    {
        if (point_selected_bigupdate_surf[i])
        {
            // laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            // corr_normvect->points[effct_feat_num] = normvec->points[i];
            effect_num_k++;
			bigupdate_pointnumbers[bigupdate_effect_num]=i;//我这里是采用的一个vector把有效点对应到原来的点，voxelmap是有一个单独的数据结构，把有效点之后计算需要的数据push进去
            bigupdate_effect_num++;
			
        }
    }
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input_bigupdate += match_cost;

	if (bigupdate_effect_num ==0) 
	{
		ekfom_data.valid = false;
		return;
	}
	double solve_start_  = omp_get_wtime();
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(bigupdate_effect_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(bigupdate_effect_num, 12);
	// ekfom_data.h_x.bottomRows(bigupdate_effect_num-bigupdate_effect_start_idx).setZero();//后几行置零
	// ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_oneset_num, 12);
	ekfom_data.z.resize(bigupdate_effect_num);
	//fprintf(fp_debug,"effect_num: %d\n",effect_oneset_num);
	// int m = 0;
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for  
	#endif 
	for (int j = 0; j < bigupdate_effect_num; j++)
	{
		// ekfom_data.converge = false;

		V3D norm_vec(normvec->points[bigupdate_pointnumbers[j]].x, normvec->points[bigupdate_pointnumbers[j]].y, normvec->points[bigupdate_pointnumbers[j]].z);
			
		if (extrinsic_est_en)
		{
				// V3D p_body = pbody_list[point_effect_numbers[j]];//有问题
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat = crossmat_bigupdate_body_list[j];
				p_imu_crossmat = crossmat_bigupdate_imu_list[j];
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
		}
		else
		{       			
				// V3D p_imu =s.rot.inverse()*(pworld_oneset_list[j]-s.pos);
				// M3D point_crossmat;
				// point_crossmat<< SKEW_SYM_MATRX(p_imu);//这里导致的问题,这里是自己的imu坐标系下面,不要去畸变

				M3D point_crossmat = crossmat_bigupdate_imu_list[bigupdate_pointnumbers[j]];
				V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		}
		ekfom_data.z(j) = -normvec->points[bigupdate_pointnumbers[j]].intensity;					
		
			// m++;
	
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
	if (normvec->size()< bigupdate_num)
		{normvec->resize(bigupdate_num);}
	int effect_num_k = 0;
	bigupdate_effect_num=0;

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
			continue;
		} 
		// Modified: 使用esti_plane拟合平面，不再依赖点的法向量
		VF(4) pabcd;
		point_selected_bigupdate_surf[j] = false;
		if (esti_plane(pabcd, points_near, plane_thr))
		{
			float pd2 = pabcd(0) * point_world_j.x + pabcd(1) * point_world_j.y + pabcd(2) * point_world_j.z + pabcd(3);
			float s = 1 - 0.9 * fabs(pd2) / sqrt(p_norm);

			if (s > 0.9)
			{
				point_selected_bigupdate_surf[j] = true;
				normvec->points[j].x = pabcd(0);
				normvec->points[j].y = pabcd(1);
				normvec->points[j].z = pabcd(2);
				normvec->points[j].intensity = pd2;
			}
		}

		
	}
		//oneset_idx++;
	
    for (int i = 0; i<bigupdate_num; i++)
    {
        if (point_selected_bigupdate_surf[i])
        {
            // laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            // corr_normvect->points[effct_feat_num] = normvec->points[i];
            effect_num_k++;
			bigupdate_pointnumbers[bigupdate_effect_num]=i;//我这里是采用的一个vector把有效点对应到原来的点，voxelmap是有一个单独的数据结构，把有效点之后计算需要的数据push进去
            bigupdate_effect_num++;
			
        }
    }
	double match_cost = omp_get_wtime() - match_start;
	match_time  += match_cost;
	match_time_h_model_input_bigupdate_1 += match_cost;
	//lyb

	double solve_start_  = omp_get_wtime();
	ekfom_data.M_Noise = laser_point_cov;
	ekfom_data.h_x.resize(bigupdate_effect_num, 12);
	ekfom_data.h_x = Eigen::MatrixXd::Zero(bigupdate_effect_num, 12);
	// ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_oneset_num, 12);
	ekfom_data.z.resize(bigupdate_effect_num);
	//fprintf(fp_debug,"effect_num: %d\n",effect_oneset_num);
	// int m = 0;
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for  
	#endif 
	for (int j = 0; j < bigupdate_effect_num; j++)
	{
		// ekfom_data.converge = false;

		V3D norm_vec(normvec->points[bigupdate_pointnumbers[j]].x, normvec->points[bigupdate_pointnumbers[j]].y, normvec->points[bigupdate_pointnumbers[j]].z);
			
		if (extrinsic_est_en)
		{
				// V3D p_body = pbody_list[point_effect_numbers[j]];//有问题
				M3D p_crossmat, p_imu_crossmat;
				p_crossmat = crossmat_bigupdate_body_list[j];
				p_imu_crossmat = crossmat_bigupdate_imu_list[j];
				V3D C(s.rot.transpose() * norm_vec);
				V3D A(p_imu_crossmat * C);
				V3D B(p_crossmat * s.offset_R_L_I.transpose() * C);
				ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
		}
		else
		{       			
				// V3D p_imu =s.rot.inverse()*(pworld_oneset_list[j]-s.pos);
				// M3D point_crossmat;
				// point_crossmat<< SKEW_SYM_MATRX(p_imu);//这里导致的问题,这里是自己的imu坐标系下面,不要去畸变

				M3D point_crossmat = crossmat_bigupdate_imu_list[bigupdate_pointnumbers[j]];
				V3D C(s.rot.transpose() * norm_vec); // conjugate().normalized()
				V3D A(point_crossmat * C);
				ekfom_data.h_x.block<1, 12>(j, 0) << norm_vec(0), norm_vec(1), norm_vec(2), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		}
		ekfom_data.z(j) = -normvec->points[bigupdate_pointnumbers[j]].intensity;					
			
			// m++;
	
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
				if (esti_plane(pabcd, points_near, plane_thr)) //(planeValid)
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
	ekfom_data.M_Noise = laser_point_cov;
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