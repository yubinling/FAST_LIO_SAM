#include "IMU_Processing.h"

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

const bool time_list1(pointwithfeature &x, pointwithfeature &y) {return (x.point.curvature < y.point.curvature);};
void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_vel_scale = scaler;
}

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true)
{
  imu_en = true;
  init_iter_num = 1;
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
  after_imu_init_ = false;
  state_cov.setIdentity();
  last_imu_.reset(new sensor_msgs::Imu());
  angvel_last     = Zero3d;

}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, 0.0);
  mean_gyr      = V3D(0, 0, 0);
  imu_need_init_    = true;
  init_iter_num     = 1;
  after_imu_init_   = false;
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  angvel_last     = Zero3d;
  time_last_scan = 0.0;
}

void ImuProcess::Set_init(Eigen::Vector3d &tmp_gravity, Eigen::Matrix3d &rot)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  // V3D tmp_gravity = - mean_acc / mean_acc.norm() * G_m_s2; // state_gravity;
  M3D hat_grav;
  hat_grav << 0.0, gravity_(2), -gravity_(1),
              -gravity_(2), 0.0, gravity_(0),
              gravity_(1), -gravity_(0), 0.0;
  double align_norm = (hat_grav * tmp_gravity).norm() / gravity_.norm() / tmp_gravity.norm();
  double align_cos = gravity_.transpose() * tmp_gravity;
  align_cos = align_cos / gravity_.norm() / tmp_gravity.norm();
  if (align_norm < 1e-6)
  {
    if (align_cos > 1e-6)
    {
      rot = Eye3d;
    }
    else
    {
      rot = -Eye3d;
    }
  }
  else
  {
    V3D align_angle = hat_grav * tmp_gravity / (hat_grav * tmp_gravity).norm() * acos(align_cos); 
    rot = Exp(align_angle(0), align_angle(1), align_angle(2));
  }
}

void ImuProcess::IMU_init(const MeasureGroup &meas, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    N ++;
  }
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_input, 24, input_ikfom> &kf_state, PointCloudXYZI &pcl_out,double acc_cov_input, double gyr_cov_input, double b_gyr_cov ,double b_acc_cov)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  deque<sensor_msgs::Imu::ConstPtr> v_imu;
  v_imu.push_back(meas.imu.back());
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_last_time;
  double imu_time = imu_deque.front()->header.stamp.toSec();
  while(imu_time<pcl_end_time){
    v_imu.emplace_back(imu_deque.front());
    imu_last = imu_next;
    imu_deque.pop_front();
    if(imu_deque.empty()) break;
    imu_time = imu_deque.front()->header.stamp.toSec(); // can be changed
    imu_next = *(imu_deque.front());
  }
  //imu_list是为了一帧结束后去畸变准备的，这里赋个初值
  // imu_list.emplace_back(*v_imu.back());
  // v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  //pcl_out_1shi
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_input imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;
  Eigen::Matrix<double, 24, 24> Q;
	Q.setZero();
	Q.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	Q.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	Q.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	Q.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * 9.81/ mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    
    // Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    // Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    // Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    // Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    // 这里应该判断是否有GPS measurements， 如果有GPS， 那么先传播到GPS，然后更新state，再重新从头开始向前传播
    // 看一下r2live的实现
    kf_state.predict(dt, Q, in,true,true);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.gravity[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in,true,true);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.inverse() * (imu_state.rot.inverse() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::UndistortPcl_end(const MeasureGroup &meas,const deque<sensor_msgs::Imu> &imu_list, esekfom::esekf<state_input, 24, input_ikfom> &kf_state, PointCloudXYZI &pcl_out,double acc_cov_input, double gyr_cov_input, double b_gyr_cov ,double b_acc_cov)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  // deque<sensor_msgs::Imu::ConstPtr> v_imu;
  // v_imu.push_back(meas.imu.back());
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_last_time;
  // double imu_time = imu_deque.front()->header.stamp.toSec();
  // while(imu_time<pcl_end_time){
  //   v_imu.emplace_back(imu_deque.front());
  //   imu_last = imu_next;
  //   imu_deque.pop_front();
  //   if(imu_deque.empty()) break;
  //   imu_time = imu_deque.front()->header.stamp.toSec(); // can be changed
  //   imu_next = *(imu_deque.front());
  // }
  //imu_list是为了一帧结束后去畸变准备的，这里赋个初值
  // imu_list.emplace_back(*v_imu.back());
  // v_imu.push_front(last_imu_);
  const double &imu_beg_time = imu_list.front().header.stamp.toSec();
  const double &imu_end_time = imu_list.back().header.stamp.toSec();

  
  /*** sort point clouds by offset time ***/
  // pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  //pcl_out_1shi
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_input imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;
  Eigen::Matrix<double, 24, 24> Q;
	Q.setZero();
	Q.block<3, 3>(3, 3).diagonal() << gyr_cov_input, gyr_cov_input, gyr_cov_input;
	Q.block<3, 3>(12, 12).diagonal() << acc_cov_input, acc_cov_input, acc_cov_input;
	Q.block<3, 3>(15, 15).diagonal() << b_gyr_cov, b_gyr_cov, b_gyr_cov;
	Q.block<3, 3>(18, 18).diagonal() << b_acc_cov, b_acc_cov, b_acc_cov;
  input_ikfom in;
  for (auto it_imu = imu_list.begin(); it_imu < (imu_list.end() - 1); it_imu++)
  {
    auto &&head = it_imu;
    auto &&tail = it_imu + 1;
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head.header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * 9.81/ mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail.header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    
    // Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    // Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    // Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    // Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    // 这里应该判断是否有GPS measurements， 如果有GPS， 那么先传播到GPS，然后更新state，再重新从头开始向前传播
    // 看一下r2live的实现
    kf_state.predict(dt, Q, in,true,false);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.gravity[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in,true,false);
  
  imu_state = kf_state.get_x();
  // last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate = imu_state.offset_R_L_I.inverse() * (imu_state.rot.inverse() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
void ImuProcess::Process(const MeasureGroup &meas, PointCloudXYZI::Ptr cur_pcl_un_)
{  
  if (imu_en)
  {
    if(meas.imu.empty())  return;

    if (imu_need_init_)
    {
      
      {
        /// The very first lidar frame
        IMU_init(meas, init_iter_num);

        imu_need_init_ = true;

        if (init_iter_num > MAX_INI_COUNT)
        {
          ROS_INFO("IMU Initializing: %.1f %%", 100.0);
          imu_need_init_ = false;
          *cur_pcl_un_ = *(meas.lidar);
        }
        // *cur_pcl_un_ = *(meas.lidar);
      }
      return;
    }
    if (!after_imu_init_) after_imu_init_ = true;
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
  else
  {
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
}

void ImuProcess::Process_1(const MeasureGroup &meas, esekfom::esekf<state_input, 24, input_ikfom> &kf_state,PointCloudXYZI::Ptr cur_pcl_un_,double acc_cov_input, double gyr_cov_input, double b_gyr_cov ,double b_acc_cov)
{  
  if (imu_en)
  {
    if(meas.imu.empty())  return;

    if (imu_need_init_)
    {
      
      {
        /// The very first lidar frame
        IMU_init(meas, init_iter_num);

        imu_need_init_ = true;

        if (init_iter_num > MAX_INI_COUNT)
        {
          ROS_INFO("IMU Initializing: %.1f %%", 100.0);
          imu_need_init_ = false;
          *cur_pcl_un_ = *(meas.lidar);
        }
        // *cur_pcl_un_ = *(meas.lidar);
      }
      return;
    }
    UndistortPcl(meas, kf_state, *cur_pcl_un_,acc_cov_input,gyr_cov_input, b_gyr_cov ,b_acc_cov);
    if (!after_imu_init_) after_imu_init_ = true;
    
    return;
  }
  else
  {
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
}

void ImuProcess::Process_2(const MeasureGroup &meas, esekfom::esekf<state_output, 30, input_ikfom> &kf_state,PointCloudXYZI::Ptr cur_pcl_un_)
{  
  if (imu_en)
  {
    if(meas.imu.empty())  return;

    if (imu_need_init_)
    {
      
      {
        /// The very first lidar frame
        IMU_init(meas, init_iter_num);

        imu_need_init_ = true;

        if (init_iter_num > MAX_INI_COUNT)
        {
          ROS_INFO("IMU Initializing: %.1f %%", 100.0);
          imu_need_init_ = false;
          *cur_pcl_un_ = *(meas.lidar);
        }
        // *cur_pcl_un_ = *(meas.lidar);
      }
      return;
    }
    if (!after_imu_init_) after_imu_init_ = true;
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
  else
  {
    *cur_pcl_un_ = *(meas.lidar);
    return;
  }
}