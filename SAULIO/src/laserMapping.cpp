// #include <so4_math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include "li_initialization.h"
#include <malloc.h>
#include <Eigen/Geometry>
#include "voxel_map_util.hpp"
#include "dataio.hpp"
#include "cfilter.hpp"
#include "cregistration.hpp"
#include "map_manager.h"
#include "../include/utility.hpp"
#include "map_viewer.h"

// #include <cv_bridge/cv_bridge.h>
// #include "matplotlibcpp.h"
// #include <ros/console.h>
using namespace lo;
using namespace std;

#define PUBFRAME_PERIOD (20)

const float MOV_THRESHOLD = 1.5f;

string root_dir = ROOT_DIR;
string pos_debug_dir = root_dir + "/Log/debug.txt";
FILE *fp_debug = fopen(pos_debug_dir.c_str(), "w");
string ratio_debug_dir = root_dir + "/Log/ratio.txt";
FILE *ratio_debug = fopen(ratio_debug_dir.c_str(), "w");
string pos_debug_dir_1 = root_dir + "/Log/debug_small.txt";
FILE *fp_debug_1 = fopen(pos_debug_dir_1.c_str(), "w");

int time_log_counter = 0; //, publish_count = 0;
bool start_predict = false;
bool init_map = false, flg_first_scan = true;

// Time Log Variables
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;
double publish_time = 0;
// 分函数统计的 match 时间（每帧重置，在各自函数内累加）
double match_time_h_model_input = 0.0;
double match_time_h_model_input_1 = 0.0;
double match_time_h_model_input_bigupdate = 0.0;
double match_time_h_model_input_bigupdate_1 = 0.0;
bool flg_reset = false, flg_exit = false;
int iter_num = 0;
double oneset_time = 0;
double last_small_check_time = 0.0; // 上次小更新的时间，用于每0.02秒检查一次特征值
double bigupdate_start_time = 0.0; // 大更新开始的时间，用于判断是否达到0.1s
bool is_last_segment_in_frame = false; // 标识是否是本帧最后一个segment
// surf feature in map
PointCloudXYZI::Ptr feats_raw(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_world(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_publish(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_publish_first(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());
std::deque<PointCloudXYZI::Ptr> depth_feats_world;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
int ii = 0;
V3D euler_cur;

nav_msgs::Path path1;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;

std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;

std::vector<M3D> var_down_body;

Eigen::Matrix4d lidar_to_imu;
deque<sensor_msgs::Imu> imu_list;


void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang;
    if (!use_imu_as_input)
    {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    }
    else
    {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }

    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2)); // Angle
    if (use_imu_as_input)
    {
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2));             // Pos
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                          // omega
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2));             // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                          // Acc
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));                // Bias_g
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));                // Bias_a
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a
    }
    else
    {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2));             // Pos
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                             // omega
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2));             // Vel
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                                             // Acc
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));                // Bias_g
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));                // Bias_a
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2)); // Bias_a
    }
    fprintf(fp, "\r\n");
    fflush(fp);
}

inline void dump_lio_state_to_log_1(FILE *fp)
{
    // V3D rot_ang;
    // if (!use_imu_as_input)
    // {
    //     rot_ang = SO3ToEuler(kf_output.x_.rot);
    // }
    // else
    // {
    //     rot_ang = SO3ToEuler(kf_input.x_.rot);
    // }
    MD(4, 4)
    T;
    T.block<3, 3>(0, 0) = kf_input.x_.rot;
    T.block<3, 1>(0, 3) = kf_input.x_.pos;
    fprintf(fp, "%lf ", lidar_end_time);
    Eigen::Matrix3d T1 = T.block<3, 3>(0, 0);
    Eigen::Quaterniond q(T1);
    // fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    // if (use_imu_as_input
    fprintf(fp, "%lf %lf %lf %lf %lf %lf %lf\n", T(0, 3),
            T(1, 3), T(2, 3), q.x(), q.y(), q.z(), q.w());
    fflush(fp);

    // fflush(fp);
}

void pointBodyLidarToIMU(PointType const *const pi, PointType *const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
        }
        else
        {
            p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    }
    else
    {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void MapIncremental()
{
    PointVector points_to_add;
    int cur_pts = feats_down_world->size();
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i)
    {
        /* decide if need add to map */
        PointType &point_world = feats_down_world->points[i];
        if (!Nearest_Points[i].empty())
        {
            const PointVector &points_near = Nearest_Points[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;
            bool need_add = true;
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++)
            {
                Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
                if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.z()) < 0.5 * filter_size_map_min)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
            {
                points_to_add.emplace_back(point_world);
            }
        }
        else
        {
            points_to_add.emplace_back(point_world);
        }
    }
    // ivox_->AddPoints(points_to_add);
}
void MapIncremental_1(const PointCloudXYZI::Ptr &feats_down_world, const std::vector<PointVector> &Nearest_incre_Points, std::shared_ptr<IVoxType> &ivox, int total_num)
{
    PointVector points_to_add;
    int cur_pts = total_num;
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i)
    {
        /* decide if need add to map */
        PointType &point_world = feats_down_world->points[i];
        if (!Nearest_incre_Points[i].empty())
        {
            const PointVector &points_near = Nearest_incre_Points[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;
            bool need_add = true;
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++)
            {
                Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
                if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.z()) < 0.5 * filter_size_map_min)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
            {
                points_to_add.emplace_back(point_world);
            }
        }
        else
        {
            points_to_add.emplace_back(point_world);
        }
    }
    ivox->AddPoints(points_to_add);
}
void publish_init_map(const ros::Publisher &pubLaserCloudFullRes)
{
    int size_init_map = init_feats_world->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*init_feats_world, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}
void publish_init_map_1(const ros::Publisher &pubLaserCloudFullRes, const PointCloudXYZI::Ptr &cloud)
{
    int size_init_map = cloud->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*cloud, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}
void publish_feature_cloud(const ros::Publisher &pubLaserCloudFullRes, const PointCloudXYZI::Ptr &cloud)
{
    int size_init_map = cloud->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*cloud, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}
void publish_all_cloud(const ros::Publisher &pubLaserCloudFullRes, const PointCloudXYZI::Ptr &cloud)
{
    int size_init_map = cloud->size();

    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*cloud, laserCloudmsg);

    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; //
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        // publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {

        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
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
void publish_frame_world_1(const ros::Publisher &pubLaserCloudFullRes)
{
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudWorld;

        if (dense_publish_en)
        {
            // 原逻辑：使用整帧去畸变点云 + 当前状态转换到世界系
            PointCloudXYZI::Ptr laserCloudFullRes(feats_undistort);
            int size = laserCloudFullRes->points.size();
            laserCloudWorld.reset(new PointCloudXYZI(size, 1));
            for (int i = 0; i < size; i++)
            {
                pointBodyToWorld(&laserCloudFullRes->points[i],
                                 &laserCloudWorld->points[i]);
            }
        }
        else
        {
            // 轻量模式：直接使用增量地图点（已在世界坐标系）
            int size = total_incre_num;
            laserCloudWorld.reset(new PointCloudXYZI(size, 1));
            for (int i = 0; i < size; i++)
            {
                laserCloudWorld->points[i] = feats_down_incre_world->points[i];
            }
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = ros::Time().fromSec(publish_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        // publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = total_incre_num;
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_incre_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_incre_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_incre_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_incre_world->points[i].intensity;
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
void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    PointCloudXYZI::Ptr laserCloudIMUBody;
    if (dense_publish_en)
    {
        int size = feats_undistort->points.size();
        laserCloudIMUBody.reset(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++)
        {
            pointBodyLidarToIMU(&feats_undistort->points[i],
                                &laserCloudIMUBody->points[i]);
        }
    }
    else
    {
        // 关闭 dense 发布时，使用世界系下的增量点云，先转换回 IMU-body 系
        int size = total_incre_num;
        laserCloudIMUBody.reset(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++)
        {
            PointType pt_world = feats_down_incre_world->points[i];
            V3D p_world(pt_world.x, pt_world.y, pt_world.z);
            V3D p_body;
            if (!use_imu_as_input)
            {
                p_body = kf_output.x_.rot.conjugate() * (p_world - kf_output.x_.pos);
            }
            else
            {
                p_body = kf_input.x_.rot.conjugate() * (p_world - kf_input.x_.pos);
            }
            laserCloudIMUBody->points[i].x = p_body(0);
            laserCloudIMUBody->points[i].y = p_body(1);
            laserCloudIMUBody->points[i].z = p_body(2);
            laserCloudIMUBody->points[i].intensity = pt_world.intensity;
        }
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

template <typename T>
void set_posestamp(T &out)
{
    if (!use_imu_as_input)
    {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        Eigen::Quaterniond q(kf_output.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
    else
    {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        Eigen::Quaterniond q(kf_input.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);

    pubOdomAftMapped.publish(odomAftMapped);

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
}
void publish_odometry_1(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(publish_time);
    }
    set_posestamp(odomAftMapped.pose.pose);

    pubOdomAftMapped.publish(odomAftMapped);

    // static tf::TransformBroadcaster br;
    // tf::Transform                   transform;
    // tf::Quaternion                  q;
    // transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
    //                                 odomAftMapped.pose.pose.position.y, \
    //                                 odomAftMapped.pose.pose.position.z));
    // q.setW(odomAftMapped.pose.pose.orientation.w);
    // q.setX(odomAftMapped.pose.pose.orientation.x);
    // q.setY(odomAftMapped.pose.pose.orientation.y);
    // q.setZ(odomAftMapped.pose.pose.orientation.z);
    // transform.setRotation( q );
    // br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body") );
}
void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path1.poses.emplace_back(msg_body_pose);
        pubPath.publish(path1);
    }
}

void publish_path_1(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = ros::Time().fromSec(publish_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path1.poses.emplace_back(msg_body_pose);
        pubPath.publish(path1);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh("~");
    ros::AsyncSpinner spinner(0);
    spinner.start();
    readParameters(nh);

    cout << "lidar_type: " << lidar_type << endl;

    ivox_all = std::make_shared<IVoxType>(ivox_options_);   // unified ivox for all points without feature segmentation
    path1.header.stamp = ros::Time().fromSec(lidar_end_time);
    path1.header.frame_id = "camera_init";

    /*** variables definition for counting ***/
    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;

    // memset(point_selected_surf, true, sizeof(point_selected_surf));
    std::fill(point_selected_surf.begin(), point_selected_surf.end(), 1);
    // 这种固定downsize的逻辑可以不要了
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    lidar_to_imu.block<3, 3>(0, 0) = Lidar_R_wrt_IMU;
    lidar_to_imu.block<3, 1>(0, 3) = Lidar_T_wrt_IMU;
    lidar_to_imu(3, 0) = 0;
    lidar_to_imu(3, 1) = 0;
    lidar_to_imu(3, 2) = 0;
    lidar_to_imu(3, 3) = 1;

    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
        else
        {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }

    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;
    double epsi[24] = {0.001};
    fill(epsi, epsi + 24, 0.001); // 从epsi填充到epsi+22 也就是全部数组置0.001
    kf_input_undistort.init_dyn_share_modified_2h(get_f_input, df_dx_input, df_dw_input, h_model_input, h_model_input_1, h_model_input_bigupdate, h_model_input_bigupdate_1, 1, epsi);
    kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input, df_dw_input, h_model_input, h_model_input_1, h_model_input_bigupdate, h_model_input_bigupdate_1, 1, epsi);
    kf_output.init_dyn_share_modified_3h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
    Eigen::Matrix<double, 24, 24> P_init; // = MD(18, 18)::Identity() * 0.1;
    reset_cov(P_init);
    kf_input.change_P(P_init);
    Eigen::Matrix<double, 30, 30> P_init_output; // = MD(24, 24)::Identity() * 0.01;
    reset_cov_output(P_init_output);
    kf_output.change_P(P_init_output);
    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();
    /*** debug record ***/

    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");
    open_file();

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 1000);
    ros::Publisher pubLaserCloudFullRes_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 1000);
    ros::Publisher pubLaserCloudFullorigin = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_origin", 1000);
    // ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
    // ("/cloud_effected", 1000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 1000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 1000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 1000);

    // ros::Publisher plane_pub = nh.advertise<visualization_msgs::Marker>
    // ("/planner_normal", 1000);
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate loop_rate(500);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit)
            break;
        ros::spinOnce();
        if (sync_packages(Measures))
        {
            if (flg_reset)
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                feats_undistort.reset(new PointCloudXYZI()); // 去畸变后的点云
                feats_raw.reset(new PointCloudXYZI());       // 原始点云
                if (use_imu_as_input)
                {
                    // state_in = kf_input.get_x();
                    state_in = state_input();
                    kf_input.change_P(P_init);
                }
                else
                {
                    // state_out = kf_output.get_x();
                    state_out = state_output();
                    kf_output.change_P(P_init_output);
                }
                flg_first_scan = true;
                is_first_frame = true;
                flg_reset = false;
                init_map = false;


            }

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                first_lidar_end_time = Measures.lidar_last_time;
                flg_first_scan = false;
                if (first_imu_time < 1)
                {
                    first_imu_time = imu_next.header.stamp.toSec();
                    printf("first imu time: %f\n", first_imu_time);
                }
                time_current = 0.0;
                if (imu_en)
                {
                    // imu_next = *(imu_deque.front());
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
                    // kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
                    // kf_output.x_.acc *= -1;

                    {
                        while (Measures.lidar_beg_time > imu_next.header.stamp.toSec()) // if it is needed for the new map?
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty())
                            {
                                break;
                            }
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            // imu_deque.pop();
                        }
                    }
                }
                else
                {
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);  // _init);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity); //_init);
                    kf_output.x_.acc << VEC_FROM_ARRAY(gravity);     //_init);
                    kf_output.x_.acc *= -1;
                    p_imu->imu_need_init_ = false;
                    // p_imu->after_imu_init_ = true;
                }
                G_m_s2 = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
            }

            double t0, t1, t2, t3, t4, t5, t6, match_start, solve_start, icp_time;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            // 本帧内各观测模型函数的 match 时间统计清零
            match_time_h_model_input = 0.0;
            match_time_h_model_input_1 = 0.0;
            match_time_h_model_input_bigupdate = 0.0;
            match_time_h_model_input_bigupdate_1 = 0.0;
            solve_time_1 = 0;
            solve_time_2 = 0;
            solve_time_3 = 0;
            icp_time = 0;
            t0 = omp_get_wtime();

            /*** downsample the feature points in a scan ***/
            if (!init_map)
            { // 如果没有建图，则第一帧去畸变
                if (!use_imu_as_input)
                { // 没有imu数据，第一帧就不去畸变，但是要排序
                    p_imu->Process_2(Measures, kf_output, feats_undistort);

                    sort(feats_undistort->points.begin(), feats_undistort->points.end(), time_list); // 将raw点云排序
                }
                else
                {
                    p_imu->Process_1(Measures, kf_input, feats_undistort, acc_cov_input, gyr_cov_input, b_gyr_cov, b_acc_cov); // 里面已经排了序了,且已经去了畸变
                }
            }
            else // 非第一帧
            {
                p_imu->Process(Measures, feats_undistort); // 非第一帧，没有去畸变
                sort(feats_undistort->points.begin(), feats_undistort->points.end(), time_list);
                // sort(feats_undistort->points.begin(), feats_undistort->points.end(), time_list); //不去畸变，然后提取特征
                // undistort是一帧原始点云，用来semantic，raw然后用来做点云流，这两个要保证数量一样，降采样也要保证删除相同的序号，可能要想想怎么加速这一块
            }
            if (!p_imu->after_imu_init_) // !p_imu->UseLIInit &&
            {
                if (!p_imu->imu_need_init_)
                {
                    V3D tmp_gravity;
                    if (imu_en)
                    {
                        tmp_gravity = -p_imu->mean_acc / p_imu->mean_acc.norm() * G_m_s2;
                    }
                    else
                    {
                        tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                        p_imu->after_imu_init_ = true;
                    }
                    // V3D tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                    M3D rot_init;
                    p_imu->Set_init(tmp_gravity, rot_init);
                    kf_input.x_.rot = rot_init;
                    //lyb把kfinput的世界坐标系从enu变成了第一帧imu，那么这里的rot也就该是单位矩阵
                    kf_input.x_.gravity = tmp_gravity;
                    kf_input.x_.rot = Eye3d;  
                    kf_output.x_.rot = rot_init;
                    // kf_input.x_.rot; //.normalize();
                    // kf_output.x_.rot; //.normalize();
                    kf_output.x_.acc = -rot_init.transpose() * kf_output.x_.gravity; // 是以东北天为原点左边，因此需要第一帧确定这个rot，通过重力来确定
                }
                else
                {
                    continue;
                }
            }
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            Eigen::Matrix3d rot_matri = kf_input.x_.rot; // pose_lo就用imu系，在mulls中用的是lidar系，这个也好转换
            Eigen::Vector3d pos_vec = kf_input.x_.pos;

            // 创建一个 4x4 的齐次变换矩阵
            Eigen::Matrix4d homogenous_matrix;

            // 旋转部分 (3x3)
            homogenous_matrix.block<3, 3>(0, 0) = rot_matri;

            // 平移部分 (3x1)
            homogenous_matrix.block<3, 1>(0, 3) = pos_vec;

            // 最后一行和最后一列填充为 [0 0 0 1]
            homogenous_matrix(3, 0) = 0;
            homogenous_matrix(3, 1) = 0;
            homogenous_matrix(3, 2) = 0;
            homogenous_matrix(3, 3) = 1;


            if (feats_down_body->empty() || (feats_down_body == NULL))
            {
                ROS_WARN("No point, skip _a_this scan!\n");
                continue;
            }
            // Modified: 不使用特征提取，直接使用原始点云feats_undistort
            // 不再需要构建sourcecloudwithfeature，直接使用feats_undistort
            cout << "sourcesize" << feats_down_body->points.size() << endl;
            feats_down_world->resize(feats_down_body->points.size());
            
            Eigen::Matrix4d trans = homogenous_matrix * lidar_to_imu;
                
            // 将所有点转换到world坐标系
#ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
            for (size_t i = 0; i < feats_down_body->points.size(); i++)
            {
                PointType &point_body_j = feats_down_body->points[i];
                PointType &point_world_j = feats_down_world->points[i];
                pointBodyToWorld(&point_body_j, &point_world_j);
            }
 
            publish_all_cloud(pubLaserCloudFullorigin, feats_down_world);
            // Modified: 不再发布特征类型的点云，因为不再进行特征分割


            if (init_map) // 没初始化地图，就不用运行下面去排序时间，直接构造就行了
            {
                // Modified: 直接使用feats_undistort，不需要sourcecloudwithfeature
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
                time_seq = time_compressing<int>(feats_down_body); // 直接使用feats_down_body进行时间压缩
                feats_down_size = feats_down_body->points.size();
                cout << "featdwnsize" << feats_down_size << endl;
         
            }

            /*** initialize the map ***/
            if (!init_map) // 利用第一帧先建图
            {
                // Modified: 不使用特征分割，直接使用所有点
                
                cout << "featssize" << feats_down_world->size() << endl;
                
                Eigen::Matrix4d trans = homogenous_matrix * lidar_to_imu;


                if (feats_down_world->size() < init_map_size)
                {
                    init_map = false;
                }
                else
                {
                    // Modified: 直接使用所有点构建初始地图，不区分特征类型
                    PointVector all_points_to_add;
                    all_points_to_add.reserve(feats_down_world->points.size());
                    for (const auto& pt : feats_down_world->points) {
                        all_points_to_add.push_back(pt);
                    }
                    ivox_all->AddPoints(all_points_to_add); // 添加所有点到统一的ivox中

                    publish_init_map_1(pubLaserCloudMap, feats_down_world); //(pubLaserCloudFullRes);
                    // lyb
                    publish_time = lidar_end_time;//发布时间
                    oneset_time = lidar_end_time;
                    //// init_feats_world.reset(new PointCloudXYZI());
                    init_map = true;
                }
                continue;
            }
            // 下面开始第一帧的逻辑
            t1 = omp_get_wtime();

            /*** ICP and Kalman filter update ***/
            feats_down_world->resize(feats_down_size); // 这是sourceclouddown的大小

            Nearest_Points.resize(feats_down_size);

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            // crossmat_list.reserve(feats_down_size);
            pbody_list.resize(feats_down_size); // 将其转换为v3d
// pbody_ext_list.reserve(feats_down_size);
#ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
            for (size_t i = 0; i < feats_down_size; i++)
            {
                // Modified: 直接使用feats_down_body，不需要sourcecloudwithfeature
                V3D point_this(feats_down_body->points[i].x,
                               feats_down_body->points[i].y,
                               feats_down_body->points[i].z);
                pbody_list[i] = point_this;

            }

            if (!use_imu_as_input)
            {
                bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/
                if (time_seq.size() > 0)
                {
                    double pcl_beg_time = Measures.lidar_beg_time;
                    idx = -1;
                    for (k = 0; k < time_seq.size(); k++)
                    {
                        PointType &point_body = feats_down_body->points[idx + time_seq[k]];

                        time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                        if (is_first_frame)
                        {
                            if (imu_en)
                            {
                                while (time_current > imu_next.header.stamp.toSec())
                                {
                                    imu_deque.pop_front();
                                    if (imu_deque.empty())
                                        break;
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                }
                                angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                                acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            }
                            is_first_frame = false;
                            imu_upda_cov = true;
                            time_update_last = time_current;
                            time_predict_last_const = time_current;
                        }
                        if (imu_en && !imu_deque.empty())
                        {
                            bool last_imu = imu_next.header.stamp.toSec() == imu_deque.front()->header.stamp.toSec();
                            while (imu_next.header.stamp.toSec() < time_predict_last_const && !imu_deque.empty())
                            {
                                if (!last_imu)
                                {
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                    break;
                                }
                                else
                                {
                                    imu_deque.pop_front();
                                    if (imu_deque.empty())
                                        break;
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                }
                            }
                            bool imu_comes = time_current > imu_next.header.stamp.toSec();
                            while (imu_comes)
                            {
                                imu_upda_cov = true;
                                angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                                acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                                /*** covariance update ***/
                                double dt = imu_next.header.stamp.toSec() - time_predict_last_const;
                                kf_output.predict(dt, Q_output, input_in, true, false);
                                time_predict_last_const = imu_next.header.stamp.toSec(); // big problem

                                {
                                    double dt_cov = imu_next.header.stamp.toSec() - time_update_last;

                                    if (dt_cov > 0.0)
                                    {
                                        time_update_last = imu_next.header.stamp.toSec();
                                        double propag_imu_start = omp_get_wtime();

                                        kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                        propag_time += omp_get_wtime() - propag_imu_start;
                                        double solve_imu_start = omp_get_wtime();
                                        kf_output.update_iterated_dyn_share_IMU();
                                        solve_time += omp_get_wtime() - solve_imu_start;
                                    }
                                }
                                imu_deque.pop_front();
                                if (imu_deque.empty())
                                    break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                imu_comes = time_current > imu_next.header.stamp.toSec();
                            }
                        }
                        if (flg_reset)
                        {
                            break;
                        }

                        double dt = time_current - time_predict_last_const;
                        double propag_state_start = omp_get_wtime();
                        if (!prop_at_freq_of_imu)
                        {
                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0)
                            {
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                time_update_last = time_current;
                            }
                        }
                        kf_output.predict(dt, Q_output, input_in, true, false);
                        propag_time += omp_get_wtime() - propag_state_start;
                        time_predict_last_const = time_current;
                        double t_update_start = omp_get_wtime();

                        if (feats_down_size < 1)
                        {
                            ROS_WARN("No point, skip _b_ this scan!\n");
                            idx += time_seq[k];
                            continue;
                        }

                        if (!kf_output.update_iterated_dyn_share_modified())
                        {
                            idx = idx + time_seq[k];
                            continue;
                        }
                        solve_start = omp_get_wtime();

                        if (publish_odometry_without_downsample)
                        {
                            /******* Publish odometry *******/

                            publish_odometry(pubOdomAftMapped);
                            if (runtime_pos_log)
                            {
                                euler_cur = SO3ToEuler(kf_output.x_.rot);
                                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose()
                                         << " " << kf_output.x_.omg.transpose() << " " << kf_output.x_.acc.transpose() << " " << kf_output.x_.gravity.transpose() << " " << kf_output.x_.bg.transpose() << " " << kf_output.x_.ba.transpose() << " " << feats_undistort->points.size() << endl;
                            }
                        }

                        oneset_time = time_current;
                        for (int j = 0; j < time_seq[k]; j++)
                        {
                            PointType &point_body_j = feats_down_body->points[idx + j + 1];
                            PointType &point_world_j = feats_down_world->points[idx + j + 1];
                            pointBodyToWorld(&point_body_j, &point_world_j);
                        }

                        solve_time += omp_get_wtime() - solve_start;

                        update_time += omp_get_wtime() - t_update_start;
                        idx += time_seq[k];
                        // cout << "pbp output effect feat num:" << effct_feat_num << endl;
                    }
                }
                else
                {
                    if (!imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());

                        while (imu_next.header.stamp.toSec() > time_current && ((imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)))
                        { // >= ?
                            if (is_first_frame)
                            {
                                {
                                    {
                                        while (imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)
                                        {
                                            // meas.imu.emplace_back(imu_deque.front()); should add to initialization
                                            imu_deque.pop_front();
                                            if (imu_deque.empty())
                                                break;
                                            imu_last = imu_next;
                                            imu_next = *(imu_deque.front());
                                        }
                                    }
                                    break;
                                }
                                angvel_avr << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;

                                acc_avr << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                                imu_upda_cov = true;
                                time_update_last = time_current;
                                time_predict_last_const = time_current;

                                is_first_frame = false;
                            }
                            time_current = imu_next.header.stamp.toSec();

                            if (!is_first_frame)
                            {
                                double dt = time_current - time_predict_last_const;
                                {
                                    double dt_cov = time_current - time_update_last;
                                    if (dt_cov > 0.0)
                                    {
                                        kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                        time_update_last = time_current;
                                    }
                                    kf_output.predict(dt, Q_output, input_in, true, false);
                                }

                                time_predict_last_const = time_current;

                                angvel_avr << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                                acc_avr << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;
                                // acc_avr_norm = acc_avr * G_m_s2 / acc_norm;
                                kf_output.update_iterated_dyn_share_IMU();
                                imu_deque.pop_front();
                                if (imu_deque.empty())
                                    break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                            else
                            {
                                imu_deque.pop_front();
                                if (imu_deque.empty())
                                    break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                        }
                    }
                }
            }
            else
            {
                bool imu_prop_cov = false;
                effct_feat_num = 0;

                if (time_seq.size() > 0)
                {
                    double pcl_beg_time = Measures.lidar_beg_time;
                    idx = -1;//整个一帧的点的序号
                    kf_input_undistort.x_ = kf_input.x_; // 将状态变量同步
                    imu_list.push_back(imu_last);        // 上一帧的最后一个imu
                    for (k = 0; k < time_seq.size(); k++)
                    {
                        // Modified: 直接使用feats_undistort，不需要sourcecloudwithfeature
                        PointType &point_body = feats_down_body->points[idx + time_seq[k]];
                        time_current = point_body.curvature / 1000.0 + pcl_beg_time; // curvature记录的是时间
                        if (is_first_frame)
                        {
                            while (time_current > imu_next.header.stamp.toSec())
                            {
                                imu_list.push_back(*imu_deque.front());
                                imu_deque.pop_front();
                                if (imu_deque.empty())
                                    break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                            imu_prop_cov = true;

                            is_first_frame = false;
                            t_last = time_current;
                            time_update_last = time_current;
                            {
                                input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                                input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                                input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                            }
                        }

                        while (time_current > imu_next.header.stamp.toSec()) // && !imu_deque.empty())
                        {
                            imu_list.push_back(*imu_deque.front());
                            imu_deque.pop_front();

                            input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                            double dt = imu_last.header.stamp.toSec() - t_last;

                            double dt_cov = imu_last.header.stamp.toSec() - time_update_last;

                            if (dt_cov > 0.0)
                            {

                                kf_input.predict(dt_cov, Q_input, input_in, false, true);
                                time_update_last = imu_last.header.stamp.toSec(); // time_current;
                            }

                            kf_input.predict(dt, Q_input, input_in, true, false);

                            t_last = imu_last.header.stamp.toSec();
                            imu_prop_cov = true;

                            if (imu_deque.empty())
                                break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            // imu_upda_cov = true;
                        }
                        if (flg_reset)
                        {
                            break;
                        }

                        double dt = time_current - t_last;
                        t_last = time_current;
                        double propag_start = omp_get_wtime();

                        if (!prop_at_freq_of_imu)
                        {
                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0)
                            {
                                kf_input.predict(dt_cov, Q_input, input_in, false, true);
                                time_update_last = time_current;
                            }
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false);

                        propag_time += omp_get_wtime() - propag_start;

                        double t_update_start = omp_get_wtime();
                        // dump_lio_state_to_log_1(fp);
                        if (feats_down_size < 1)
                        {
                            ROS_WARN("No point, skip _c_this scan!\n");

                            idx += time_seq[k];
                            continue;
                        }

                        // Modified: 设置是否是本帧最后一个segment的标志
                        is_last_segment_in_frame = (k == time_seq.size() - 1);
                        
                        // Modified: 设置小更新的迭代次数
                        kf_input.maximum_iter = small_update_max_iterations;
                        bool successornot = kf_input.update_iterated_dyn_share_modified_1();
                        if (!successornot && k != time_seq.size() - 1) // 看到这里,如果到了本帧的最后一次，就不要continue了
                        {
                            idx = idx + time_seq[k];
                            continue;
                        }

                        // solve_start = omp_get_wtime();

                        if (publish_odometry_without_downsample)
                        {
                            /******* Publish odometry *******/

                            publish_odometry(pubOdomAftMapped);
                            if (runtime_pos_log)
                            {
                                euler_cur = SO3ToEuler(kf_input.x_.rot);
                                // fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose()
                                //          << " " << kf_input.x_.bg.transpose() << " " << kf_input.x_.ba.transpose() << " " << kf_input.x_.gravity.transpose() << " " << feats_undistort->points.size() << endl;
                            }
                        }


                        // 在填充前确保 bigupdate 缓冲区有足够容量，避免越界
                        // ensureBigupdateCapacity(static_cast<std::size_t>(bigupdate_num + total_oneset_num));
                        std::size_t required_bigupdate_size = static_cast<std::size_t>(bigupdate_num + total_oneset_num);
                        if (feats_down_bigupdate_body->size() < required_bigupdate_size)
                        {
                            feats_down_bigupdate_body->resize(required_bigupdate_size);
                        }
                        if (feats_down_bigupdate_world->size() < required_bigupdate_size)
                        {
                            feats_down_bigupdate_world->resize(required_bigupdate_size);
                        }
                        if (Nearest_bigupdate_Points.size() < required_bigupdate_size)
                        {
                            Nearest_bigupdate_Points.resize(required_bigupdate_size);
                        }
                        // if (bigupdate_feature.size() < required_bigupdate_size)
                        // {
                        //     bigupdate_feature.resize(required_bigupdate_size);
                        // }
                        if (successornot)
                        {
#ifdef MP_EN
                            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif // 并行
                            for (int j = 0; j < total_oneset_num; j++)
                            {
                                std::size_t target_idx = static_cast<std::size_t>(bigupdate_num + j);
                                // if (target_idx >= feats_down_bigupdate_body->size() ||
                                //     target_idx >= feats_down_bigupdate_world->size() ||
                                //     target_idx >= Nearest_bigupdate_Points.size())
                                // {
                                //     std::ofstream ofs((root_dir + "/Log/bigupdate_bounds_debug.txt").c_str(), std::ios::app);
                                //     if (ofs.is_open())
                                //     {
                                //         ofs << "[laserMapping_success_copy] t=" << time_current
                                //             << " target_idx=" << target_idx
                                //             << " bigupdate_num=" << bigupdate_num
                                //             << " total_oneset_num=" << total_oneset_num
                                //             << " body_size=" << feats_down_bigupdate_body->size()
                                //             << " world_size=" << feats_down_bigupdate_world->size()
                                //             << " nearest_size=" << Nearest_bigupdate_Points.size()
                                //             << std::endl;
                                //     }
                                //     std::abort();
                                // }
                                PointType &point_body_j = feats_down_oneset_body->points[j];
                                PointType &point_world_j = feats_down_oneset_world->points[j];
                                pointBodyToWorld(&point_body_j, &point_world_j);
                                feats_down_bigupdate_body->points[bigupdate_num + j] = point_body_j;
                                feats_down_bigupdate_world->points[bigupdate_num + j] = point_world_j;

                                Nearest_bigupdate_Points[bigupdate_num + j] = Nearest_oneset_Points[j];
                                // bigupdate_feature[bigupdate_num + j] = oneset_feature[j];
                                // Nearest_incre_Points[total_incre_num]=Nearest_oneset_Points[j];
                                // total_incre_num++;//积累要更新的点,一次更新需要积累的有：点数，nearestpoints，featsdownworld
                            }
                        }
                        else
                        { // 到了这一帧的最后一次了，即使小更新条件不满足，也要去畸变后加入大更新
#ifdef MP_EN
                            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif // 并行
                            for (int j = 0; j < total_oneset_num; j++)
                            {
                                std::size_t target_idx = static_cast<std::size_t>(bigupdate_num + j);
                                // if (target_idx >= feats_down_bigupdate_body->size() ||
                                //     target_idx >= feats_down_bigupdate_world->size() ||
                                //     target_idx >= Nearest_bigupdate_Points.size())
                                // {
                                //     std::ofstream ofs((root_dir + "/Log/bigupdate_bounds_debug.txt").c_str(), std::ios::app);
                                //     if (ofs.is_open())
                                //     {
                                //         ofs << "[laserMapping_fail_copy] t=" << time_current
                                //             << " target_idx=" << target_idx
                                //             << " bigupdate_num=" << bigupdate_num
                                //             << " total_oneset_num=" << total_oneset_num
                                //             << " body_size=" << feats_down_bigupdate_body->size()
                                //             << " world_size=" << feats_down_bigupdate_world->size()
                                //             << " nearest_size=" << Nearest_bigupdate_Points.size()
                                //             << std::endl;
                                //     }
                                //     std::abort();
                                // }
                                // Modified: 即使小更新不成功，也要将点去畸变后加入大更新
                                // 在h_model_input中，如果是最后一帧，即使特征值检查不通过，也会执行去畸变代码（443-477行）
                                // 所以feats_down_oneset_body中的点已经去畸变过了
                                PointType &point_body_j = feats_down_oneset_body->points[j];
                                PointType &point_world_j = feats_down_oneset_world->points[j];
                                // 确保world坐标是最新的（不用重新计算一次，在h_model_input中已经计算过了）
                                // pointBodyToWorld(&point_body_j, &point_world_j);
                                feats_down_bigupdate_body->points[bigupdate_num + j] = point_body_j;
                                feats_down_bigupdate_world->points[bigupdate_num + j] = point_world_j;
                                Nearest_bigupdate_Points[bigupdate_num + j] = Nearest_oneset_Points[j];
                                // bigupdate_feature[bigupdate_num + j] = oneset_feature[j];
                            }
                        }
                        // cout<<"totalonesetnum "<<total_oneset_num<<endl;
                        // cout<<"bigupdate_num  "<<bigupdate_num<<endl;
                        bigupdate_num += total_oneset_num;
                        cout << "bigupdate_num" << bigupdate_num << endl;


                        bigupdate_effect_num += effect_oneset_num;
                        cout << "setseffect_num" << bigupdate_effect_num << endl;

                        // Modified: 大更新机制 - 到达一帧的最后一个segment时触发
                        // 如果是最后一帧，触发大更新
                        if (k == time_seq.size() - 1)
                        {
                            // Modified: 设置大更新的迭代次数
                            kf_input.maximum_iter = bigupdate_max_iterations;
                            kf_input.update_iterated_dyn_share_modified_bigupdate();


                            std::size_t required_incre_size = static_cast<std::size_t>(total_incre_num + bigupdate_num);
                            if (feats_down_incre_world->size() < required_incre_size)
                            {
                                feats_down_incre_world->resize(required_incre_size);
                            }
                            if (Nearest_incre_Points.size() < required_incre_size)
                            {
                                Nearest_incre_Points.resize(required_incre_size);
                            }
                            for (int j = 0; j < bigupdate_num; j++)
                            {
                                PointType &point_body_j = feats_down_bigupdate_body->points[j];
                                // cout<<"body"<<point_body_j.x<<" "<<point_body_j.y<<" "<<point_body_j.z<<endl;
                                PointType &point_world_j = feats_down_bigupdate_world->points[j];
                                pointBodyToWorld(&point_body_j, &point_world_j);

                                // Save all points to unified point cloud without feature segmentation
                                feats_down_incre_world->points[total_incre_num] = point_world_j;
                                Nearest_incre_Points[total_incre_num] = Nearest_bigupdate_Points[j];
                                total_incre_num++; // 积累要更新的点,一次更新需要积累的有：点数，nearestpoints，featsdownworld
                            }

                            // Modified: add all points to unified ivox
                            t4 = omp_get_wtime();
                            MapIncremental_1(feats_down_incre_world, Nearest_incre_Points, ivox_all, total_incre_num);
                            t5 = omp_get_wtime();
                            bigupdate_effect_num = 0;
                          
                            bigupdate_num = 0;
                  
                            bigupdate_effect_num = 0;
                            // total_incre_num=0;
                            publish_time = time_current;
                            if (dense_publish_en)
                            {
                                p_imu->UndistortPcl_end(Measures, imu_list, kf_input_undistort, *feats_undistort,
                                                        acc_cov_input, gyr_cov_input, b_gyr_cov, b_acc_cov);
                            }
  
                            if (path_en)
                                publish_path_1(pubPath);
                            if (scan_pub_en || pcd_save_en)
                                publish_frame_world_1(pubLaserCloudFullRes);
                            if (scan_pub_en && scan_body_pub_en)
                                publish_frame_body(pubLaserCloudFullRes_body);
                            ii++;
                            if (!publish_odometry_without_downsample)
                            {
                                publish_odometry_1(pubOdomAftMapped);
                            }
                            dump_lio_state_to_log_1(fp);
                            // cout<<"ddd: "<<kf_input.P_<<endl;
                            // }
                            total_incre_num = 0;

                            imu_list.clear();

                            // }
                            // }
                        }
                        // solve_time += omp_get_wtime() - solve_start;
                        // icp_time+=omp_get_wtime() - t2;
                        update_time += omp_get_wtime() - t_update_start;
                        // if (time_current-publish_time>0.1){

                        // }
                        idx = idx + time_seq[k];
                        oneset_idx = 0;
                        oneset_startupdate_idx = 0;
                        oneset_num = 0;
                        effect_oneset_num = 0;
                        total_oneset_num = 0;
                        if_effect_start = false;
                        // dump_lio_state_to_log_1(fp);
                    }
                }
                else
                {
                    if (!imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        while (imu_next.header.stamp.toSec() > time_current && ((imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)))
                        { // >= ?
                            if (is_first_frame)
                            {
                                {
                                    {
                                        while (imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)
                                        {
                                            imu_deque.pop_front();
                                            if (imu_deque.empty())
                                                break;
                                            imu_last = imu_next;
                                            imu_next = *(imu_deque.front());
                                        }
                                    }

                                    break;
                                }
                                imu_prop_cov = true;

                                t_last = time_current;
                                time_update_last = time_current;
                                input_in.gyro << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                                input_in.acc << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                                input_in.acc = input_in.acc * G_m_s2 / acc_norm;

                                is_first_frame = false;
                            }
                            time_current = imu_next.header.stamp.toSec();

                            if (!is_first_frame)
                            {
                                double dt = time_current - t_last;

                                double dt_cov = time_current - time_update_last;
                                if (dt_cov > 0.0)
                                {
                                    // kf_input.predict(dt_cov, Q_input, input_in, false, true);
                                    time_update_last = imu_next.header.stamp.toSec(); // time_current;
                                }
                                // kf_input.predict(dt, Q_input, input_in, true, false);

                                t_last = imu_next.header.stamp.toSec();

                                input_in.gyro << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                                input_in.acc << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;
                                input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                                imu_deque.pop_front();
                                if (imu_deque.empty())
                                    break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                            else
                            {
                                imu_deque.pop_front();
                                if (imu_deque.empty())
                                    break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                            }
                        }
                    }
                }
            }
            // M3D rot_cur_lidar;
            // {
            //     rot_cur_lidar = state.rot_end;
            // }
            // euler_cur = RotMtoEuler(rot_cur_lidar);
            // geoQuat = tf::createQuaternionMsgFromRollPitchYaw
            //                     (euler_cur(0), euler_cur(1), euler_cur(2));
            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample)
            {
                publish_odometry_1(pubOdomAftMapped);
            }

            effect_num_oneframe = 0;
            // t5 = omp_get_wtime();
            t6 = omp_get_wtime();
            /******* Publish points *******/
            // if (path_en)                         publish_path(pubPath);
            // if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFullRes);
            // if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);
            // cout<<"time1:"<<solve_time_1<<endl;
            // cout<<"time2:"<<solve_time_2<<endl;
            // cout<<"time3:"<<solve_time_3<<endl;
            // 统计本帧 h_model_input 中 GetClosestPoint 的总耗时（solve_time_1 在 h_model_input 内部累加）
            std::cout << "[Frame h_model_input GetClosestPoint] solve_time_1 = "
                      << solve_time_1 << std::endl;
            // 打印当前帧四个观测模型函数各自的 match 时间占比（同一函数多次调用已在内部累加）
            std::cout << "[Frame match time] total_match_time = " << match_time
                      << ", h_model_input = " << match_time_h_model_input
                      << ", h_model_input_1 = " << match_time_h_model_input_1
                      << ", h_model_input_bigupdate = " << match_time_h_model_input_bigupdate
                      << ", h_model_input_bigupdate_1 = " << match_time_h_model_input_bigupdate_1
                      << std::endl;
            /*** Debug variables Logging ***/
            if (runtime_pos_log)
            {
                frame_num++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t6 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time_3 / frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t6 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n", t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t4, aver_time_consu, aver_time_icp, aver_time_propag);
                if (!publish_odometry_without_downsample)

                {
                    if (!use_imu_as_input)
                    {
                        euler_cur = SO3ToEuler(kf_output.x_.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose()
                                 << " " << kf_output.x_.omg.transpose() << " " << kf_output.x_.acc.transpose() << " " << kf_output.x_.gravity.transpose() << " " << kf_output.x_.bg.transpose() << " " << kf_output.x_.ba.transpose() << " " << feats_undistort->points.size() << endl;
                    }
                    else
                    {
                        euler_cur = SO3ToEuler(kf_input.x_.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose()
                                 << " " << kf_input.x_.bg.transpose() << " " << kf_input.x_.ba.transpose() << " " << kf_input.x_.gravity.transpose() << " " << feats_undistort->points.size() << endl;
                    }
                }
                // dump_lio_state_to_log_1(fp);
            }
        }
        // cout<<endl<<"ii:"<<ii<<endl;
        status = ros::ok();
        loop_rate.sleep();
    }
    //--------------------------save map-----------------------------------
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    fout_out.close();
    fout_imu_pbp.close();
    return 0;
}
