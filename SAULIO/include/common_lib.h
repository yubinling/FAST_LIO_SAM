#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <../include/IKFoM/IKFoM_toolkit/esekfom/esekfom.hpp>
#include <queue>
#include <array>
#include <cmath>
#include <map_manager.h>
using namespace std;
using namespace Eigen;


typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_input,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, bg))
((vect3, ba))
((vect3, gravity))
);

MTK_BUILD_MANIFOLD(state_output,
((vect3, pos))
((SO3, rot))
((SO3, offset_R_L_I))
((vect3, offset_T_L_I))
((vect3, vel))
((vect3, omg))
((vect3, acc))
((vect3, gravity))
((vect3, bg))
((vect3, ba))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_input,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

MTK_BUILD_MANIFOLD(process_noise_output,
((vect3, vel))
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

extern esekfom::esekf<state_input, 24, input_ikfom> kf_input;
extern esekfom::esekf<state_input, 24, input_ikfom> kf_input_undistort;
extern esekfom::esekf<state_output, 30, input_ikfom> kf_output;

#define PBWIDTH 30
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"

#define PI_M (3.14159265358)
// #define G_m_s2 (9.81)         // Gravaty const in GuangDong/China
#define DIM_STATE (24)      // Dimension of states (Let Dim(SO(3)) = 3)
#define DIM_PROC_N (12)      // Dimension of process noise (Let Dim(SO(3)) = 3)
#define CUBE_LEN  (6.0)
#define LIDAR_SP_LEN    (2)
#define INIT_COV   (0.0001)
#define NUM_MATCH_POINTS    (5)
#define MAX_MEAS_DIM        (10000)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define VEC_FROM_ARRAY_SIX(v)        v[0],v[1],v[2],v[3],v[4],v[5]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define CONSTRAIN(v,min,max)     ((v>min)?((v<max)?v:max):min)
#define ARRAY_FROM_EIGEN(mat)    mat.data(), mat.data() + mat.rows() * mat.cols()
#define STD_VEC_FROM_EIGEN(mat)  vector<decltype(mat)::Scalar> (mat.data(), mat.data() + mat.rows() * mat.cols())
#define DEBUG_FILE_DIR(name)     (string(string(ROOT_DIR) + "Log/"+ name))

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZRGB     PointTypeRGB;
typedef pcl::PointCloud<PointType>    PointCloudXYZI;
typedef pcl::PointCloud<PointTypeRGB> PointCloudXYZRGB;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;
typedef Vector3f V3F;
typedef Matrix3f M3F;

extern double ranging_cov;
extern double angle_cov;
extern double laser_point_cov;

#define MD(a,b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a,b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>

enum class LidarResidualType
{
    Invalid = 0,
    PointPlane,
    PointLine
};

// 保存近邻点 PCA 的结果，用来判断这一组近邻更像线特征还是面特征。
struct LocalFeaturePCA
{
    V3D centroid = V3D::Zero();
    V3D eigen_values = V3D::Zero();      // ascending order: small, medium, large
    M3D eigen_vectors = M3D::Identity();
    double linearity = 0.0;
    double planarity = 0.0;
};

// 统一保存一次匹配的结果；点面产生 1 行残差，点线产生 2 行垂线方向残差。
struct LidarMatchResult
{
    LidarResidualType type = LidarResidualType::Invalid;
    VF(4) plane_coeff = VF(4)::Zero();
    V3D line_center = V3D::Zero();
    V3D line_direction = V3D::Zero();
    std::array<V3D, 2> residual_normals = {V3D::Zero(), V3D::Zero()};
    std::array<double, 2> residuals = {0.0, 0.0};
    std::array<double, 2> feature_variances = {0.0, 0.0};
    int residual_count = 0;
    double residual = 0.0;
};

const M3D Eye3d(M3D::Identity());
const M3F Eye3f(M3F::Identity());
const V3D Zero3d(0, 0, 0);
const V3F Zero3f(0, 0, 0);

struct Pose6D
{
    double offset_time;
    double acc[3], gyr[3], vel[3], pos[3], rot[9];
};


template<typename T>
auto set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g, \
                const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return move(rot_kp);
}
struct MeasureGroup     // Lidar data and imu dates for the curent process
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        lidar_last_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_last_time;
    PointCloudXYZI::Ptr lidar;
    deque<sensor_msgs::Imu::ConstPtr> imu;
};

template <typename T>
T calc_dist(PointType p1, PointType p2){
    T d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

template <typename T>
T calc_dist(Eigen::Vector3d p1, PointType p2){
    T d = (p1(0) - p2.x) * (p1(0) - p2.x) + (p1(1) - p2.y) * (p1(1) - p2.y) + (p1(2) - p2.z) * (p1(2) - p2.z);
    return d;
}

template<typename T>
std::vector<int> time_compressing(const PointCloudXYZI::Ptr &point_cloud)
//lyb
//std::vector<int> time_compressing(const std::vector<pointwithfeature> &sourcecloud)
{
  int points_size = point_cloud->points.size();
  int j = 0;
  std::vector<int> time_seq;
  // time_seq.clear();
  time_seq.reserve(points_size);
  for(int i = 0; i < points_size - 1; i++)
  {
    j++;
    if (point_cloud->points[i+1].curvature > point_cloud->points[i].curvature)
    {
      time_seq.emplace_back(j);
      j = 0;
    }
  }
//   if (j == 0)
//   {
//     time_seq.emplace_back(1);
//   }
//   else
  {
    time_seq.emplace_back(j+1);
  }
  return time_seq;
}

template<typename T>
std::vector<int> time_compressing1(const std::vector<pointwithfeature> &sourcecloud)
{
  int points_size = sourcecloud.size();
  int j = 0;
  std::vector<int> time_seq;
  // time_seq.clear();
  time_seq.reserve(points_size);
  for(int i = 0; i < points_size - 1; i++)
  {
    j++;
    if (sourcecloud[i+1].point.curvature > sourcecloud[i].point.curvature)
    {
      time_seq.emplace_back(j);
      j = 0;
    }
  }
//   if (j == 0)
//   {
//     time_seq.emplace_back(1);
//   }
//   else
  {
    time_seq.emplace_back(j+1);
  }
  return time_seq;
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec:  normalized x0
*/
template<typename T>
bool esti_normvector(Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold, const int &point_num)
{
    MatrixXf A(point_num, 3);
    MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);
    
    for (int j = 0; j < point_num; j++)
    {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold)
        {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

inline double min_observation_variance()
{
    return std::max(1e-8, laser_point_cov * 1e-3);
}

// VoxelMap 风格的单个激光点测量协方差：沿射线方向用测距噪声，
// 垂直射线的切平面方向用角度噪声，并随距离放大。
inline M3D calc_lidar_point_cov(const V3D &point, const double range_inc,
                                const double degree_inc)
{
    const double range = point.norm();
    if (!std::isfinite(range) || range < 1e-9)
    {
        return M3D::Identity() * std::max(min_observation_variance(), laser_point_cov);
    }

    const double range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    const double angle_sigma = std::sin(degree_inc * PI_M / 180.0);
    direction_var << angle_sigma * angle_sigma, 0.0, 0.0, angle_sigma * angle_sigma;

    V3D direction = point / range;
    M3D direction_hat;
    direction_hat << 0.0, -direction(2), direction(1),
                     direction(2), 0.0, -direction(0),
                    -direction(1), direction(0), 0.0;

    V3D base_vector1;
    const double eps = 1e-6;
    if (std::abs(direction(2)) > eps)
    {
        base_vector1 = V3D(1.0, 1.0, -(direction(0) + direction(1)) / direction(2));
    }
    else if (std::abs(direction(1)) > eps)
    {
        base_vector1 = V3D(1.0, -(direction(0) + direction(2)) / direction(1), 1.0);
    }
    else if (std::abs(direction(0)) > eps)
    {
        base_vector1 = V3D(-(direction(1) + direction(2)) / direction(0), 1.0, 1.0);
    }
    else
    {
        base_vector1 = V3D::UnitX();
    }
    base_vector1.normalize();

    V3D base_vector2 = base_vector1.cross(direction);
    if (base_vector2.norm() < eps)
    {
        base_vector2 = direction.unitOrthogonal();
    }
    base_vector2.normalize();

    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0),
         base_vector1(1), base_vector2(1),
         base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;

    M3D cov = direction * range_var * direction.transpose() +
              A * direction_var * A.transpose();
    cov += M3D::Identity() * min_observation_variance();
    return cov;
}

inline Eigen::Matrix<double, 6, 6>
compute_pca_direction_center_covariance(const PointVector &points,
                                        const LocalFeaturePCA &feature,
                                        const int direction_eigen_index)
{
    Eigen::Matrix<double, 6, 6> feature_cov =
        Eigen::Matrix<double, 6, 6>::Zero();
    const int point_num = NUM_MATCH_POINTS;
    if (points.size() < point_num || direction_eigen_index < 0 ||
        direction_eigen_index >= 3)
    {
        return feature_cov;
    }

    const M3D &U = feature.eigen_vectors;
    const V3D &lambda = feature.eigen_values;
    const V3D direction = U.col(direction_eigen_index);
    const M3D J_center = M3D::Identity() / static_cast<double>(point_num);

    // 不改变原来的 PCA 分类/拟合结果，只把同一组最近邻点的测量协方差
    // 通过特征向量和中心点雅可比传播到 [direction(or normal), center]。
    for (int i = 0; i < point_num; ++i)
    {
        const V3D point(points[i].x, points[i].y, points[i].z);
        M3D F = M3D::Zero();
        for (int m = 0; m < 3; ++m)
        {
            if (m == direction_eigen_index)
            {
                continue;
            }

            double denom = static_cast<double>(point_num) *
                           (lambda(direction_eigen_index) - lambda(m));
            if (std::abs(denom) < 1e-9)
            {
                denom = (denom >= 0.0) ? 1e-9 : -1e-9;
            }

            const M3D eig_mix = U.col(m) * direction.transpose() +
                                direction * U.col(m).transpose();
            F.row(m) = (point - feature.centroid).transpose() / denom * eig_mix;
        }

        Eigen::Matrix<double, 6, 3> J;
        J.block<3, 3>(0, 0) = U * F;
        J.block<3, 3>(3, 0) = J_center;

        const M3D point_cov = calc_lidar_point_cov(point, ranging_cov, angle_cov);
        feature_cov += J * point_cov * J.transpose();
    }
    return feature_cov;
}

inline double calc_plane_feature_observation_variance(
    const V3D &point_world, const V3D &normal, const V3D &center,
    const Eigen::Matrix<double, 6, 6> &feature_cov)
{
    // 点面残差 r = n^T(p - q)，这里计算平面参数不确定性对 r 的贡献。
    Eigen::Matrix<double, 1, 6> J;
    J.block<1, 3>(0, 0) = (point_world - center).transpose();
    J.block<1, 3>(0, 3) = -normal.transpose();
    const double variance = (J * feature_cov * J.transpose())(0, 0);
    return std::max(min_observation_variance(), variance);
}

inline double calc_line_feature_observation_variance(
    const V3D &point_world, const V3D &line_center, const V3D &line_direction,
    const V3D &residual_normal,
    const Eigen::Matrix<double, 6, 6> &feature_cov)
{
    // 点线残差拆成两条垂直于线方向的标量残差；这里计算每条残差
    // 中线方向和线中心不确定性带来的方差。
    const V3D diff = point_world - line_center;
    const double along = line_direction.dot(diff);
    Eigen::Matrix<double, 1, 6> J;
    J.block<1, 3>(0, 0) = -along * residual_normal.transpose();
    J.block<1, 3>(0, 3) = -residual_normal.transpose();
    const double variance = (J * feature_cov * J.transpose())(0, 0);
    return std::max(min_observation_variance(), variance);
}

// 对最近邻地图点做 PCA，提取特征值、特征向量、线性度和面性度。
inline bool compute_local_pca(const PointVector &points, LocalFeaturePCA &feature)
{
    if (points.size() < NUM_MATCH_POINTS)
    {
        feature = LocalFeaturePCA();
        return false;
    }

    feature = LocalFeaturePCA();
    for (int i = 0; i < NUM_MATCH_POINTS; ++i)
    {
        feature.centroid += V3D(points[i].x, points[i].y, points[i].z);
    }
    feature.centroid /= static_cast<double>(NUM_MATCH_POINTS);

    M3D cov = M3D::Zero();
    for (int i = 0; i < NUM_MATCH_POINTS; ++i)
    {
        V3D diff(points[i].x, points[i].y, points[i].z);
        diff -= feature.centroid;
        cov += diff * diff.transpose();
    }
    cov /= static_cast<double>(NUM_MATCH_POINTS);

    Eigen::SelfAdjointEigenSolver<M3D> solver(cov);
    if (solver.info() != Eigen::Success)
    {
        feature = LocalFeaturePCA();
        return false;
    }

    feature.eigen_values = solver.eigenvalues();
    feature.eigen_vectors = solver.eigenvectors();

    const double lambda_large = feature.eigen_values(2);
    if (lambda_large <= 1e-12)
    {
        feature = LocalFeaturePCA();
        return false;
    }

    feature.linearity = (feature.eigen_values(2) - feature.eigen_values(1)) / lambda_large;
    feature.planarity = (feature.eigen_values(1) - feature.eigen_values(0)) / lambda_large;
    return true;
}

// 根据 PCA 的线性度和面性度，把当前近邻分类为点线、点面或无效。
inline LidarResidualType classify_lidar_feature(const LocalFeaturePCA &feature,
                                                double linearity_threshold,
                                                double planarity_threshold)
{
    if (feature.linearity >= linearity_threshold && feature.planarity < planarity_threshold)
    {
        return LidarResidualType::PointLine;
    }
    if (feature.planarity >= planarity_threshold)
    {
        return LidarResidualType::PointPlane;
    }
    return LidarResidualType::Invalid;
}

// 构造 PCA 点面观测；直接复用 PCA 法向和均值，避免分类后再次调用 esti_plane。
inline bool build_point_plane_match(const PointType &point_world,
                                     double point_body_norm,
                                     const LocalFeaturePCA &feature,
                                     const PointVector &points_near,
                                     float plane_threshold,
                                     LidarMatchResult &match)
{
    match = LidarMatchResult();
    match.type = LidarResidualType::PointPlane;

    V3D normal = feature.eigen_vectors.col(0);
    const double normal_norm = normal.norm();
    if (normal_norm <= 1e-12)
    {
        match.type = LidarResidualType::Invalid;
        return false;
    }
    normal /= normal_norm;
    const double d = -normal.dot(feature.centroid);

    // 用 PCA 平面重新检查近邻一致性，保留原来平面拟合阈值的过滤作用。
    for (int i = 0; i < NUM_MATCH_POINTS; ++i)
    {
        const V3D p(points_near[i].x, points_near[i].y, points_near[i].z);
        if (std::fabs(normal.dot(p) + d) > plane_threshold)
        {
            match.type = LidarResidualType::Invalid;
            return false;
        }
    }

    const double pd2 = normal.x() * point_world.x + normal.y() * point_world.y +
                       normal.z() * point_world.z + d;
    const double s = 1 - 0.9 * std::fabs(pd2) / std::sqrt(point_body_norm);
    if (s <= 0.9)
    {
        match.type = LidarResidualType::Invalid;
        return false;
    }

    VF(4) pabcd;
    pabcd << static_cast<float>(normal.x()), static_cast<float>(normal.y()),
             static_cast<float>(normal.z()), static_cast<float>(d);
    match.plane_coeff = pabcd;
    match.residual_normals[0] = normal;
    match.residuals[0] = pd2;
    const Eigen::Matrix<double, 6, 6> feature_cov =
        compute_pca_direction_center_covariance(points_near, feature, 0);
    const V3D point(point_world.x, point_world.y, point_world.z);
    match.feature_variances[0] =
        calc_plane_feature_observation_variance(point, normal, feature.centroid,
                                                feature_cov);
    match.residual_count = 1;
    match.residual = pd2;
    return true;
}

// 构造点线观测；用线方向的两个垂直方向形成两行点到线残差。
inline bool build_point_line_match(const PointType &point_world,
                                   const LocalFeaturePCA &feature,
                                   const PointVector &points_near,
                                   double line_residual_threshold,
                                   LidarMatchResult &match)
{
    match = LidarMatchResult();
    match.type = LidarResidualType::PointLine;
    match.line_center = feature.centroid;
    match.line_direction = feature.eigen_vectors.col(2);

    const double dir_norm = match.line_direction.norm();
    if (dir_norm <= 1e-12)
    {
        match.type = LidarResidualType::Invalid;
        return false;
    }
    match.line_direction /= dir_norm;

    V3D helper = (std::fabs(match.line_direction.z()) < 0.9) ?
                 V3D::UnitZ() : V3D::UnitX();
    V3D n1 = match.line_direction.cross(helper);
    if (n1.norm() <= 1e-12)
    {
        helper = V3D::UnitY();
        n1 = match.line_direction.cross(helper);
    }
    n1.normalize();
    V3D n2 = match.line_direction.cross(n1);
    n2.normalize();

    const V3D p(point_world.x, point_world.y, point_world.z);
    const V3D diff = p - match.line_center;
    const double r1 = n1.dot(diff);
    const double r2 = n2.dot(diff);
    const double dist2 = r1 * r1 + r2 * r2;
    if (dist2 > line_residual_threshold * line_residual_threshold)
    {
        match.type = LidarResidualType::Invalid;
        return false;
    }

    match.residual_normals[0] = n1;
    match.residual_normals[1] = n2;
    match.residuals[0] = r1;
    match.residuals[1] = r2;
    const Eigen::Matrix<double, 6, 6> feature_cov =
        compute_pca_direction_center_covariance(points_near, feature, 2);
    match.feature_variances[0] =
        calc_line_feature_observation_variance(p, match.line_center,
                                               match.line_direction, n1,
                                               feature_cov);
    match.feature_variances[1] =
        calc_line_feature_observation_variance(p, match.line_center,
                                               match.line_direction, n2,
                                               feature_cov);
    match.residual_count = 2;
    match.residual = std::sqrt(dist2);
    return true;
}
// const bool time_list(PointType &x, PointType &y); // {return (x.curvature < y.curvature);};
// template<typename T>
// const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

#endif
