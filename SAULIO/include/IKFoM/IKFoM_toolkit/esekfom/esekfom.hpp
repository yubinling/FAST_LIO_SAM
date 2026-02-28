/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Author: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ESEKFOM_EKF_HPP
#define ESEKFOM_EKF_HPP

#include <vector>
#include <cstdlib>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/types/SEn.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"
extern double solve_time_3;
extern FILE *fp_debug;
extern double time_current;
extern double first_lidar_time;
extern int effect_oneset_num;
extern double publish_time;
extern bool start_predict;
using namespace std;
namespace esekfom
{

	using namespace Eigen;

	template <typename T>
	struct dyn_share_modified
	{
		bool valid;
		bool converge;
		T M_Noise;
		Eigen::Matrix<T, Eigen::Dynamic, 1> z;
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> h_x;
		Eigen::Matrix<T, 6, 1> z_IMU;
		Eigen::Matrix<T, 6, 1> R_IMU;
		bool satu_check[6];
	};

	template <typename state, int process_noise_dof, typename input = state, typename measurement = state, int measurement_noise_dof = 0>
	class esekf
	{

		typedef esekf self;
		enum
		{
			n = state::DOF,
			m = state::DIM,
			l = measurement::DOF
		};

	public:
		typedef typename state::scalar scalar_type;
		typedef Matrix<scalar_type, n, n> cov;
		typedef Matrix<scalar_type, m, n> cov_;
		typedef SparseMatrix<scalar_type> spMt;
		typedef Matrix<scalar_type, n, 1> vectorized_state;
		typedef Matrix<scalar_type, m, 1> flatted_state;
		typedef flatted_state processModel(state &, const input &);
		typedef Eigen::Matrix<scalar_type, m, n> processMatrix1(state &, const input &);
		typedef Eigen::Matrix<scalar_type, m, process_noise_dof> processMatrix2(state &, const input &);
		typedef Eigen::Matrix<scalar_type, process_noise_dof, process_noise_dof> processnoisecovariance;

		typedef void measurementModel_dyn_share_modified_cov(state &, Eigen::Matrix3d, Eigen::Matrix3d, dyn_share_modified<scalar_type> &);
		typedef void measurementModel_dyn_share_modified(state &, dyn_share_modified<scalar_type> &);
		typedef Eigen::Matrix<scalar_type, l, n> measurementMatrix1(state &);
		typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, n> measurementMatrix1_dyn(state &);
		typedef Eigen::Matrix<scalar_type, l, measurement_noise_dof> measurementMatrix2(state &);
		typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementMatrix2_dyn(state &);
		typedef Eigen::Matrix<scalar_type, measurement_noise_dof, measurement_noise_dof> measurementnoisecovariance;
		typedef Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> measurementnoisecovariance_dyn;

		int maximum_iter = 0; // 本来是private，为了修改改成了public
		esekf(const state &x = state(),
			  const cov &P = cov::Identity()) : x_(x), P_(P) {};

		void init_dyn_share_modified_2h(processModel f_in, processMatrix1 f_x_in, processMatrix2 f_w_in,measurementModel_dyn_share_modified_cov h_dyn_share_in1, measurementModel_dyn_share_modified h_dyn_share_in2,measurementModel_dyn_share_modified h_dyn_share_in3, measurementModel_dyn_share_modified h_dyn_share_in4,int maximum_iteration_1, scalar_type limit_vector[n])
		{
			f = f_in;
			f_x = f_x_in;
			f_w = f_w_in;
			h_dyn_share_modified_1 = h_dyn_share_in1;
			h_dyn_share_modified_2 = h_dyn_share_in2;
			h_dyn_share_modified_3 = h_dyn_share_in3;//bigupdate
			h_dyn_share_modified_4= h_dyn_share_in4;
			maximum_iter = 1; // 由1改成3
			for(int i=0; i<n; i++)
		    {
			limit[i] = limit_vector[i];
		   }
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
			
		}

		void init_dyn_share_modified_3h(processModel f_in, processMatrix1 f_x_in, measurementModel_dyn_share_modified_cov h_dyn_share_in1, measurementModel_dyn_share_modified h_dyn_share_in2)
		{
			f = f_in;
			f_x = f_x_in;
			// f_w = f_w_in;
			h_dyn_share_modified_1 = h_dyn_share_in1;
			h_dyn_share_modified_2 = h_dyn_share_in2;
			// h_dyn_share_modified_3 = h_dyn_share_in3;
			maximum_iter = 1;
			x_.build_S2_state();
			x_.build_SO3_state();
			x_.build_vect_state();
			x_.build_SEN_state();
		}

		// iterated error state EKF propogation
		// void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov)
		// {
		// 	if (predict_state)
		// 	{
		// 		flatted_state f_ = f(x_, i_in);
		// 		x_.oplus(f_, dt);
		// 	}

		// 	if (prop_cov)
		// 	{
		// 		flatted_state f_ = f(x_, i_in);
		// 		// state x_before = x_;

		// 		cov_ f_x_ = f_x(x_, i_in);
		// 		cov f_x_final;
		// 		F_x1 = cov::Identity();
		// 		for (std::vector<std::pair<std::pair<int, int>, int>>::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++)
		// 		{
		// 			int idx = (*it).first.first;
		// 			int dim = (*it).first.second;
		// 			int dof = (*it).second;
		// 			for (int i = 0; i < n; i++)
		// 			{
		// 				for (int j = 0; j < dof; j++)
		// 				{
		// 					f_x_final(idx + j, i) = f_x_(dim + j, i);
		// 				}
		// 			}
		// 		}

		// 		Matrix<scalar_type, 3, 3> res_temp_SO3;
		// 		MTK::vect<3, scalar_type> seg_SO3;
		// 		for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
		// 		{
		// 			int idx = (*it).first;
		// 			int dim = (*it).second;
		// 			for (int i = 0; i < 3; i++)
		// 			{
		// 				seg_SO3(i) = -1 * f_(dim + i) * dt;
		// 			}
		// 			// MTK::SO3<scalar_type> res;
		// 			// res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
		// 			F_x1.template block<3, 3>(idx, idx) = MTK::SO3<scalar_type>::exp(seg_SO3); // res.normalized().toRotationMatrix();
		// 			res_temp_SO3 = MTK::A_matrix(seg_SO3);
		// 			for (int i = 0; i < n; i++)
		// 			{
		// 				f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.template block<3, 1>(dim, i));
		// 			}
		// 		}

		// 		F_x1 += f_x_final * dt;
		// 		P_ = F_x1 * P_ * (F_x1).transpose() + Q * (dt * dt);
		// 	}
		// }
		void predict(double &dt, processnoisecovariance &Q, const input &i_in, bool predict_state, bool prop_cov)

		{
			if (start_predict==false)
			{
				x_cov=x_;
				start_predict=true;
			}
			if (predict_state)//fastlio
			{
				flatted_state f_ = f(x_, i_in);
				x_.oplus(f_, dt);
			}

			if (prop_cov)
			{   
			
				flatted_state f_ = f(x_cov, i_in);

				cov_ f_x_ = f_x(x_cov, i_in);
			
				cov f_x_final;
                
				Matrix<scalar_type, m, process_noise_dof> f_w_ = f_w(x_cov, i_in);
			
				Matrix<scalar_type, n, process_noise_dof> f_w_final;
				state x_before = x_cov;
				
				x_cov.oplus(f_, dt);
          
				F_x1 = cov::Identity();
				for (std::vector<std::pair<std::pair<int, int>, int> >::iterator it = x_cov.vect_state.begin(); it != x_cov.vect_state.end(); it++) {
					int idx = (*it).first.first;
					int dim = (*it).first.second;
					int dof = (*it).second;
					for(int i = 0; i < n; i++){
						for(int j=0; j<dof; j++)
						{f_x_final(idx+j, i) = f_x_(dim+j, i);}	
					}
					for(int i = 0; i < process_noise_dof; i++){
						for(int j=0; j<dof; j++)
						{f_w_final(idx+j, i) = f_w_(dim+j, i);}
					}
				}
				
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for (std::vector<std::pair<int, int> >::iterator it = x_cov.SO3_state.begin(); it != x_cov.SO3_state.end(); it++) {
					int idx = (*it).first;
					int dim = (*it).second;
					for(int i = 0; i < 3; i++){
						seg_SO3(i) = -1 * f_(dim + i) * dt;
					}
					//MTK::SO3<scalar_type> res;
					//res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
					//LYB
					MTK::SO3<scalar_type> res;
					res = MTK::SO3<scalar_type>::exp(seg_SO3);
				#ifdef USE_sparse
					res_temp_SO3 = res.toRotationMatrix();
					for(int i = 0; i < 3; i++){
						for(int j = 0; j < 3; j++){
							f_x_1.coeffRef(idx + i, idx + j) = res_temp_SO3(i, j);
						}
					}
				#else
					F_x1.template block<3, 3>(idx, idx) = res;
				#endif			
					res_temp_SO3 = MTK::A_matrix(seg_SO3);
					for(int i = 0; i < n; i++){
						f_x_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_. template block<3, 1>(dim, i));	
					}
					for(int i = 0; i < process_noise_dof; i++){
						f_w_final. template block<3, 1>(idx, i) = res_temp_SO3 * (f_w_. template block<3, 1>(dim, i));
					}
				}
				
				
				// Matrix<scalar_type, 2, 3> res_temp_S2;
				// Matrix<scalar_type, 2, 2> res_temp_S2_;
				// MTK::vect<3, scalar_type> seg_S2;
				// for (std::vector<std::pair<int, int> >::iterator it = x_.S2_state.begin(); it != x_.S2_state.end(); it++) {
				// 	int idx = (*it).first;
				// 	int dim = (*it).second;
				// 	for(int i = 0; i < 3; i++){
				// 		seg_S2(i) = f_(dim + i) * dt;
				// 	}
				// 	MTK::vect<2, scalar_type> vec = MTK::vect<2, scalar_type>::Zero();
				// 	MTK::SO3<scalar_type> res;
				// 	res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_S2, scalar_type(1/2));
				// 	Eigen::Matrix<scalar_type, 2, 3> Nx;
				// 	Eigen::Matrix<scalar_type, 3, 2> Mx;
				// 	x_.S2_Nx_yy(Nx, idx);
				// 	x_before.S2_Mx(Mx, vec, idx);
				// #ifdef USE_sparse
				// 	res_temp_S2_ = Nx * res.toRotationMatrix() * Mx;
				// 	for(int i = 0; i < 2; i++){
				// 		for(int j = 0; j < 2; j++){
				// 			f_x_1.coeffRef(idx + i, idx + j) = res_temp_S2_(i, j);
				// 		}
				// 	}
				// #else
				// 	F_x1.template block<2, 2>(idx, idx) = Nx * res.toRotationMatrix() * Mx;
				// #endif

				// 	Eigen::Matrix<scalar_type, 3, 3> x_before_hat;
				// 	x_before.S2_hat(x_before_hat, idx);
				// 	res_temp_S2 = -Nx * res.toRotationMatrix() * x_before_hat*MTK::A_matrix(seg_S2).transpose();
					
				// 	for(int i = 0; i < n; i++){
				// 		f_x_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_x_. template block<3, 1>(dim, i));
						
				// 	}
				// 	for(int i = 0; i < process_noise_dof; i++){
				// 		f_w_final. template block<2, 1>(idx, i) = res_temp_S2 * (f_w_. template block<3, 1>(dim, i));
				// 	}
				// }
			
			#ifdef USE_sparse
				f_x_1.makeCompressed();
				spMt f_x2 = f_x_final.sparseView();
				spMt f_w1 = f_w_final.sparseView();
				spMt xp = f_x_1 + f_x2 * dt;
				P_ = xp * P_ * xp.transpose() + (f_w1 * dt) * Q * (f_w1 * dt).transpose();
			#else
				F_x1 += f_x_final * dt;
				P_ = (F_x1) * P_ * (F_x1).transpose() + (dt * f_w_final) * Q * (dt * f_w_final).transpose();
			#endif
				// flatted_state f_ = f(x_, i_in);
				// // state x_before = x_;

				// cov_ f_x_ = f_x(x_, i_in);
				// cov f_x_final;
				// F_x1 = cov::Identity();
				// for (std::vector<std::pair<std::pair<int, int>, int>>::iterator it = x_.vect_state.begin(); it != x_.vect_state.end(); it++)
				// {
				// 	int idx = (*it).first.first;
				// 	int dim = (*it).first.second;
				// 	int dof = (*it).second;
				// 	for (int i = 0; i < n; i++)
				// 	{
				// 		for (int j = 0; j < dof; j++)
				// 		{
				// 			f_x_final(idx + j, i) = f_x_(dim + j, i);
				// 		}
				// 	}
				// }

				// Matrix<scalar_type, 3, 3> res_temp_SO3;
				// MTK::vect<3, scalar_type> seg_SO3;
				// for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
				// {
				// 	int idx = (*it).first;
				// 	int dim = (*it).second;
				// 	for (int i = 0; i < 3; i++)
				// 	{
				// 		seg_SO3(i) = -1 * f_(dim + i) * dt;
				// 	}
				// 	// MTK::SO3<scalar_type> res;
				// 	// res.w() = MTK::exp<scalar_type, 3>(res.vec(), seg_SO3, scalar_type(1/2));
				// 	F_x1.template block<3, 3>(idx, idx) = MTK::SO3<scalar_type>::exp(seg_SO3); // res.normalized().toRotationMatrix();
				// 	res_temp_SO3 = MTK::A_matrix(seg_SO3);
				// 	for (int i = 0; i < n; i++)
				// 	{
				// 		f_x_final.template block<3, 1>(idx, i) = res_temp_SO3 * (f_x_.template block<3, 1>(dim, i));
				// 	}
				// }

				// F_x1 += f_x_final * dt;
				// P_ = F_x1 * P_ * (F_x1).transpose() + Q * (dt * dt);
				// //lyb
				// // P_ = F_x1 * P_ * (F_x1).transpose() + Q ;
			}
		}
		bool update_iterated_dyn_share_modified()
		{
			dyn_share_modified<scalar_type> dyn_share;
			state x_propagated = x_;
			int dof_Measurement;
			double m_noise;
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;

		
				h_dyn_share_modified_1(x_, P_.template block<3, 3>(0, 0), P_.template block<3, 3>(3, 3), dyn_share); // 如果是第一次，需要积累点
				

				// solve_time_1+=omp_get_wtime()-t1;
				if (!dyn_share.valid)
				{
					return false;
					// continue;
				}

				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
				// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R = dyn_share.R;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
				// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v = dyn_share.h_v;
				dof_Measurement = h_x.rows();
				m_noise = dyn_share.M_Noise;
				// dof_Measurement_noise = dyn_share.R.rows();
				// vectorized_state dx, dx_new;
				// x_.boxminus(dx, x_propagated);
				// dx_new = dx;
				// P_ = P_propagated;

				Matrix<scalar_type, n, Eigen::Dynamic> PHT;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;
				Matrix<scalar_type, n, Eigen::Dynamic> K_;
				if (n > dof_Measurement)
				{
					PHT = P_.template block<n, 12>(0, 0) * h_x.transpose();
					HPHT = h_x * PHT.topRows(12);
					for (int m = 0; m < dof_Measurement; m++)
					{
						HPHT(m, m) += m_noise;
					}
					K_ = PHT * HPHT.inverse();
				}
				else
				{
					Matrix<scalar_type, 12, 12> HTH = 1 / m_noise * h_x.transpose() * h_x;
					Matrix<scalar_type, n, n> P_inv = P_.inverse();
					P_inv.template block<12, 12>(0, 0) += HTH;
					P_inv = P_inv.inverse();
					K_ = P_inv.template block<n, 12>(0, 0) * h_x.transpose() / m_noise;
				}
				Matrix<scalar_type, n, 1> dx_ = K_ * z; // - h) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
				// state x_before = x_;

				x_.boxplus(dx_);
				{
					P_ = P_ - K_ * h_x * P_.template block<12, n>(0, 0);
				}


			}
			return true;
		}

		bool update_iterated_dyn_share_modified_1()
		{

			dyn_share_modified<scalar_type> dyn_share;
			dyn_share.valid = true;
			dyn_share.converge = true;
			int t = 0;
			// 获取上一次的状态和协方差矩阵
			state x_propagated = x_;
			cov P_propagated = P_;
			int dof_Measurement;

			Matrix<scalar_type, n, 1> K_h;
			Matrix<scalar_type, n, n> K_x;

			vectorized_state dx_new = vectorized_state::Zero();
			// 最多进行maximum_iter次迭代优化


			
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;
				// 计算测量模型方程的雅克比，也就是点面残差的导数 H(代码里是h_x)
				if (i == 0)
				{
					h_dyn_share_modified_1(x_, P_.template block<3, 3>(0, 0), P_.template block<3, 3>(3, 3), dyn_share); // 如果是第一次，需要积累点
				}
				else
				{
					h_dyn_share_modified_2(x_, dyn_share); // 第二次之后
				}
				// Matrix<scalar_type, Eigen::Dynamic, 1> h = h_dyn_share(x_, dyn_share);
				double R = dyn_share.M_Noise;
				
				if (!dyn_share.valid)
				{
					// continue;
					
										// continue;
					if(i ==0){
					return false;}
					else{
						continue;
					}
				}
				
				double solve_start = omp_get_wtime();
				// 获取测量模型的雅克比d(pos, rot, 0, 0)/dx
				Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ = dyn_share.h_x;

				dof_Measurement = h_x_.rows(); // 观测方程个数m
				if (dof_Measurement <= 0 || dyn_share.z.rows() != dof_Measurement ||
				    !h_x_.allFinite() || !dyn_share.z.allFinite() || !std::isfinite(R) || R <= 0.0)
				{
					// 大更新来自多段累计点云，偶发空观测或异常数值时跳过本轮，避免矩阵求逆崩溃。
					std::cout << "[saulio bigupdate] skip invalid measurement, rows="
					          << dof_Measurement << ", z_rows=" << dyn_share.z.rows()
					          << ", noise=" << R << std::endl;
					dyn_share.valid = false;
					continue;
				}
				vectorized_state dx;		   // 定义误差状态
				x_.boxminus(dx, x_propagated); // 获取误差dx
				dx_new = dx;				   // 用于迭代的误差状态

				// 预测得到的误差状态协方差矩阵
				// 协方差矩阵在迭代过程中不会代入下一次迭代，直到最后一次退出时更新，在迭代过程中更新的只是先验
				P_ = P_propagated;
				// 这一大段都在求协方差的先验更新，大致上是P=(J^-1)*P*(J^-T)如论文式16~18
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
				{
					int idx = (*it).first;
					int dim = (*it).second;
					for (int i = 0; i < 3; i++)
					{
						seg_SO3(i) = dx(idx + i);
					}

					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
					for (int i = 0; i < n; i++)
					{
						P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
					}
					for (int i = 0; i < n; i++)
					{
						P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}

				// 状态维度 n > 测量维度 dof_Measurement
				// 如果状态量维度大于观测方程 n > m，不满秩
				if (n > dof_Measurement)
				{
					// #ifdef USE_sparse
					//  Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_temp = h_x * P_ * h_x.transpose();
					//  spMt R_temp = h_v * R_ * h_v.transpose();
					//  K_temp += R_temp;
					Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					// 每一次迭代将重新计算增益K，即论文式18
					h_x_cur.topLeftCorner(dof_Measurement, 12) = h_x_;
					/*
					h_x_cur.col(0) = h_x_.col(0);
					h_x_cur.col(1) = h_x_.col(1);
					h_x_cur.col(2) = h_x_.col(2);
					h_x_cur.col(3) = h_x_.col(3);
					h_x_cur.col(4) = h_x_.col(4);
					h_x_cur.col(5) = h_x_.col(5);
					h_x_cur.col(6) = h_x_.col(6);
					h_x_cur.col(7) = h_x_.col(7);
					h_x_cur.col(8) = h_x_.col(8);
					h_x_cur.col(9) = h_x_.col(9);
					h_x_cur.col(10) = h_x_.col(10);
					h_x_cur.col(11) = h_x_.col(11);
					*/
					// 重新计算增益矩阵K
					Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose() / R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse() / R;
					// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose() + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement) * R).inverse(); // 考虑大数吃小数，所以没有直接加R，是变换为了乘除的形式
					K_h = K_ * dyn_share.z;
					K_x = K_ * h_x_cur;
				}
				else
				{
					// 避免求逆矩阵，K按稀疏矩阵分解的方法如论文式20

					cov P_temp = (P_ / R).inverse();
					// Eigen::Matrix<scalar_type, 12, Eigen::Dynamic> h_T = h_x_.transpose();
					Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_.transpose() * h_x_;
					P_temp.template block<12, 12>(0, 0) += HTH;
					/*
					Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					//std::cout << "line 1767" << std::endl;
					h_x_cur.col(0) = h_x_.col(0);
					h_x_cur.col(1) = h_x_.col(1);
					h_x_cur.col(2) = h_x_.col(2);
					h_x_cur.col(3) = h_x_.col(3);
					h_x_cur.col(4) = h_x_.col(4);
					h_x_cur.col(5) = h_x_.col(5);
					h_x_cur.col(6) = h_x_.col(6);
					h_x_cur.col(7) = h_x_.col(7);
					h_x_cur.col(8) = h_x_.col(8);
					h_x_cur.col(9) = h_x_.col(9);
					h_x_cur.col(10) = h_x_.col(10);
					h_x_cur.col(11) = h_x_.col(11);
					*/
					cov P_inv = P_temp.inverse();
					// std::cout << "line 1781" << std::endl;
					K_h = P_inv.template block<n, 12>(0, 0) * h_x_.transpose() * dyn_share.z; // (H_T_H + P^-1)^-1 * H^T * h(残差) = K * h
					// std::cout << "line 1780" << std::endl;
					// cov HTH_cur = cov::Zero();
					// HTH_cur. template block<12, 12>(0, 0) = HTH;
					K_x.setZero(); // = cov::Zero();

					K_x.template block<n, 12>(0, 0) = P_inv.template block<n, 12>(0, 0) * HTH; //(H_T_H + P^-1)^-1 * H_T_H = KH

					// K_= (h_x_.transpose() * h_x_ + (P_/R).inverse()).inverse()*h_x_.transpose();
				}
				// if(i==0)
				// {
				// 	Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_.transpose() * h_x_;
				// 	EigenSolver<Eigen::MatrixXd> solver(HTH);
				// 	VectorXd eigenvalues= solver.eigenvalues().real();
    			// 	fprintf(fp_debug,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",time_current-first_lidar_time,eigenvalues[0],eigenvalues[1],eigenvalues[2],\
				// 	eigenvalues[3],eigenvalues[4],eigenvalues[5],eigenvalues[6],eigenvalues[7],eigenvalues[8],eigenvalues[9],eigenvalues[10],eigenvalues[11]);
    			// 	fflush(fp_debug);
				// }

				// K_x = K_ * h_x_;
				// 由于是误差迭代KF，得到的是误差的最优估计！
				Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; // 误差增量后验 K*h + (K*H - I) dx

				state x_before = x_; // 加上校正后的误差状态dx_
				x_.boxplus(dx_);	 // 根据计算得到的误差增量后验，更新状态量
				// 判断迭代是否发散
				dyn_share.converge = true;
				// 判断已收敛的条件是误差的估计值小于阈值
				for (int i = 0; i < n; i++)
				{
					if (std::fabs(dx_[i]) > limit[i])
					{
						dyn_share.converge = false;
						break;
					}
				}
				if (dyn_share.converge)
					t++;

				if (!t && i == maximum_iter - 2)
				{
					dyn_share.converge = true;
				}
				// 迭代完成后更新误差状态协方差矩阵
				// 结束迭代后，更新协方差矩阵的后验值，大致上是P=(I-K*H)*P，如论文式19
				if (t > 1 || i == maximum_iter - 1)
				{
					// double solve_start = omp_get_wtime();
					L_ = P_;
					// std::cout << "iteration time" << t << "," << i << std::endl;
					Matrix<scalar_type, 3, 3> res_temp_SO3;
					MTK::vect<3, scalar_type> seg_SO3;
					for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
					{
						int idx = (*it).first;
						for (int i = 0; i < 3; i++)
						{
							seg_SO3(i) = dx_(i + idx);
						}
						res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
						for (int i = 0; i < n; i++)
						{
							L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
						}
						// if(n > dof_Measurement)
						// {
						// 	for(int i = 0; i < dof_Measurement; i++){
						// 		K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
						// 	}
						// }
						// else
						// {
						for (int i = 0; i < 12; i++)
						{
							K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
						}
						//}
						for (int i = 0; i < n; i++)
						{
							L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
							P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						}
					}

					// if(n > dof_Measurement)
					// {
					// 	Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					// 	h_x_cur.topLeftCorner(dof_Measurement, 12) = h_x_;
					// 	/*
					// 	h_x_cur.col(0) = h_x_.col(0);
					// 	h_x_cur.col(1) = h_x_.col(1);
					// 	h_x_cur.col(2) = h_x_.col(2);
					// 	h_x_cur.col(3) = h_x_.col(3);
					// 	h_x_cur.col(4) = h_x_.col(4);
					// 	h_x_cur.col(5) = h_x_.col(5);
					// 	h_x_cur.col(6) = h_x_.col(6);
					// 	h_x_cur.col(7) = h_x_.col(7);
					// 	h_x_cur.col(8) = h_x_.col(8);
					// 	h_x_cur.col(9) = h_x_.col(9);
					// 	h_x_cur.col(10) = h_x_.col(10);
					// 	h_x_cur.col(11) = h_x_.col(11);
					// 	*/
					// 	P_ = L_ - K_*h_x_cur * P_;
					// }
					// else
					//{
					P_ = L_ - K_x.template block<n, 12>(0, 0) * P_.template block<12, n>(0, 0);
					//}
					solve_time_3 += omp_get_wtime() - solve_start;
					// if (effect_oneset_num<100){
					// 	break;
					// }
					return true;
				}
				solve_time_3 += omp_get_wtime() - solve_start;
				// if (effect_oneset_num<100){
				// 		break;
				// }

			}
	
			return true;
		}

		bool update_iterated_dyn_share_modified_bigupdate()
		{

			dyn_share_modified<scalar_type> dyn_share;
			dyn_share.valid = true;
			dyn_share.converge = true;
			int t = 0;
			// 获取上一次的状态和协方差矩阵
			state x_propagated = x_;
			cov P_propagated = P_;
			int dof_Measurement;

			Matrix<scalar_type, n, 1> K_h;
			Matrix<scalar_type, n, n> K_x;

			vectorized_state dx_new = vectorized_state::Zero();
			// 最多进行maximum_iter次迭代优化

			bool a =true;
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;
				cout<<i<<endl<<endl;
				// std::cout<<"9"<<std::endl;
				// h_dyn_share_modified_3(x_, dyn_share);
				// 计算测量模型方程的雅克比，也就是点面残差的导数 H(代码里是h_x)
				// if (i == 0)
				// {
				// 	h_dyn_share_modified_1(x_, P_.template block<3, 3>(0, 0), P_.template block<3, 3>(3, 3), dyn_share); // 如果是第一次，需要积累点
				// }
				// else
				// {
				// 	h_dyn_share_modified_2(x_, dyn_share); // 第二次之后
				// }
				// // Matrix<scalar_type, Eigen::Dynamic, 1> h = h_dyn_share(x_, dyn_share);
				if (i==0){
					a= true;
					
					h_dyn_share_modified_3(x_, dyn_share);
									// if(i==0)
				// {

    				// fprintf(fp_debug,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",time_current-first_lidar_time,eigenvalues[0],eigenvalues[1],eigenvalues[2],\
					// eigenvalues[3],eigenvalues[4],eigenvalues[5],eigenvalues[6],eigenvalues[7],eigenvalues[8],eigenvalues[9],eigenvalues[10],eigenvalues[11]);
    				// fflush(fp_debug);
				// }
				}
				else{
					h_dyn_share_modified_4(x_, dyn_share);
				}
				double R = dyn_share.M_Noise;
				if (!dyn_share.valid)
				{
					// continue;
										// continue;
					// if(i ==0){
					// return false;}
					// else{
						continue;
					// }
				}
				double solve_start = omp_get_wtime();
				// 获取测量模型的雅克比d(pos, rot, 0, 0)/dx
				Eigen::Matrix<scalar_type, Eigen::Dynamic, 12> h_x_ = dyn_share.h_x;
				// if (i==0){
									    
				// 	// std::cout<<"Hx: "<<h_x_<<std::endl;
				// 	Eigen::Matrix<scalar_type, 12, 12> HT_H = h_x_.transpose() * h_x_;
				// 	// std::cout<<"HTH: "<<HT_H<<std::endl;
				// 	EigenSolver<Eigen::MatrixXd> solver(HT_H);
			
				// 	VectorXd eigenvalues= solver.eigenvalues().real();
				// 	// std::cout<<"eigenvalues[0]"<<eigenvalues[0]<<std::endl;
				// 	// std::cout<<"eigenvalues[1]"<<eigenvalues[1]<<std::endl;
				// 	// std::cout<<"eigenvalues[2]"<<eigenvalues[2]<<std::endl;
				// 	// std::cout<<"eigenvalues[3]"<<eigenvalues[3]<<std::endl;
				//     // std::cout<<"eigenvalues[4]"<<eigenvalues[4]<<std::endl;
				// 	// std::cout<<"eigenvalues[5]"<<eigenvalues[5]<<std::endl;
				// 	if (eigenvalues[0]<1000000 || eigenvalues[1]<680000 ||eigenvalues[2]<150000 ||eigenvalues[3]<1200||eigenvalues[4]<1200 ||eigenvalues[5]<1000 ){
				// 	// if (eigenvalues[0]<300000 || eigenvalues[1]<200000 ||eigenvalues[2]<80000 ||eigenvalues[3]<600||eigenvalues[4]<600 ||eigenvalues[5]<500 ){
				// 		a=false;
				// 	}
				// 	// if(time_current-first_lidar_time<2){
				// 	// 	if (time_current-publish_time>0.1){
				// 	// 	a=true;}
				// 	// }
				// 	// else{
				// 	if (time_current-publish_time>0.2)
				// 		{a=true;}
						
				// 	// }
				// 	if (a==false){
				// 		return false;
				// 	}
				// }
                
                
				dof_Measurement = h_x_.rows(); // 观测方程个数m
				vectorized_state dx;		   // 定义误差状态
				x_.boxminus(dx, x_propagated); // 获取误差dx
				dx_new = dx;				   // 用于迭代的误差状态
                
				// 预测得到的误差状态协方差矩阵
				// 协方差矩阵在迭代过程中不会代入下一次迭代，直到最后一次退出时更新，在迭代过程中更新的只是先验
				P_ = P_propagated;
				
				// 这一大段都在求协方差的先验更新，大致上是P=(J^-1)*P*(J^-T)如论文式16~18
				Matrix<scalar_type, 3, 3> res_temp_SO3;
				MTK::vect<3, scalar_type> seg_SO3;
				for (std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
				{
					int idx = (*it).first;
					int dim = (*it).second;
					for (int i = 0; i < 3; i++)
					{
						seg_SO3(i) = dx(idx + i);
					}

					res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
					dx_new.template block<3, 1>(idx, 0) = res_temp_SO3 * dx_new.template block<3, 1>(idx, 0);
					for (int i = 0; i < n; i++)
					{
						P_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
					}
					for (int i = 0; i < n; i++)
					{
						P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
					}
				}
               
				// 状态维度 n > 测量维度 dof_Measurement
				// 如果状态量维度大于观测方程 n > m，不满秩
				if (n > dof_Measurement)
				{
					// #ifdef USE_sparse
					//  Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_temp = h_x * P_ * h_x.transpose();
					//  spMt R_temp = h_v * R_ * h_v.transpose();
					//  K_temp += R_temp;
					Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					// 每一次迭代将重新计算增益K，即论文式18
					h_x_cur.topLeftCorner(dof_Measurement, 12) = h_x_;
					/*
					h_x_cur.col(0) = h_x_.col(0);
					h_x_cur.col(1) = h_x_.col(1);
					h_x_cur.col(2) = h_x_.col(2);
					h_x_cur.col(3) = h_x_.col(3);
					h_x_cur.col(4) = h_x_.col(4);
					h_x_cur.col(5) = h_x_.col(5);
					h_x_cur.col(6) = h_x_.col(6);
					h_x_cur.col(7) = h_x_.col(7);
					h_x_cur.col(8) = h_x_.col(8);
					h_x_cur.col(9) = h_x_.col(9);
					h_x_cur.col(10) = h_x_.col(10);
					h_x_cur.col(11) = h_x_.col(11);
					*/
					// 重新计算增益矩阵K
					Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose() / R + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement)).inverse() / R;
					// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> K_ = P_ * h_x_cur.transpose() * (h_x_cur * P_ * h_x_cur.transpose() + Eigen::Matrix<double, Dynamic, Dynamic>::Identity(dof_Measurement, dof_Measurement) * R).inverse(); // 考虑大数吃小数，所以没有直接加R，是变换为了乘除的形式
					K_h = K_ * dyn_share.z;
					K_x = K_ * h_x_cur;
				}
				else
				{
					// 避免求逆矩阵，K按稀疏矩阵分解的方法如论文式20

					cov P_temp = (P_ / R).inverse();
					// Eigen::Matrix<scalar_type, 12, Eigen::Dynamic> h_T = h_x_.transpose();
					Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_.transpose() * h_x_;
					P_temp.template block<12, 12>(0, 0) += HTH;
					/*
					Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					//std::cout << "line 1767" << std::endl;
					h_x_cur.col(0) = h_x_.col(0);
					h_x_cur.col(1) = h_x_.col(1);
					h_x_cur.col(2) = h_x_.col(2);
					h_x_cur.col(3) = h_x_.col(3);
					h_x_cur.col(4) = h_x_.col(4);
					h_x_cur.col(5) = h_x_.col(5);
					h_x_cur.col(6) = h_x_.col(6);
					h_x_cur.col(7) = h_x_.col(7);
					h_x_cur.col(8) = h_x_.col(8);
					h_x_cur.col(9) = h_x_.col(9);
					h_x_cur.col(10) = h_x_.col(10);
					h_x_cur.col(11) = h_x_.col(11);
					*/
					cov P_inv = P_temp.inverse();
					// std::cout << "line 1781" << std::endl;
					K_h = P_inv.template block<n, 12>(0, 0) * h_x_.transpose() * dyn_share.z; // (H_T_H + P^-1)^-1 * H^T * h(残差) = K * h
					// std::cout << "line 1780" << std::endl;
					// cov HTH_cur = cov::Zero();
					// HTH_cur. template block<12, 12>(0, 0) = HTH;
					K_x.setZero(); // = cov::Zero();

					K_x.template block<n, 12>(0, 0) = P_inv.template block<n, 12>(0, 0) * HTH; //(H_T_H + P^-1)^-1 * H_T_H = KH

					// K_= (h_x_.transpose() * h_x_ + (P_/R).inverse()).inverse()*h_x_.transpose();
				}
				// if(i==0)
				// {
				// 	Eigen::Matrix<scalar_type, 12, 12> HTH = h_x_.transpose() * h_x_;
				// 	EigenSolver<Eigen::MatrixXd> solver(HTH);
				// 	VectorXd eigenvalues= solver.eigenvalues().real();
    			// 	fprintf(fp_debug,"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",time_current-first_lidar_time,eigenvalues[0],eigenvalues[1],[2],\
				// 	eigenvalues[3],eigenvalues[4],eigenvalues[5],eigenvalues[6],eigenvalues[7],eigenvalues[8],eigenvalues[9],eigenvalues[10],eigenvalues[11]);
    			// 	fflush(fp_debug);
				// }
               
				// K_x = K_ * h_x_;
				// 由于是误差迭代KF，得到的是误差的最优估计！
				Matrix<scalar_type, n, 1> dx_ = K_h + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new; // 误差增量后验 K*h + (K*H - I) dx
				if (!dx_.allFinite())
				{
					// 数值异常时回退到进入本次 bigupdate 前的状态，防止坏状态继续传播。
					std::cout << "[saulio bigupdate] skip non-finite dx" << std::endl;
					x_ = x_propagated;
					P_ = P_propagated;
					return false;
				}
                
				state x_before = x_; // 加上校正后的误差状态dx_
				x_.boxplus(dx_);	 // 根据计算得到的误差增量后验，更新状态量
				// 判断迭代是否发散
				dyn_share.converge = true;
				// 判断已收敛的条件是误差的估计值小于阈值
				for (int i = 0; i < n; i++)
				{
					if (std::fabs(dx_[i]) > limit[i])
					{ 
						cout<<"iiiii"<<endl;
						cout<<"dx:"<<std::fabs(dx_[i])<<endl;
						dyn_share.converge = false;
						break;
					}
				}
				if (dyn_share.converge){
					t++;
					cout<<"jkjlj"<<endl;
				}

				if (!t && i == maximum_iter - 2)
				{
					dyn_share.converge = true;
				}
				if (!dyn_share.converge && i == maximum_iter - 1)
				{
					// 最后一轮仍然超过收敛阈值时不做协方差更新，保持上一帧稳定状态。
					std::cout << "[saulio bigupdate] not converged, rollback update" << std::endl;
					x_ = x_propagated;
					P_ = P_propagated;
					return false;
				}
				// 迭代完成后更新误差状态协方差矩阵
				// 结束迭代后，更新协方差矩阵的后验值，大致上是P=(I-K*H)*P，如论文式19
				if (t > 1 || i == maximum_iter - 1)
				{
					// double solve_start = omp_get_wtime();
				
					L_ = P_;
					// std::cout << "iteration time" << t << "," << i << std::endl;
					Matrix<scalar_type, 3, 3> res_temp_SO3;
					MTK::vect<3, scalar_type> seg_SO3;
					for (typename std::vector<std::pair<int, int>>::iterator it = x_.SO3_state.begin(); it != x_.SO3_state.end(); it++)
					{
						
						int idx = (*it).first;
						for (int i = 0; i < 3; i++)
						{
							seg_SO3(i) = dx_(i + idx);
						}
						res_temp_SO3 = MTK::A_matrix(seg_SO3).transpose();
						for (int i = 0; i < n; i++)
						{
							L_.template block<3, 1>(idx, i) = res_temp_SO3 * (P_.template block<3, 1>(idx, i));
						}
						// if(n > dof_Measurement)
						// {
						// 	for(int i = 0; i < dof_Measurement; i++){
						// 		K_.template block<3, 1>(idx, i) = res_temp_SO3 * (K_. template block<3, 1>(idx, i));
						// 	}
						// }
						// else
						// {
						for (int i = 0; i < 12; i++)
						{
							K_x.template block<3, 1>(idx, i) = res_temp_SO3 * (K_x.template block<3, 1>(idx, i));
						}
						//}
						for (int i = 0; i < n; i++)
						{
							L_.template block<1, 3>(i, idx) = (L_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
							P_.template block<1, 3>(i, idx) = (P_.template block<1, 3>(i, idx)) * res_temp_SO3.transpose();
						}
					}

					// if(n > dof_Measurement)
					// {
					// 	Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x_cur = Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>::Zero(dof_Measurement, n);
					// 	h_x_cur.topLeftCorner(dof_Measurement, 12) = h_x_;
					// 	/*
					// 	h_x_cur.col(0) = h_x_.col(0);
					// 	h_x_cur.col(1) = h_x_.col(1);
					// 	h_x_cur.col(2) = h_x_.col(2);
					// 	h_x_cur.col(3) = h_x_.col(3);
					// 	h_x_cur.col(4) = h_x_.col(4);
					// 	h_x_cur.col(5) = h_x_.col(5);
					// 	h_x_cur.col(6) = h_x_.col(6);
					// 	h_x_cur.col(7) = h_x_.col(7);
					// 	h_x_cur.col(8) = h_x_.col(8);
					// 	h_x_cur.col(9) = h_x_.col(9);
					// 	h_x_cur.col(10) = h_x_.col(10);
					// 	h_x_cur.col(11) = h_x_.col(11);
					// 	*/
					// 	P_ = L_ - K_*h_x_cur * P_;
					// }
					// else
					//{
					
					P_ = L_ - K_x.template block<n, 12>(0, 0) * P_.template block<12, n>(0, 0);
					
					//}
					solve_time_3 += omp_get_wtime() - solve_start;
					// if (effect_oneset_num<100){
					// 	break;
					// }
					return true;
				}
				solve_time_3 += omp_get_wtime() - solve_start;
				// if (effect_oneset_num<100){
				// 		break;
				// }

			}
			
	
			return true;
		}


	bool update_iterated_dyn_share_modified_2()
		{
			dyn_share_modified<scalar_type> dyn_share;
			dyn_share.converge=true;
			int t = 0;
			state x_propagated = x_;
			int dof_Measurement;
			double m_noise;
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;

				if (i == 0)
				{
					h_dyn_share_modified_1(x_, P_.template block<3, 3>(0, 0), P_.template block<3, 3>(3, 3), dyn_share); // 如果是第一次，需要积累点
				}
				else
				{
					h_dyn_share_modified_2(x_, dyn_share); // 第二次之后
				}
				// solve_time_1+=omp_get_wtime()-t1;
				if (!dyn_share.valid)
				{
					return false;
					// continue;
				}

				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> z = dyn_share.z;
				// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> R = dyn_share.R;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_x = dyn_share.h_x;
				// Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> h_v = dyn_share.h_v;
				dof_Measurement = h_x.rows();
				m_noise = dyn_share.M_Noise;
				// dof_Measurement_noise = dyn_share.R.rows();
				// vectorized_state dx, dx_new;
				// x_.boxminus(dx, x_propagated);
				// dx_new = dx;
				// P_ = P_propagated;

				Matrix<scalar_type, n, Eigen::Dynamic> PHT;
				Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic> HPHT;
				Matrix<scalar_type, n, Eigen::Dynamic> K_;
				if (n > dof_Measurement)
				{
					PHT = P_.template block<n, 12>(0, 0) * h_x.transpose();
					HPHT = h_x * PHT.topRows(12);
					for (int m = 0; m < dof_Measurement; m++)
					{
						HPHT(m, m) += m_noise;
					}
					K_ = PHT * HPHT.inverse();
				}
				else
				{
					Matrix<scalar_type, 12, 12> HTH = 1 / m_noise * h_x.transpose() * h_x;
					Matrix<scalar_type, n, n> P_inv = P_.inverse();
					P_inv.template block<12, 12>(0, 0) += HTH;
					P_inv = P_inv.inverse();
					K_ = P_inv.template block<n, 12>(0, 0) * h_x.transpose() / m_noise;
				}
				Matrix<scalar_type, n, 1> dx_ = K_ * z; // - h) + (K_x - Matrix<scalar_type, n, n>::Identity()) * dx_new;
				// state x_before = x_;

				x_.boxplus(dx_);
				// {
				// 	P_ = P_ - K_ * h_x * P_.template block<12, n>(0, 0);
				// }
				for (int i = 0; i < n; i++)
				{
					if (std::fabs(dx_[i]) > limit[i])
					{
						dyn_share.converge = false;
						break;
					}
				}
				if (dyn_share.converge)
					t++;

				if (!t && i == maximum_iter - 2)
				{
					dyn_share.converge = true;
				}
				if (t > 1 || i == maximum_iter - 1)
				{
					double solve_start = omp_get_wtime();
					
                    P_ = P_ - K_ * h_x * P_.template block<12, n>(0, 0);
					//P_ = L_ - K_x.template block<n, 12>(0, 0) * P_.template block<12, n>(0, 0);
					//}
					solve_time_3 += omp_get_wtime() - solve_start;
					return true;
				}
				

			}
			return true;
		}
		void update_iterated_dyn_share_IMU()
		{

			dyn_share_modified<scalar_type> dyn_share;
			for (int i = 0; i < maximum_iter; i++)
			{
				dyn_share.valid = true;
				h_dyn_share_modified_2(x_, dyn_share);

				Matrix<scalar_type, 6, 1> z = dyn_share.z_IMU;

				Matrix<double, 30, 6> PHT;
				Matrix<double, 6, 30> HP;
				Matrix<double, 6, 6> HPHT;
				PHT.setZero();
				HP.setZero();
				HPHT.setZero();
				for (int l_ = 0; l_ < 6; l_++)
				{
					if (!dyn_share.satu_check[l_])
					{
						PHT.col(l_) = P_.col(15 + l_) + P_.col(24 + l_);
						HP.row(l_) = P_.row(15 + l_) + P_.row(24 + l_);
					}
				}
				for (int l_ = 0; l_ < 6; l_++)
				{
					if (!dyn_share.satu_check[l_])
					{
						HPHT.col(l_) = HP.col(15 + l_) + HP.col(24 + l_);
					}
					HPHT(l_, l_) += dyn_share.R_IMU(l_); //, l);
				}
				Eigen::Matrix<double, 30, 6> K = PHT * HPHT.inverse();

				Matrix<scalar_type, n, 1> dx_ = K * z;

				P_ -= K * HP;
				x_.boxplus(dx_);
			}
			return;
		}

		void change_x(state &input_state)
		{
			x_ = input_state;

			if ((!x_.vect_state.size()) && (!x_.SO3_state.size()) && (!x_.S2_state.size()) && (!x_.SEN_state.size()))
			{
				x_.build_S2_state();
				x_.build_SO3_state();
				x_.build_vect_state();
				x_.build_SEN_state();
			}
		}

		void change_P(cov &input_cov)
		{
			P_ = input_cov;
		}

		const state &get_x() const
		{
			return x_;
		}
		const cov &get_P() const
		{
			return P_;
		}
		cov P_;
		state x_;
		state x_cov;//求predict的cov的时候用的变量

	private:
		measurement m_;
		spMt l_;
		spMt f_x_1;
		spMt f_x_2;
		cov F_x1 = cov::Identity();
		cov F_x2 = cov::Identity();
		cov L_ = cov::Identity();

		processModel *f;
		processMatrix1 *f_x;
		processMatrix2 *f_w;

		measurementMatrix1 *h_x;
		measurementMatrix2 *h_v;

		measurementMatrix1_dyn *h_x_dyn;
		measurementMatrix2_dyn *h_v_dyn;

		measurementModel_dyn_share_modified_cov *h_dyn_share_modified_1;

		measurementModel_dyn_share_modified *h_dyn_share_modified_2;

		measurementModel_dyn_share_modified *h_dyn_share_modified_3;

		measurementModel_dyn_share_modified *h_dyn_share_modified_4;
		scalar_type limit[n];

	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

} // namespace esekfom

#endif //  ESEKFOM_EKF_HPP
