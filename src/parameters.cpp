#include "parameters.h"

bool is_first_frame = true;
double lidar_end_time = 0.0, first_lidar_time = 0.0;
double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0;
int pcd_index = 0;
IVoxType::Options ivox_options_;
int ivox_nearby_type = 18;

std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
state_input state_in;
state_output state_out;
std::string lid_topic("/livox/lidar"), imu_topic("/livox/imu");
int frame_merge_num = 1;
bool prop_at_freq_of_imu = true, check_satu = true;
bool use_imu_as_input = false, space_down_sample = true, publish_odometry_without_downsample = false;
int  init_map_size = 100;
double match_s = 81, satu_acc = 3.0, satu_gyro = 35.0;
float  plane_thr = 0.1f;
double filter_size_surf_min = 0.5, filter_size_map_min = 0.5;
bool   imu_en = true;
double imu_time_inte = 0.005;
double laser_point_cov = 0.01, acc_norm = 1.0;
double vel_cov = 20.0, acc_cov_input = 0.1, gyr_cov_input = 0.01;
double gyr_cov_output = 1000.0, acc_cov_output = 500.0, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double imu_meas_acc_cov = 0.1, imu_meas_omg_cov = 0.1;
int    pcd_save_interval = -1;
std::vector<double> gravity_init(3, 0.0), gravity{0.0, 0.0, -9.81};
bool   runtime_pos_log = false, pcd_save_en = false, path_en = true, extrinsic_est_en = true;
bool   scan_pub_en = true, scan_body_pub_en = false;
shared_ptr<Preprocess> p_pre;
shared_ptr<ImuProcess> p_imu;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;
double time_diff_lidar_to_imu = 0.0;

double lidar_time_inte = 0.1, first_imu_time = 0.0;

MeasureGroup Measures;

ofstream fout_out, fout_imu_pbp;

double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];

void readParameters(rclcpp::Node * node)
{
  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  auto declare_get_bool = [&](const char * name, bool & out) {
    node->declare_parameter(name, out);
    node->get_parameter(name, out);
  };
  auto declare_get_int = [&](const char * name, int & out) {
    node->declare_parameter(name, out);
    node->get_parameter(name, out);
  };
  auto declare_get_double = [&](const char * name, double & out) {
    node->declare_parameter(name, out);
    node->get_parameter(name, out);
  };
  auto declare_get_float = [&](const char * name, float & out) {
    node->declare_parameter(name, out);
    node->get_parameter(name, out);
  };
  auto declare_get_string = [&](const char * name, std::string & out) {
    node->declare_parameter(name, out);
    node->get_parameter(name, out);
  };
  auto declare_get_vec = [&](const char * name, std::vector<double> & out) {
    node->declare_parameter(name, out);
    node->get_parameter(name, out);
  };

  declare_get_bool("prop_at_freq_of_imu", prop_at_freq_of_imu);
  declare_get_bool("use_imu_as_input", use_imu_as_input);
  declare_get_bool("check_satu", check_satu);
  declare_get_int("init_map_size", init_map_size);
  declare_get_bool("space_down_sample", space_down_sample);
  declare_get_double("mapping.satu_acc", satu_acc);
  declare_get_double("mapping.satu_gyro", satu_gyro);
  declare_get_double("mapping.acc_norm", acc_norm);
  declare_get_float("mapping.plane_thr", plane_thr);
  declare_get_string("common.lid_topic", lid_topic);
  declare_get_string("common.imu_topic", imu_topic);
  declare_get_int("common.frame_merge_num", frame_merge_num);
  declare_get_double("common.time_diff_lidar_to_imu", time_diff_lidar_to_imu);
  declare_get_double("filter_size_surf", filter_size_surf_min);
  declare_get_double("filter_size_map", filter_size_map_min);
  declare_get_bool("mapping.imu_en", imu_en);
  declare_get_bool("mapping.extrinsic_est_en", extrinsic_est_en);
  declare_get_double("mapping.imu_time_inte", imu_time_inte);
  declare_get_double("mapping.lidar_meas_cov", laser_point_cov);
  declare_get_double("mapping.acc_cov_input", acc_cov_input);
  declare_get_double("mapping.vel_cov", vel_cov);
  declare_get_double("mapping.gyr_cov_input", gyr_cov_input);
  declare_get_double("mapping.gyr_cov_output", gyr_cov_output);
  declare_get_double("mapping.acc_cov_output", acc_cov_output);
  declare_get_double("mapping.b_gyr_cov", b_gyr_cov);
  declare_get_double("mapping.b_acc_cov", b_acc_cov);
  declare_get_double("mapping.imu_meas_acc_cov", imu_meas_acc_cov);
  declare_get_double("mapping.imu_meas_omg_cov", imu_meas_omg_cov);
  declare_get_double("preprocess.blind", p_pre->blind);
  declare_get_double("mapping.match_s", match_s);
  declare_get_vec("mapping.gravity", gravity);
  declare_get_vec("mapping.gravity_init", gravity_init);
  declare_get_vec("mapping.extrinsic_T", extrinT);
  declare_get_vec("mapping.extrinsic_R", extrinR);
  declare_get_bool("odometry.publish_odometry_without_downsample", publish_odometry_without_downsample);
  declare_get_bool("publish.path_en", path_en);
  declare_get_bool("publish.scan_publish_en", scan_pub_en);
  declare_get_bool("publish.scan_bodyframe_pub_en", scan_body_pub_en);
  declare_get_bool("runtime_pos_log_enable", runtime_pos_log);
  declare_get_bool("pcd_save.pcd_save_en", pcd_save_en);
  declare_get_int("pcd_save.interval", pcd_save_interval);

  declare_get_double("mapping.lidar_time_inte", lidar_time_inte);

  declare_get_float("mapping.ivox_grid_resolution", ivox_options_.resolution_);
  declare_get_int("ivox_nearby_type", ivox_nearby_type);
  if (ivox_nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (ivox_nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (ivox_nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (ivox_nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
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
