#include <chrono>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "geometry_msgs/msg/twist.hpp"

using LaserScan = sensor_msgs::msg::LaserScan;
using Twist = geometry_msgs::msg::Twist;
using namespace std::placeholders;
using namespace std::chrono_literals;


class Filter : public rclcpp::Node
{
public:
    Filter() : Node("wheel_filter"),
        tget_lin_vel{0}, tget_ang_vel{0},
        ctrl_lin_vel{0}, ctrl_lin_acc{1},
        ctrl_ang_vel{0}, ctrl_ang_acc{2},
        ctrl_time{std::chrono::system_clock::now()}
    {
        laser_subscription = this->create_subscription<LaserScan>(
            "laser", rclcpp::QoS(rclcpp::SystemDefaultsQoS()), std::bind(&Filter::laser_callback, this, _1));
        scan_publisher = this->create_publisher<LaserScan>(
            "scan", 10);
        vel_subscription = this->create_subscription<Twist>(
            "cmd_vel", rclcpp::QoS(rclcpp::SystemDefaultsQoS()), std::bind(&Filter::vel_callback, this, _1));
        vel_publisher = this->create_publisher<Twist>(
            "vel", 10);
        timer = this->create_wall_timer(
            100ms, std::bind(&Filter::timer_callback, this));
    }

private:
    // Adjust LaserScan topic for Cartographer
    // https://github.com/cartographer-project/cartographer_ros/issues/354
    void laser_callback(const LaserScan::SharedPtr msg) const
    {
        auto scan = LaserScan();
        scan.header = msg->header;
        scan.angle_min = msg->angle_min;
        scan.angle_max = msg->angle_max;
        scan.angle_increment = msg->angle_increment;
        scan.time_increment = msg->time_increment;
        scan.scan_time = msg->scan_time;
        scan.range_min = msg->range_min;
        scan.range_max = msg->range_max;
        scan.intensities = msg->intensities;

        scan.ranges = msg->ranges;
        for(auto& r: scan.ranges) {
            if(r > scan.range_max) r = scan.range_max;
        }

        scan_publisher->publish(scan);
    }
    rclcpp::Subscription<LaserScan>::SharedPtr laser_subscription;
    rclcpp::Publisher<LaserScan>::SharedPtr scan_publisher;
    
    void vel_callback(const Twist::SharedPtr msg)
    {
        tget_lin_vel = msg->linear.x;
        tget_ang_vel = msg->angular.z;
    }
    float tget_lin_vel, tget_ang_vel;
    rclcpp::Subscription<Twist>::SharedPtr vel_subscription;

    // Simple velocity profiling: apply acceleration limit
    void timer_callback()
    {
        auto t = std::chrono::system_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::milliseconds>(t - ctrl_time);
        if(d.count() < 0) return;
        ctrl_time = t;

        auto diff_lin_vel = tget_lin_vel - ctrl_lin_vel;
        auto diff_lin_acc = diff_lin_vel / d.count() * 1000;
        if(abs(diff_lin_acc) < ctrl_lin_acc) ctrl_lin_vel = tget_lin_vel;
        else ctrl_lin_vel = diff_lin_vel > 0 ? ctrl_lin_vel + ctrl_lin_acc * d.count() / 1000 : ctrl_lin_vel - ctrl_lin_acc * d.count() / 1000;

        auto diff_ang_vel = tget_ang_vel - ctrl_ang_vel;
        auto diff_ang_acc = diff_ang_vel / d.count() * 1000;
        if(abs(diff_ang_acc) < ctrl_ang_acc) ctrl_ang_vel = tget_ang_vel;
        else ctrl_ang_vel = diff_ang_vel > 0 ? ctrl_ang_vel + ctrl_ang_acc * d.count() / 1000 : ctrl_ang_vel - ctrl_ang_acc * d.count() / 1000;

        auto msg = Twist();
        msg.linear.x = ctrl_lin_vel;
        msg.angular.z = ctrl_ang_vel;
        vel_publisher->publish(msg);
    }
    float ctrl_lin_vel, ctrl_lin_acc;
    float ctrl_ang_vel, ctrl_ang_acc;
    rclcpp::TimerBase::SharedPtr timer;
    std::chrono::system_clock::time_point ctrl_time;
    rclcpp::Publisher<Twist>::SharedPtr vel_publisher;
};


int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Filter>());
    rclcpp::shutdown();
    return 0;
}
