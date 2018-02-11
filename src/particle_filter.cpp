/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;

    std::default_random_engine gen;

    std::normal_distribution<double> N_x(x, std[0]);
    std::normal_distribution<double> N_y(y, std[1]);
    std::normal_distribution<double> N_theta(theta, std[2]);

    for (int i =0; i < num_particles; i++){

        Particle particle;
        particle.id = 1;
        particle.x = N_x(gen);
        particle.y = N_y(gen);
        particle.theta = N_theta(gen);
        particle.weight = 1;

        weights.push_back(particle.weight);
        particles.push_back(particle);


    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    for(int i = 0; i < num_particles; i++) {
        double new_x;
        double new_y;
        double new_theta;

        if(yaw_rate == 0){
            new_x = particles[i].x + velocity*delta_t*sin(particles[i].theta);
            new_y = particles[i].y + velocity*delta_t*cos(particles[i].theta);
            new_theta = particles[i].theta;
        }

        else {
            new_x = particles[i].x + velocity / yaw_rate * (sin(new_theta + yaw_rate * delta_t) - sin(new_theta));
            new_y = particles[i].y + velocity / yaw_rate * (cos(new_theta) - cos(new_theta + yaw_rate * delta_t));
            new_theta = particles[i].theta + yaw_rate * delta_t;
        }

        normal_distribution<double> N_x(new_x, std_pos[0]);
        normal_distribution<double> N_y(new_y, std_pos[1]);
        normal_distribution<double> N_theta(new_theta, std_pos[2]);

        particles[i].x = N_x(gen);
        particles[i].y = N_y(gen);
        particles[i].theta = N_theta(gen);

    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    int no_observation = observations.size();

    int no_predictions = predicted.size();

    for (int i = 0; i < no_observation; i++) {

        double min_distance = numeric_limits<double>::max();

        int map_id = -1;

        for (int j = 0; j < no_predictions; j++) {
            double x_distance = observations[i].x - predicted[j].x;
            double y_distance = observations[i].y - predicted[j].y;

            double distance = x_distance * x_distance + y_distance * y_distance;
            if (distance < min_distance) {
                min_distance = distance;
                map_id = predicted[j].id;
            }
        }
        observations[i].id = map_id;

    }
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    weights.clear();

    for (int i = 0; i < num_particles; i++) {

        std::vector<LandmarkObs> predicted;


        // define coordinates and theta
        double x_part = particles[i].x;
        double y_part = particles[i].y;
        double theta_part = particles[i].theta;

        //clearing garbage data
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();

        double weight = 1;

        for (int j = 0; j < observations.size(); j++) {

            double x_obs = observations[j].x;
            double y_obs = observations[j].y;
            double x_map = x_obs * cos(theta_part) - y_obs * sin(theta_part) + x_part;
            double y_map = x_obs * sin(theta_part) + y_obs * cos(theta_part) + y_part;

            if (pow(pow(x_map - x_part, 2) + pow(y_map - y_part, 2), 0.5) > sensor_range) {

                continue;

            }
                particles[i].sense_x.push_back(x_map);
                particles[i].sense_y.push_back(y_map);

                double min_range = 1000000000;

                int min_k = -1;

                for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {

                    double landmark_x = map_landmarks.landmark_list[k].x_f;
                    double landmark_y = map_landmarks.landmark_list[k].y_f;
                    double diff_x = landmark_x - x_map;
                    double diff_y = landmark_y - y_map;
                    double range = pow(pow(diff_x, 2) + pow(diff_y, 2), 0.5);


                    if (range < min_range) {
                        min_range = range;
                        min_k = k;
                    }
                }

                double landmark_x = map_landmarks.landmark_list[min_k].x_f;

                double landmark_y = map_landmarks.landmark_list[min_k].y_f;

                particles[i].associations.push_back(map_landmarks.landmark_list[min_k].id_i);

                weight = weight * exp(-0.5 * (pow((landmark_x - x_map) / std_landmark[0], 2) +
                                              pow((landmark_y - y_map) / std_landmark[1], 2))) /
                         (2 * M_PI * std_landmark[0] * std_landmark[1]);

            }
            weights.push_back(weight);

            particles[i].weight = weight;

        }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> resample_particles;

    default_random_engine gen;

    discrete_distribution<int> distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {
        int chosen = distribution(gen);
        resample_particles.push_back(particles[chosen]);
        weights.push_back(particles[chosen].weight);

    }
    particles = resample_particles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
