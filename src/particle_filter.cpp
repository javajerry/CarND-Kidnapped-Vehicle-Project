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
	default_random_engine gen;
	num_particles = 1;

	// define normal distributions for sensor noise
	normal_distribution<double> nd_x (0, std[0]);
  	normal_distribution<double> nd_y (0, std[1]);
  	normal_distribution<double> nd_theta (0, std[2]);

  	// initialize particles
  	for (int i = 0; i < num_particles; i++) {
    	Particle p;
    	p.id = i;
    	p.x = x;
    	p.y = y;
    	p.theta = theta;
    	p.weight = 1.0;

    	// add noise
    	// where "gen" is the random engine initialized 
    	p.x += nd_x (gen);
    	p.y += nd_y (gen);
    	p.theta += nd_theta (gen);

    	particles.push_back(p);
  		weights.push_back(p.weight);

  	}

  	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	// ormal distributions for adding noise
  	normal_distribution<double> nd_x(0, std_pos[0]);
  	normal_distribution<double> nd_y(0, std_pos[1]);
  	normal_distribution<double> nd_theta(0, std_pos[2]);

  	for (int i = 0; i < num_particles; i++) {

    	// calculate new state
    	if (fabs(yaw_rate) < 0.00001) {  
      		particles[i].x += velocity * delta_t * cos(particles[i].theta);
      		particles[i].y += velocity * delta_t * sin(particles[i].theta);
    	} 
    	else {
      		particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      		particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      		particles[i].theta += yaw_rate * delta_t;
    	}

    	// add noise
    	particles[i].x += nd_x(gen);
    	particles[i].y += nd_y(gen);
    	particles[i].theta += nd_theta(gen);
  	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i) {
    	LandmarkObs& o = observations[i];
    	double min_dist = numeric_limits<double>::max();
    	for (int j = 0; j < predicted.size(); ++j) {
    		LandmarkObs p = predicted[j];
    		double distance = dist(o.x, o.y, p.x, p.y);
    		if (distance < min_dist) {
    			min_dist = distance;
    			// set to nearest neighbor
    			o.id = j;
    		}
    	}
    }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	// for each particle...
  	for (int i = 0; i < num_particles; i++) {

		Particle &p = particles[i];
		double p_x = p.x;
    	double p_y = p.y;
    	double p_theta = p.theta;

		// transform observations to map coordinate
		vector<LandmarkObs> transformed_os;
		for (int j = 0; j < observations.size(); ++j) {
			LandmarkObs obs = observations[j];
			LandmarkObs transformed_obs;
			transformed_obs.x = obs.x * cos(p_theta) - obs.y * sin(p_theta) + p_x;
			transformed_obs.y = obs.x * sin(p_theta) + obs.y * cos(p_theta) + p_y;

			transformed_os.push_back(transformed_obs);
		}

		// predict mesurements
		vector<LandmarkObs> predictions;
		for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {

			// get id,x,y coordinates
    		int lm_id = map_landmarks.landmark_list[k].id_i;
      		float lm_x = map_landmarks.landmark_list[k].x_f;
      		float lm_y = map_landmarks.landmark_list[k].y_f;

			if (dist(p_x, p_y, lm_x, lm_y) < sensor_range) {
				LandmarkObs pred = {lm_id, lm_x, lm_y};
				predictions.push_back(pred);
			}
		} //

		dataAssociation(predictions, transformed_os);

		// update particle weight
		// initialize to 1
		p.weight = 1;
		for (int l = 0; l < transformed_os.size(); ++l) {
			LandmarkObs obs = transformed_os[l];
			LandmarkObs pred = predictions[obs.id];

			double std_x = std_landmark[0]; // sigma_x
			double std_y = std_landmark[1]; // sigma_y
	
			double obs_w =  exp( -( pow(pred.x-obs.x,2)/(2*pow(std_x, 2)) + (pow(pred.y-obs.y,2)/(2*pow(std_y, 2))) ) )/(2*M_PI*std_x*std_y);

			p.weight *= obs_w;
		}
	
		weights[i] = p.weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//resample wheel
	vector<Particle> new_particles;
	default_random_engine gen;

  	// generate random starting index 
  	uniform_int_distribution<int> uniintdist(0, num_particles-1);
  	auto index = uniintdist(gen);

  	// get max weight
  	double max_weight = *max_element(weights.begin(), weights.end());

  	// uniform random distribution [0.0, max_weight)
  	uniform_real_distribution<double> unirealdist(0.0, max_weight);

  	double beta = 0.0;

  	// resampling wheel!
  	for (int i = 0; i < num_particles; i++) {
    	beta += unirealdist(gen) * 2.0;
    	while (beta > weights[index]) {
      		beta -= weights[index];
      		index = (index + 1) % num_particles;
    	}
    	new_particles.push_back(particles[index]);
  	}

  	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
