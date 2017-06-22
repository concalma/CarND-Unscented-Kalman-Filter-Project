#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#undef _DEBUG
#undef _DEBUG_NIS
#define _DEBUG_NIS_RADAR


/** NIS plotting is at google sheets here:
 *
 * https://docs.google.com/spreadsheets/d/1bt7vY7Q6pRQz5opdKBC0UnvpZ2daR-F1Xa22Kr059SQ/edit?usp=sharing
 *
 * PDF provided in src directory. This NIS was obtained after proper tweaking of std_a_ and std_yawdd_.
 *
 */

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1; // updated values 

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/8; // Updated values

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  n_x_ = 5; // state dimension
  n_aug_ = 7;
  lambda_ = 3-n_aug_; // lambda init. as suggested  updated
  previous_timestamp_ = 0;
}

UKF::~UKF() {}

double UKF::normalizeAngle(double a) {
         return a - (2*M_PI) * floor((a + M_PI ) / (2*M_PI) );
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if (!is_initialized_) {
        /**
         * Initialize the state ekf_.x_ with the first measurement.
         * Create the covariance matrix.
         * Remember: you'll need to convert radar from polar to cartesian coordinates.
         */


        // first measurement
        x_ << 1, 1, 1, 1, 1;

        // state covariance matrix P. We use placeholder values '1' for speed, '1' for uncertainty in velocity
        // yaw, and yaw_dot
        P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0,1, 0,
            0 ,0 ,0 ,0, 1;


        // Updating first state directly with first measurement
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
              Convert radar from polar to cartesian coordinates and initialize state.
              */
            float ro =    meas_package.raw_measurements_[0];
            float theta = meas_package.raw_measurements_[1];
            float rodot = meas_package.raw_measurements_[2];

            x_ << ro*cos(theta), ro*sin(theta), 0, 0, 0 ;  

        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            //set the state with the initial location and zero velocity
            x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;  


        }

        // done initializing, no need to predict or update
        is_initialized_ = true;
        previous_timestamp_ = meas_package.timestamp_;
        return;
    }

    // if no previous timestamp take measurement as is. Consider dt=0. This triggers F = Id and Q =0 first time around
    double dt = 0;
    if( previous_timestamp_ != 0 ) {
        //compute the time elapsed between the current and previous measurements
        dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    }

    previous_timestamp_ = meas_package.timestamp_;
    
    Prediction(dt);

#ifdef _DEBUG
    cout << "predicted:" << endl;
    cout << P_ << endl << x_ << endl;
#endif

    if( use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR ) {
        UpdateRadar(meas_package);
#ifdef _DEBUG
        cout << "updated radar:" << endl;
#endif
    } 

    if( use_laser_ &&  meas_package.sensor_type_ == MeasurementPackage::LASER ) {
        UpdateLidar(meas_package);
#ifdef _DEBUG
        cout << "updated laser:" << endl;
#endif
    }

#ifdef _DEBUG
    cout << P_ << endl << x_ << endl;
#endif
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // AUGMENTING THE STATE and CREATE SIGMA POINTS
  // --------------------------------------------
  // 
  // Input: x_, P_
  // Ouput: Xsig_aug
  //
  // We need to augment our state vector with noise vectors because the prediction
  // function is not linear and we need to "pass the noise through the prediction func"
  VectorXd x_aug = VectorXd(7);
  MatrixXd P_aug = MatrixXd(7,7);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2* n_aug_+1);

  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
      Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_ ) * L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_ ) * L.col(i);
  }

  // PREDICTING SIGMA POINTS
  // -----------------------
  // 
  // Input: Xsig_aug    (7x15)
  // Output: Xsig_pred  (5x15)
  
  // predict sigma points
  // iterate over every colum of Xsig_aug, the augmented sigma points
  Xsig_pred_ = MatrixXd( n_x_, 2*n_aug_ + 1);

  for (int i = 0; i< 2*n_aug_+1; i++)
  {
      // pull vals
      double p_x = Xsig_aug(0,i);
      double p_y = Xsig_aug(1,i);
      double v   =   Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yawd = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_yawdd = Xsig_aug(6,i);

      //predicted state values
      double px_p, py_p;

      // avoid division by zero
      if ( fabs(yawd) > 0.001 ) {
          px_p = p_x + v/yawd * ( sin (yaw + yawd * delta_t ) - sin(yaw));
          py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw + yawd*delta_t ) );
      }
      else {
          px_p = p_x + v*delta_t*cos(yaw);
          py_p = p_y + v*delta_t*sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd*delta_t;
      double yawd_p = yawd;

      //add noise
      px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
      py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
      v_p = v_p + nu_a*delta_t;

      yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
      yawd_p = yawd_p + nu_yawdd*delta_t;

      //write predicted sigma point into right column
      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yawd_p;
  } 


  // CALCULATE PREDICTED STATE FROM PREDICTED SIGMA POINTS
  // -----------------------------------------------------
  //
  // Input:   X_sig_pred   (5x15)
  // Output:  x_, P_ (newly predicted)  [5, 5x5]
  
  weights_ = VectorXd(2*n_aug_+1);

  // set weights
  double weight_0 = lambda_ / (lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_ +1; i++) {  //2n+1 weights
      double weight = 0.5 / ( n_aug_ + lambda_ );
      weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
      x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      //YAW (psi) angle normalization
      x_diff(3) = normalizeAngle(x_diff(3));


      P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
    VectorXd z = VectorXd(2);
    z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];



    MatrixXd H_laser_ = MatrixXd(2, 5);
    MatrixXd R_laser_ = MatrixXd(2, 2);



    /**
     * Finish initializing the FusionEKF.
     * Set the process and measurement noises
     */
    //measurement covariance matrix - laser
    R_laser_ <<     std_laspx_*std_laspx_, 0,
                    0, std_laspy_*std_laspy_;
    H_laser_ << 1,0,0,0,0,
             0,1,0,0,0;

        
     /**
     * update the state by using Kalman Filter equations
     * H_ and R_ must be set before this
     */
    VectorXd z_pred = H_laser_ * x_;                // z_pred = 2x1
    VectorXd y = z - z_pred;                        // y = 2x1 
    MatrixXd Ht = H_laser_.transpose();             // H_laser=2x5, Ht=5x2
    MatrixXd S = H_laser_ * P_ * Ht + R_laser_;     // R_laser_=2x2   S = 2x2
    MatrixXd Si = S.inverse();                      // Si = 2x2
    MatrixXd PHt = P_ * Ht;                         // P_= 5x5 Ht=5x2
    MatrixXd K = PHt * Si;                          // PHt = 5x2  Si=2x2. K=5x2

    //new estimate
    x_ = x_ + (K * y);                              // y=2x1 K=5x2 x_=5x1
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_laser_) * P_;                   

    //NIS 
    // zdiff = z_k+1 - z_k+1_est = y
    // epsilon (Normalized Innovation Squared) = zdiff_t * S_inverse * zdiff.    1x2 * 2x2 * 2x1
    double epsilon = y.transpose() * Si * y;

#ifdef _DEBUG_NIS
    cout << epsilon << endl;
#endif

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
    int n_z = 3;
    VectorXd z = meas_package.raw_measurements_;

    MatrixXd Zsig = MatrixXd(3,2*n_aug_+1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < 2*n_aug_+1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        z_diff(1) = normalizeAngle( z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
    R <<    std_radr_*std_radr_, 0, 0,
      0, std_radphi_*std_radphi_, 0,
      0, 0,std_radrd_*std_radrd_;
    S = S + R;
   

    
    // calculate cross correlation matrix. Cross correlation between sigma points in state space
    // and measurement space. Defined in Lesson 7.28
    MatrixXd Tc = MatrixXd( n_x_, n_z );
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = normalizeAngle( z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = normalizeAngle( x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    z_diff(1) = normalizeAngle( z_diff(1));

    // NIS calculation
    // zdiff = z_k+1 - z_k+1_est = y
    // epsilon (Normalized Innovation Squared) = zdiff_t * S_inverse * zdiff.    1x2 * 2x2 * 2x1
    double epsilon = z_diff.transpose() * S.inverse() * z_diff;

#ifdef _DEBUG_NIS_RADAR
    cout << epsilon << endl;
#endif



    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
}
