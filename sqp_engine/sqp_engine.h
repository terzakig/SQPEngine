//
// sqp_engine.h
//
// Implementation of SQP functionality used in the following papers:
//
// 1. "A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem" by G. Terzakis and M. Lourakis
//     a) Paper:         https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460.pdf
//     b) Supplementary: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460-supp.pdf
//
// 2. "Fast and Consistently Accurate Perspective-n-Line Pose Estimation"
//     a) Paper:         https://www.researchgate.net/publication/386377725_Fast_and_Consistently_Accurate_Perspective-n-Line_Pose_Estimation
//     b) Supplementary: https://www.researchgate.net/publication/391902738_sqpnl_supplementarypdf
//
// George Terzakis (terzakig-at-hotmail-dot-com), September 2020 (revised, May, 2025)
// Optimizations by Manolis Lourakis, February 2022, February 2024
//
// Nearest orthogonal approximation code by Manolis Lourakis, 2019

#ifndef SQP_ENGINE_H__
#define SQP_ENGINE_H__

#include <iostream>
#include <Eigen/Dense>

namespace sqp_engine
{
    //! Nearest rotation method
    enum class NearestRotationMethod
    {
        FOAM,
        SVD
    };

    //! Method for the decomposition of Omega
    enum class OmegaNullspaceMethod
    {
        RRQR,
        CPRRQR,
        SVD
    };

    /**
     * @brief Simple SVD-based nearest rotation matrix. Argument should be a *row-major* matrix representation.
     *
     * @param e The 3x3 matrix in row-major 9x1 vector form.
     * @param Returns a row-major vector representation of the nearest rotation matrix.
     */
    void NearestRotationMatrix_SVD(const Eigen::Matrix<double, 9, 1> &e, Eigen::Matrix<double, 9, 1> &r);

#if 0
    // EVD-based nearest rotation matrix. The nearest rotation matrix is given by B*inv(sqrtm(B'*B)).
    // If B'*B = U*T*U' is the eigendecomposition, the square root simplifies to sqrtm(B'*B) = (B'*B)^(1/2) = U*T^(1/2)*U'
    // and thus B*inv(sqrtm(B'*B)) equals B*U*T^(-1/2)*U'
    // See https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf  (copy: https://archive.is/INhvn)
    void NearestRotationMatrix_EVD(const Eigen::Matrix<double, 9, 1>& e, Eigen::Matrix<double, 9, 1>& r);
#endif

    /**
     * @brief Faster nearest rotation computation based on FOAM. See M. Lourakis: "An Efficient Solution to Absolute Orientation", ICPR 2016
     *        and M. Lourakis, G. Terzakis: "Efficient Absolute Orientation Revisited", IROS 2018.
     *        Solve the nearest orthogonal approximation problem
     *        i.e., given B, find R minimizing ||R-B||_F
     *
     *        The computation borrows from Markley's FOAM algorithm
     *        "Attitude Determination Using Vector Observations: A Fast Optimal Matrix Algorithm", J. Astronaut. Sci. 1993.
     *
     *        Copyright (C) 2019 Manolis Lourakis (lourakis **at** ics forth gr)
     *        Institute of Computer Science, Foundation for Research & Technology - Hellas
     *          Heraklion, Crete, Greece.
     */
    void NearestRotationMatrix_FOAM(const Eigen::Matrix<double, 9, 1> &e, Eigen::Matrix<double, 9, 1> &r);

    //! SQP configuration
    struct SQPConfig
    {
        //! Default method for nearest rotation method
        static const NearestRotationMethod DEFAULT_NEAREST_ROTATION_METHOD;
        //! Default SQP tolerance
        static const double DEFAULT_SQP_SQUARED_TOLERANCE;
        //! Determinant threshold (used to check whether the converged solution is sufficiently close to a rotation matrix)
        static const double DEFAULT_SQP_DET_THRESHOLD;
        //! Max. number of SQP steps
        static const int DEFAULT_SQP_MAX_ITERATION = 15;

        //! SQP tolerance
        double sqp_squared_tolerance;
        //! Determinant threshold (for imposing orthonormality at the end)
        double sqp_det_threshold;
        //! Max. number of SQP iterations
        int sqp_max_iteration;
        //! The nearest rotation method
        void (*NearestRotationMatrix)(const Eigen::Matrix<double, 9, 1> &e, Eigen::Matrix<double, 9, 1> &r);

        //! Construct an SQPConfig struct
        inline SQPConfig(                                                                                                                             //
            const double &_sqp_squared_tolerance = DEFAULT_SQP_SQUARED_TOLERANCE,                                                                     //
            const double &_sqp_det_threshold = DEFAULT_SQP_DET_THRESHOLD,                                                                             //
            const int _sqp_max_iteration = DEFAULT_SQP_MAX_ITERATION,                                                                                 //
            const NearestRotationMethod &_nearest_rotation_method = DEFAULT_NEAREST_ROTATION_METHOD) : sqp_squared_tolerance(_sqp_squared_tolerance), //
                                                                                                       sqp_det_threshold(_sqp_det_threshold),         //
                                                                                                       sqp_max_iteration(_sqp_max_iteration)          //
        {
            // assign nearest rotation
            SetNearestRotationMethod(_nearest_rotation_method);
        }

        //! Assign nearest rotation method
        inline void SetNearestRotationMethod(const NearestRotationMethod &_nearest_rotation_method)
        {
            if (_nearest_rotation_method == NearestRotationMethod::FOAM)
            {
                NearestRotationMatrix = NearestRotationMatrix_FOAM;
            }
            else // if ( _nearest_rotation_method == NearestRotationMethod::SVD )
            {
                NearestRotationMatrix = NearestRotationMatrix_SVD;
            }
        }
    };

    //! Solver parameters (include SQP config)
    struct SolverParameters
    {
        //! rank tolerance in orthogonality checks
        static const double DEFAULT_RANK_TOLERANCE;
        double rank_tolerance = DEFAULT_RANK_TOLERANCE;
        //! Squared error in orthogonality checks
        static const double DEFAULT_ORTHOGONALITY_SQUARED_ERROR_THRESHOLD;
        double orthogonality_squared_error_threshold = DEFAULT_ORTHOGONALITY_SQUARED_ERROR_THRESHOLD;

        //! Null space method for 9x9 data matrix
        static const OmegaNullspaceMethod DEFAULT_OMEGA_NULLSPACE_METHOD;
        OmegaNullspaceMethod omega_nullspace_method = DEFAULT_OMEGA_NULLSPACE_METHOD;

        //! Threshold in equality checks for solution vectors
        static const double DEFAULT_EQUAL_VECTORS_SQUARED_DIFF;
        double equal_vectors_squared_diff = DEFAULT_EQUAL_VECTORS_SQUARED_DIFF;
        //! Threshold in equality check between errors
        static const double DEFAULT_EQUAL_SQUARED_ERRORS_DIFF;
        double equal_squared_errors_diff = DEFAULT_EQUAL_SQUARED_ERRORS_DIFF;

        //! SQP settings
        SQPConfig sqp_config = SQPConfig();

        //! Check cheirality
        bool enable_cheirality_check = true;
        //! For cheirality checks in PnL
        static const double PI;

        //! Compute Omega's null space based on the method specified by omega_nullspace_method
        void computeNullSpace(const Eigen::Matrix<double, 9, 9>& Omega, Eigen::Matrix<double, 9, 9>& U, Eigen::Matrix<double, 9, 1>& s) const
        {
            if ( omega_nullspace_method == OmegaNullspaceMethod::RRQR )
            {
                // Rank revealing QR nullspace computation with full pivoting.
                // This is slightly less accurate compared to SVD but x2 faster
                Eigen::FullPivHouseholderQR<Eigen::Matrix<double, 9, 9> > rrqr(Omega);
                U = rrqr.matrixQ();

                Eigen::Matrix<double, 9, 9> R = rrqr.matrixQR().template triangularView<Eigen::Upper>();
                s = R.diagonal().array().abs();
            }
            else if ( omega_nullspace_method == OmegaNullspaceMethod::CPRRQR )
            {
                // Rank revealing QR nullspace computation with column pivoting.
                // This is potentially less accurate compared to RRQR but faster
                Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 9> > cprrqr(Omega);
                U = cprrqr.householderQ();

                Eigen::Matrix<double, 9, 9> R = cprrqr.matrixR().template triangularView<Eigen::Upper>();
                s = R.diagonal().array().abs();
            }
            else // if ( omega_nullspace_method == OmegaNullspaceMethod::SVD )
            {
                // SVD-based nullspace computation. This is the most accurate but slowest option
                Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(Omega, Eigen::ComputeFullU);
                U = svd.matrixU();
                s = svd.singularValues();
            }
        }
    };

    /**
     * @brief Contains an SQP rotation matrix solution
     */
    struct SQPSolution
    {
        Eigen::Matrix<double, 9, 1> r;     // Actual matrix upon convergence
        Eigen::Matrix<double, 9, 1> r_hat; // "Clean" (nearest) rotation matrix
        Eigen::Vector3d t;
        int num_iterations;
        double sq_error;
    };

    inline std::ostream &operator<<(std::ostream &os, const SQPSolution &solution)
    {
        return os << "\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
                  << "R: [ " << solution.r_hat[0] << ", " << solution.r_hat[1] << ", " << solution.r_hat[2] << ";\n"
                  << "     " << solution.r_hat[3] << ", " << solution.r_hat[4] << ", " << solution.r_hat[5] << ";\n"
                  << "     " << solution.r_hat[6] << ", " << solution.r_hat[7] << ", " << solution.r_hat[8] << " ]\n"
                  << "t: " << "[ " << solution.t[0] << "; " << solution.t[1] << "; " << solution.t[2] << " ]\n"
                  << "r: " << "[ " << solution.r_hat[0] << "; " << solution.r_hat[1] << "; " << solution.r_hat[2] << "; "
                  << solution.r_hat[3] << "; " << solution.r_hat[4] << "; " << solution.r_hat[5] << "; "
                  << solution.r_hat[6] << "; " << solution.r_hat[7] << "; " << solution.r_hat[8] << " ]\n"
                  << "squared error: " << solution.sq_error << "\n"
                  << "number of SQP iterations : " << solution.num_iterations << "\n"
                  << "-------------------------------------------------------------------------------------------------\n";
    }

    //! Transpose a rotation vector (to another rotation vector via regular 3x3 matrix transposition)
    Eigen::Matrix<double, 9, 1> TransposeRotationVector(const Eigen::Matrix<double, 9, 1> &r);

    //! Determinant of a matrix stored in a 9x1 row-major vector
    double Determinant9x1(const Eigen::Matrix<double, 9, 1> &r);

    //! Determinant of 3x3 matrix
    double Determinant3x3(const Eigen::Matrix3d &M);

    //! Invert a 3x3 symmetric matrix (using low triangle values only)
    bool InvertSymmetric3x3(     //
        const Eigen::Matrix3d Q, //
        Eigen::Matrix3d &Qinv,   //
        const double &det_threshold = 1e-10);

    /**
     * @brief Produce a distance from being orthogonal for a random 3x3 matrix (passed as a row-major 9x1 vector)
     *
     * @param a The 9x1 unrolled matrix vector
     * @return double The error
     */
    double OrthogonalityError(const Eigen::Matrix<double, 9, 1> &a);

    /**
     * @brief Compute the 3D null space (N) and 6D normal space (H) of the constraint Jacobian at a 9D vector r
     *        (r is not necessarily a rotation but it must represent a rank-3 matrix)
     *        NOTE: K is lower-triangular, so upper triangle may contain trash (is not filled by the function)...
     *
     * @param Omega The original QCQP 9x9 matrix
     * @param r The rotation matrix estimate (noty necessarily strictly a rotation)
     * @param H The return row space of the constraint Jacobian as a 9x6 matrix
     * @param N The null space of the constraint Jacobian as a 9x3 matrix
     * @param K The product J*H where J is the 6x9 Jacobian of the **orthonormality** (i.e., O(3), NOT SO(3)) constraints.
     * @param norm_threshold Used to discard columns of Pn when finding null space threshold for column vector norm (of Pn). Default value, 0.1.
     */
    void RowAndNullSpace(                         //
        const Eigen::Matrix<double, 9, 9> &Omega, // The original QCQP matrix (does not change during SQP iteration)
        const Eigen::Matrix<double, 9, 1> &r,     // The current SQP estimate
        Eigen::Matrix<double, 9, 6> &H,           // Row space
        Eigen::Matrix<double, 9, 3> &N,           // Null space
        Eigen::Matrix<double, 6, 6> &K,           // J*Q (J - Jacobian of constraints)
        const double &norm_threshold = 0.1        // Used to discard columns of Pn when finding null space threshold for column vector norm (of Pn)
    );

    // Solve A*x=b for 3x3 SPD A.
    // The solution involves computing a lower triangular sqrt-free Cholesky factor
    // A=L*D*L' (L has ones on its diagonal, D is diagonal).
    //
    // Only the lower triangular part of A is accessed.
    //
    // The function returns 0 if successful, non-zero otherwise
    //
    // see http://euler.nmt.edu/~brian/ldlt.html
    //
    /**
     * @brief Solve A*x=b for 3x3 SPD A. The solution involves computing a lower triangular
     *        sqrt-free Cholesky factor A=L*D*L' (L has ones on its diagonal, D is diagonal).
     *
     *        Only the lower triangular part of A is accessed.
     *
     *       see http://euler.nmt.edu/~brian/ldlt.html
     *
     * @param A The data matrix
     * @param b The data vector
     * @param x The solution
     * @return The function returns 0 if successful, non-zero otherwise
     */
    int AxbSolveLDLt3x3(          //
        const Eigen::Matrix3d &A, //
        const Eigen::Vector3d &b, //
        Eigen::Vector3d &x);

    /**
     * @brief Solve the SQP system **efficiently** (avoids inversion; executed in each SQP step).
     *
     * @param Omega The original QCQP matrix.
     * @param r The current rotation estimate as a 9x1 row-major vector
     * @param delta Stores the return perturbation of r (the solution of the system)
     */
    void SolveSQPSystem(const Eigen::Matrix<double, 9, 9> &Omega, const Eigen::Matrix<double, 9, 1> &r, Eigen::Matrix<double, 9, 1> &delta);

    /**
     * @brief Run sequential quadratic programming with orthogonality constraints
     *
     * @param Omega The original QCQP matrix
     * @param r0 Initial rotation as a 9x1 (row-major) vector
     * @return An SQPSolution struct
     */
    SQPSolution RunSQP(const Eigen::Matrix<double, 9, 9> &Omega, const Eigen::Matrix<double, 9, 1> &r0, const SQPConfig &config = SQPConfig());

} // namespace sqp_engine

#endif
