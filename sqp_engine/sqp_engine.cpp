#include "sqp_engine.h"

namespace sqp_engine
{
    const double SolverParameters::DEFAULT_RANK_TOLERANCE = 1e-7;
    const OmegaNullspaceMethod SolverParameters::DEFAULT_OMEGA_NULLSPACE_METHOD = OmegaNullspaceMethod::RRQR;
    const double SolverParameters::DEFAULT_ORTHOGONALITY_SQUARED_ERROR_THRESHOLD = 1e-8;
    const double SolverParameters::DEFAULT_EQUAL_VECTORS_SQUARED_DIFF = 1e-10;
    const double SolverParameters::DEFAULT_EQUAL_SQUARED_ERRORS_DIFF = 1e-6;
    const double SolverParameters::PI = 3.14159265358979323846;

    const double SQPConfig::DEFAULT_SQP_SQUARED_TOLERANCE = 1e-10;
    const double SQPConfig::DEFAULT_SQP_DET_THRESHOLD = 1.001;
    const NearestRotationMethod SQPConfig::DEFAULT_NEAREST_ROTATION_METHOD = NearestRotationMethod::FOAM;

    Eigen::Matrix<double, 9, 1> TransposeRotationVector(const Eigen::Matrix<double, 9, 1> &r)
    {
        Eigen::Matrix<double, 9, 1> rt;
        rt << r[0], r[3], r[6], //
            r[1], r[4], r[7],   //
            r[2], r[5], r[8];
        return rt;
    }

    double Determinant9x1(const Eigen::Matrix<double, 9, 1> &r)
    {
        return (r[0] * r[4] * r[8] + r[1] * r[5] * r[6] + r[2] * r[3] * r[7]) - (r[6] * r[4] * r[2] + r[7] * r[5] * r[0] + r[8] * r[3] * r[1]);
    }

    double Determinant3x3(const Eigen::Matrix<double, 3, 3> &M)
    {
        return M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) - M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) + M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
    }

    bool InvertSymmetric3x3(                 //
        const Eigen::Matrix<double, 3, 3> Q, //
        Eigen::Matrix<double, 3, 3> &Qinv,   //
        const double &det_threshold)
    {
        // 1. Get the elements of the matrix
        double a = Q(0, 0),
               b = Q(1, 0), d = Q(1, 1),
               c = Q(2, 0), e = Q(2, 1), f = Q(2, 2);

        // 2. Determinant
        double t2, t4, t7, t9, t12;
        t2 = e * e;
        t4 = a * d;
        t7 = b * b;
        t9 = b * c;
        t12 = c * c;
        double det = -t4 * f + a * t2 + t7 * f - 2.0 * t9 * e + t12 * d;

        if (fabs(det) < det_threshold)
        {
            Qinv = Q.completeOrthogonalDecomposition().pseudoInverse();
            return false;
        } // fall back to pseudoinverse

        // 3. Inverse
        double t15, t20, t24, t30;
        t15 = 1.0 / det;
        t20 = (-b * f + c * e) * t15;
        t24 = (b * e - c * d) * t15;
        t30 = (a * e - t9) * t15;
        Qinv(0, 0) = (-d * f + t2) * t15;
        Qinv(0, 1) = Qinv(1, 0) = -t20;
        Qinv(0, 2) = Qinv(2, 0) = -t24;
        Qinv(1, 1) = -(a * f - t12) * t15;
        Qinv(1, 2) = Qinv(2, 1) = t30;
        Qinv(2, 2) = -(t4 - t7) * t15;

        return true;
    }

    void NearestRotationMatrix_SVD(const Eigen::Matrix<double, 9, 1> &e, Eigen::Matrix<double, 9, 1> &r)
    {
        // const Eigen::Matrix<double, 3, 3> E = e.reshaped(3, 3);
        const Eigen::Map<const Eigen::Matrix<double, 3, 3>> E(e.data(), 3, 3);
        Eigen::JacobiSVD<Eigen::Matrix<double, 3, 3>> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        double detUV = Determinant3x3(svd.matrixU()) * Determinant3x3(svd.matrixV());
        // so we return back a row-major vector representation of the orthogonal matrix
        Eigen::Matrix<double, 3, 3> R = svd.matrixU() * Eigen::Matrix<double, 3, 1>({1, 1, detUV}).asDiagonal() * svd.matrixV().transpose();
        r = Eigen::Map<Eigen::Matrix<double, 9, 1>>(R.data(), 9, 1);
    }

#if 0
    // EVD-based nearest rotation matrix. The nearest rotation matrix is given by B*inv(sqrtm(B'*B)).
    // If B'*B = U*T*U' is the eigendecomposition, the square root simplifies to sqrtm(B'*B) = (B'*B)^(1/2) = U*T^(1/2)*U'
    // and thus B*inv(sqrtm(B'*B)) equals B*U*T^(-1/2)*U'
    // See https://people.eecs.berkeley.edu/~wkahan/Math128/NearestQ.pdf
    void NearestRotationMatrix_EVD(const Eigen::Matrix<double, 9, 1>& e, Eigen::Matrix<double, 9, 1>& r)
    {
      const Eigen::Map<const Eigen::Matrix<double, 3, 3>> B(e.data(), 3, 3);
      const Eigen::Matrix<double, 3, 3> BtB = B.transpose()*B;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> es(BtB);
      const Eigen::Matrix<double, 3, 3> U = es.eigenvectors();
      //const Eigen::Matrix<double, 3, 3> T = es.eigenvalues().asDiagonal();
      const Eigen::Matrix<double, 3, 3> S = es.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal(); // T^(-1/2), i.e. T=inv(S*S)

      // the inverse square root of BtB is U*S*U'
      Eigen::Matrix<double, 3, 3> R = B * U * S * U.transpose();
      if (R.determinant() < 0.0) R = -R;
      r = Eigen::Map<Eigen::Matrix<double, 9, 1>>(R.data(), 9, 1);
    }
#endif

    void NearestRotationMatrix_FOAM(const Eigen::Matrix<double, 9, 1> &e, Eigen::Matrix<double, 9, 1> &r)
    {
        int i;
        const double *B = e.data();
        double l, lprev, detB, Bsq, adjBsq, adjB[9];

        // det(B)
        detB = (B[0] * B[4] * B[8] - B[0] * B[5] * B[7] - B[1] * B[3] * B[8]) + (B[2] * B[3] * B[7] + B[1] * B[6] * B[5] - B[2] * B[6] * B[4]);
        if (fabs(detB) < 1E-04)
        { // singular, let SVD handle it
            NearestRotationMatrix_SVD(e, r);
            return;
        }

        // B's adjoint
        adjB[0] = B[4] * B[8] - B[5] * B[7];
        adjB[1] = B[2] * B[7] - B[1] * B[8];
        adjB[2] = B[1] * B[5] - B[2] * B[4];
        adjB[3] = B[5] * B[6] - B[3] * B[8];
        adjB[4] = B[0] * B[8] - B[2] * B[6];
        adjB[5] = B[2] * B[3] - B[0] * B[5];
        adjB[6] = B[3] * B[7] - B[4] * B[6];
        adjB[7] = B[1] * B[6] - B[0] * B[7];
        adjB[8] = B[0] * B[4] - B[1] * B[3];

        // ||B||^2, ||adj(B)||^2
        Bsq = (B[0] * B[0] + B[1] * B[1] + B[2] * B[2]) + (B[3] * B[3] + B[4] * B[4] + B[5] * B[5]) + (B[6] * B[6] + B[7] * B[7] + B[8] * B[8]);
        adjBsq = (adjB[0] * adjB[0] + adjB[1] * adjB[1] + adjB[2] * adjB[2]) + (adjB[3] * adjB[3] + adjB[4] * adjB[4] + adjB[5] * adjB[5]) + (adjB[6] * adjB[6] + adjB[7] * adjB[7] + adjB[8] * adjB[8]);

        // compute l_max with Newton-Raphson from FOAM's characteristic polynomial, i.e. eq.(23) - (26)
        l = 0.5 * (Bsq + 3.0); // 1/2*(trace(B*B') + trace(eye(3)))
        if (detB < 0.0)
            l = -l; // trB & detB have opposite signs!
        for (i = 15, lprev = 0.0; fabs(l - lprev) > 1E-12 * fabs(lprev) && i > 0; --i)
        {
            double tmp, p, pp;

            tmp = (l * l - Bsq);
            p = (tmp * tmp - 8.0 * l * detB - 4.0 * adjBsq);
            pp = 8.0 * (0.5 * tmp * l - detB);

            lprev = l;
            l -= p / pp;
        }

        // the rotation matrix equals ((l^2 + Bsq)*B + 2*l*adj(B') - 2*B*B'*B) / (l*(l*l-Bsq) - 2*det(B)), i.e. eq.(14) using (18), (19)
        {
            // compute (l^2 + Bsq)*B
            double tmp[9], BBt[9], denom;
            const double a = l * l + Bsq;

            // BBt=B*B'
            BBt[0] = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
            BBt[1] = B[0] * B[3] + B[1] * B[4] + B[2] * B[5];
            BBt[2] = B[0] * B[6] + B[1] * B[7] + B[2] * B[8];

            BBt[3] = BBt[1];
            BBt[4] = B[3] * B[3] + B[4] * B[4] + B[5] * B[5];
            BBt[5] = B[3] * B[6] + B[4] * B[7] + B[5] * B[8];

            BBt[6] = BBt[2];
            BBt[7] = BBt[5];
            BBt[8] = B[6] * B[6] + B[7] * B[7] + B[8] * B[8];

            // tmp=BBt*B
            tmp[0] = BBt[0] * B[0] + BBt[1] * B[3] + BBt[2] * B[6];
            tmp[1] = BBt[0] * B[1] + BBt[1] * B[4] + BBt[2] * B[7];
            tmp[2] = BBt[0] * B[2] + BBt[1] * B[5] + BBt[2] * B[8];

            tmp[3] = BBt[3] * B[0] + BBt[4] * B[3] + BBt[5] * B[6];
            tmp[4] = BBt[3] * B[1] + BBt[4] * B[4] + BBt[5] * B[7];
            tmp[5] = BBt[3] * B[2] + BBt[4] * B[5] + BBt[5] * B[8];

            tmp[6] = BBt[6] * B[0] + BBt[7] * B[3] + BBt[8] * B[6];
            tmp[7] = BBt[6] * B[1] + BBt[7] * B[4] + BBt[8] * B[7];
            tmp[8] = BBt[6] * B[2] + BBt[7] * B[5] + BBt[8] * B[8];

            // compute R as (a*B + 2*(l*adj(B)' - tmp))*denom; note that adj(B')=adj(B)'
            denom = l * (l * l - Bsq) - 2.0 * detB;
            denom = 1.0 / denom;
            r(0) = (a * B[0] + 2.0 * (l * adjB[0] - tmp[0])) * denom;
            r(1) = (a * B[1] + 2.0 * (l * adjB[3] - tmp[1])) * denom;
            r(2) = (a * B[2] + 2.0 * (l * adjB[6] - tmp[2])) * denom;

            r(3) = (a * B[3] + 2.0 * (l * adjB[1] - tmp[3])) * denom;
            r(4) = (a * B[4] + 2.0 * (l * adjB[4] - tmp[4])) * denom;
            r(5) = (a * B[5] + 2.0 * (l * adjB[7] - tmp[5])) * denom;

            r(6) = (a * B[6] + 2.0 * (l * adjB[2] - tmp[6])) * denom;
            r(7) = (a * B[7] + 2.0 * (l * adjB[5] - tmp[7])) * denom;
            r(8) = (a * B[8] + 2.0 * (l * adjB[8] - tmp[8])) * denom;
        }

        // double R[9];
        // r=Eigen::Map<Eigen::Matrix<double, 9, 1>>(R);
    }

    double OrthogonalityError(const Eigen::Matrix<double, 9, 1> &a)
    {
        double sq_norm_a1 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2],
               sq_norm_a2 = a[3] * a[3] + a[4] * a[4] + a[5] * a[5],
               sq_norm_a3 = a[6] * a[6] + a[7] * a[7] + a[8] * a[8];
        double dot_a1a2 = a[0] * a[3] + a[1] * a[4] + a[2] * a[5],
               dot_a1a3 = a[0] * a[6] + a[1] * a[7] + a[2] * a[8],
               dot_a2a3 = a[3] * a[6] + a[4] * a[7] + a[5] * a[8];

        return ((sq_norm_a1 - 1) * (sq_norm_a1 - 1) + (sq_norm_a2 - 1) * (sq_norm_a2 - 1)) + ((sq_norm_a3 - 1) * (sq_norm_a3 - 1) +
                                                                                              2 * (dot_a1a2 * dot_a1a2 + dot_a1a3 * dot_a1a3 + dot_a2a3 * dot_a2a3));
    }

    void RowAndNullSpace(                         //
        const Eigen::Matrix<double, 9, 9> &Omega, // The original QCQP matrix
        const Eigen::Matrix<double, 9, 1> &r,     // The current SQP estimate
        Eigen::Matrix<double, 9, 6> &H,           // Row space
        Eigen::Matrix<double, 9, 3> &N,           // Null space
        Eigen::Matrix<double, 6, 6> &K,           // = J*H (J - The 6x9 Jacobian of constraints)
        const double &norm_threshold              // Used to discard columns of Pn when finding null space
        )                                         // threshold for column vector norm (of Pn)
    {
        // Applying Gram-Schmidt orthogonalization on the Jacobian.
        // The steps are fixed here to take advantage of the sparse form of the matrix
        //
        H = Eigen::Matrix<double, 9, 6>::Zero();

        // 1. H(:, 1)
        double norm_r1 = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        double inv_norm_r1 = norm_r1 > 1e-5 ? 1.0 / norm_r1 : 0.0;
        H(0, 0) = r[0] * inv_norm_r1;
        H(1, 0) = r[1] * inv_norm_r1;
        H(2, 0) = r[2] * inv_norm_r1;
        K(0, 0) = 2 * norm_r1;

        // 2. H(:, 2)
        double norm_r2 = sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5]);
        double inv_norm_r2 = 1.0 / norm_r2;
        H(3, 1) = r[3] * inv_norm_r2;
        H(4, 1) = r[4] * inv_norm_r2;
        H(5, 1) = r[5] * inv_norm_r2;
        K(1, 0) = 0;
        K(1, 1) = 2 * norm_r2;

        // 3. H(:, 3) = (r3'*H(:, 2))*H(:, 2) - (r3'*H(:, 1))*H(:, 1) ; H(:, 3) = H(:, 3)/norm(H(:, 3))
        double norm_r3 = sqrt(r[6] * r[6] + r[7] * r[7] + r[8] * r[8]);
        double inv_norm_r3 = 1.0 / norm_r3;
        H(6, 2) = r[6] * inv_norm_r3;
        H(7, 2) = r[7] * inv_norm_r3;
        H(8, 2) = r[8] * inv_norm_r3;
        K(2, 0) = K(2, 1) = 0;
        K(2, 2) = 2 * norm_r3;

        // 4. H(:, 4)
        double dot_j4q1 = r[3] * H(0, 0) + r[4] * H(1, 0) + r[5] * H(2, 0),
               dot_j4q2 = r[0] * H(3, 1) + r[1] * H(4, 1) + r[2] * H(5, 1);

        H(0, 3) = r[3] - dot_j4q1 * H(0, 0);
        H(1, 3) = r[4] - dot_j4q1 * H(1, 0);
        H(2, 3) = r[5] - dot_j4q1 * H(2, 0);
        H(3, 3) = r[0] - dot_j4q2 * H(3, 1);
        H(4, 3) = r[1] - dot_j4q2 * H(4, 1);
        H(5, 3) = r[2] - dot_j4q2 * H(5, 1);
        double inv_norm_j4 = 1.0 / sqrt(H(0, 3) * H(0, 3) + H(1, 3) * H(1, 3) + H(2, 3) * H(2, 3) +
                                        H(3, 3) * H(3, 3) + H(4, 3) * H(4, 3) + H(5, 3) * H(5, 3));

        H(0, 3) *= inv_norm_j4;
        H(1, 3) *= inv_norm_j4;
        H(2, 3) *= inv_norm_j4;
        H(3, 3) *= inv_norm_j4;
        H(4, 3) *= inv_norm_j4;
        H(5, 3) *= inv_norm_j4;

        K(3, 0) = r[3] * H(0, 0) + r[4] * H(1, 0) + r[5] * H(2, 0);
        K(3, 1) = r[0] * H(3, 1) + r[1] * H(4, 1) + r[2] * H(5, 1);
        K(3, 2) = 0;
        K(3, 3) = r[3] * H(0, 3) + r[4] * H(1, 3) + r[5] * H(2, 3) + r[0] * H(3, 3) + r[1] * H(4, 3) + r[2] * H(5, 3);

        // 5. H(:, 5)
        double dot_j5q2 = r[6] * H(3, 1) + r[7] * H(4, 1) + r[8] * H(5, 1),
               dot_j5q3 = r[3] * H(6, 2) + r[4] * H(7, 2) + r[5] * H(8, 2),
               dot_j5q4 = r[6] * H(3, 3) + r[7] * H(4, 3) + r[8] * H(5, 3);

        H(0, 4) = -dot_j5q4 * H(0, 3);
        H(1, 4) = -dot_j5q4 * H(1, 3);
        H(2, 4) = -dot_j5q4 * H(2, 3);
        H(3, 4) = r[6] - dot_j5q2 * H(3, 1) - dot_j5q4 * H(3, 3);
        H(4, 4) = r[7] - dot_j5q2 * H(4, 1) - dot_j5q4 * H(4, 3);
        H(5, 4) = r[8] - dot_j5q2 * H(5, 1) - dot_j5q4 * H(5, 3);
        H(6, 4) = r[3] - dot_j5q3 * H(6, 2);
        H(7, 4) = r[4] - dot_j5q3 * H(7, 2);
        H(8, 4) = r[5] - dot_j5q3 * H(8, 2);

        H.col(4).normalize();

        K(4, 0) = 0;
        K(4, 1) = r[6] * H(3, 1) + r[7] * H(4, 1) + r[8] * H(5, 1);
        K(4, 2) = r[3] * H(6, 2) + r[4] * H(7, 2) + r[5] * H(8, 2);
        K(4, 3) = r[6] * H(3, 3) + r[7] * H(4, 3) + r[8] * H(5, 3);
        K(4, 4) = r[6] * H(3, 4) + r[7] * H(4, 4) + r[8] * H(5, 4) + r[3] * H(6, 4) + r[4] * H(7, 4) + r[5] * H(8, 4);

        // 4. H(:, 6)
        double dot_j6q1 = r[6] * H(0, 0) + r[7] * H(1, 0) + r[8] * H(2, 0),
               dot_j6q3 = r[0] * H(6, 2) + r[1] * H(7, 2) + r[2] * H(8, 2),
               dot_j6q4 = r[6] * H(0, 3) + r[7] * H(1, 3) + r[8] * H(2, 3),
               dot_j6q5 = r[0] * H(6, 4) + r[1] * H(7, 4) + r[2] * H(8, 4) + r[6] * H(0, 4) + r[7] * H(1, 4) + r[8] * H(2, 4);

        H(0, 5) = r[6] - dot_j6q1 * H(0, 0) - dot_j6q4 * H(0, 3) - dot_j6q5 * H(0, 4);
        H(1, 5) = r[7] - dot_j6q1 * H(1, 0) - dot_j6q4 * H(1, 3) - dot_j6q5 * H(1, 4);
        H(2, 5) = r[8] - dot_j6q1 * H(2, 0) - dot_j6q4 * H(2, 3) - dot_j6q5 * H(2, 4);

        H(3, 5) = -dot_j6q5 * H(3, 4) - dot_j6q4 * H(3, 3);
        H(4, 5) = -dot_j6q5 * H(4, 4) - dot_j6q4 * H(4, 3);
        H(5, 5) = -dot_j6q5 * H(5, 4) - dot_j6q4 * H(5, 3);

        H(6, 5) = r[0] - dot_j6q3 * H(6, 2) - dot_j6q5 * H(6, 4);
        H(7, 5) = r[1] - dot_j6q3 * H(7, 2) - dot_j6q5 * H(7, 4);
        H(8, 5) = r[2] - dot_j6q3 * H(8, 2) - dot_j6q5 * H(8, 4);

        H.col(5).normalize();

        K(5, 0) = r[6] * H(0, 0) + r[7] * H(1, 0) + r[8] * H(2, 0);
        K(5, 1) = 0;
        K(5, 2) = r[0] * H(6, 2) + r[1] * H(7, 2) + r[2] * H(8, 2);
        K(5, 3) = r[6] * H(0, 3) + r[7] * H(1, 3) + r[8] * H(2, 3);
        K(5, 4) = r[6] * H(0, 4) + r[7] * H(1, 4) + r[8] * H(2, 4) + r[0] * H(6, 4) + r[1] * H(7, 4) + r[2] * H(8, 4);
        K(5, 5) = r[6] * H(0, 5) + r[7] * H(1, 5) + r[8] * H(2, 5) + r[0] * H(6, 5) + r[1] * H(7, 5) + r[2] * H(8, 5);

        // Great! Now H is an orthogonalized, sparse basis of the Jacobian row space and K is filled.
        //
        // Now get a projector onto the null space of H:
        const Eigen::Matrix<double, 9, 9> Pn = Eigen::Matrix<double, 9, 9>::Identity() - (H * H.transpose());

        // Now we need to pick 3 columns of P with non-zero norm (> 0.3) and some angle between them (> 0.3).
        //
        // Find the 3 columns of Pn with largest norms
        int index1 = -1,
            index2 = -1,
            index3 = -1;
        double max_norm1 = std::numeric_limits<double>::min(),
               min_dot12 = std::numeric_limits<double>::max(),
               min_dot1323 = std::numeric_limits<double>::max();

        double col_norms[9];
        for (int i = 0; i < 9; i++)
        {
            col_norms[i] = Pn.col(i).norm();
            if (col_norms[i] >= norm_threshold)
            {
                if (max_norm1 < col_norms[i])
                {
                    max_norm1 = col_norms[i];
                    index1 = i;
                }
            }
        }
        const auto &v1 = Pn.col(index1);
        N.col(0) = v1 * (1.0 / max_norm1);
        col_norms[index1] = -1.0; // mark to avoid use in subsequent loops

        for (int i = 0; i < 9; i++)
        {
            // if (i == index1) continue;
            if (col_norms[i] >= norm_threshold)
            {
                double cos_v1_x_col = fabs(Pn.col(i).dot(v1) / col_norms[i]);

                if (cos_v1_x_col <= min_dot12)
                {
                    index2 = i;
                    min_dot12 = cos_v1_x_col;
                }
            }
        }
        const auto &v2 = Pn.col(index2);
        N.col(1) = v2 - v2.dot(N.col(0)) * N.col(0);
        N.col(1).normalize();
        col_norms[index2] = -1.0; // mark to avoid use in loop below

        for (int i = 0; i < 9; i++)
        {
            // if (i == index2 || i == index1) continue;
            if (col_norms[i] >= norm_threshold)
            {
                double inv_norm = 1.0 / col_norms[i];
                double cos_v1_x_col = fabs(Pn.col(i).dot(v1) * inv_norm);
                double cos_v2_x_col = fabs(Pn.col(i).dot(v2) * inv_norm);

                if (cos_v1_x_col + cos_v2_x_col <= min_dot1323)
                {
                    index3 = i;
                    min_dot1323 = cos_v1_x_col + cos_v2_x_col;
                }
            }
        }

        // Now orthogonalize the remaining 2 vectors v2, v3 into N
        const auto &v3 = Pn.col(index3);
        N.col(2) = v3 - (v3.dot(N.col(1)) * N.col(1)) - (v3.dot(N.col(0)) * N.col(0));
        N.col(2).normalize();
    }

    int AxbSolveLDLt3x3(                      //
        const Eigen::Matrix<double, 3, 3> &A, //
        const Eigen::Matrix<double, 3, 1> &b, //
        Eigen::Matrix<double, 3, 1> &x)
    {
        double L[3 * 3], v[2];

        // D is stored in L's diagonal, i.e. L[0], L[4], L[8]
        // its elements should be positive
        v[0] = L[0] = A(0, 0);
        if (v[0] < 1E-10)
            return 1;
        v[1] = 1.0 / v[0];
        L[3] = A(1, 0) * v[1];
        L[6] = A(2, 0) * v[1];
        // L[1] = L[2] = 0.0;

        v[0] = L[3] * L[0];
        v[1] = L[4] = A(1, 1) - L[3] * v[0];
        if (v[1] < 1E-10)
            return 2;
        L[7] = (A(2, 1) - L[6] * v[0]) / v[1];
        // L[5] = 0.0;

        v[0] = L[6] * L[0];
        v[1] = L[7] * L[4];
        L[8] = A(2, 2) - L[6] * v[0] - L[7] * v[1];
        if (L[8] < 1E-10)
            return 3;

        // Forward solve L*x = b
        x[0] = b[0];
        x[1] = b[1] - L[3] * x[0];
        x[2] = b[2] - L[6] * x[0] - L[7] * x[1];

        // Backward solve D*L'*x = y
        x[2] = x[2] / L[8];
        x[1] = x[1] / L[4] - L[7] * x[2];
        x[0] = x[0] / L[0] - L[3] * x[1] - L[6] * x[2];

        return 0;
    }

    void SolveSQPSystem(const Eigen::Matrix<double, 9, 9> &Omega, const Eigen::Matrix<double, 9, 1> &r, Eigen::Matrix<double, 9, 1> &delta)
    {
        const double sqnorm_r1 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2],
                     sqnorm_r2 = r[3] * r[3] + r[4] * r[4] + r[5] * r[5],
                     sqnorm_r3 = r[6] * r[6] + r[7] * r[7] + r[8] * r[8];
        const double dot_r1r2 = r[0] * r[3] + r[1] * r[4] + r[2] * r[5],
                     dot_r1r3 = r[0] * r[6] + r[1] * r[7] + r[2] * r[8],
                     dot_r2r3 = r[3] * r[6] + r[4] * r[7] + r[5] * r[8];

        // Obtain 6D normal (H) and 3D null space of the constraint Jacobian-J at the estimate (r)
        // NOTE: This is done via Gram-Schmidt orthogonalization
        Eigen::Matrix<double, 9, 3> N;  // Null space of J
        Eigen::Matrix<double, 9, 6> H;  // Row space of J
        Eigen::Matrix<double, 6, 6> JH; // The lower triangular matrix J*Q

        RowAndNullSpace(Omega, r, H, N, JH);

        // Great, now if delta = H*x + N*y, we first compute x by solving:
        //
        //              (J*H)*x = g
        //
        // where g is the constraint vector g = [   1 - norm(r1)^2;
        // 					     	   1 - norm(r2)^2;
        //					     	   1 - norm(r3)^2;
        //					           -r1'*r2;
        //						   -r2'*r3;
        //						   -r1'*r3 ];
        Eigen::Matrix<double, 6, 1> g;
        g[0] = 1 - sqnorm_r1;
        g[1] = 1 - sqnorm_r2;
        g[2] = 1 - sqnorm_r3;
        g[3] = -dot_r1r2;
        g[4] = -dot_r2r3;
        g[5] = -dot_r1r3;

        Eigen::Matrix<double, 6, 1> x;
        x[0] = g[0] / JH(0, 0);
        x[1] = g[1] / JH(1, 1);
        x[2] = g[2] / JH(2, 2);
        x[3] = (g[3] - JH(3, 0) * x[0] - JH(3, 1) * x[1]) / JH(3, 3);
        x[4] = (g[4] - JH(4, 1) * x[1] - JH(4, 2) * x[2] - JH(4, 3) * x[3]) / JH(4, 4);
        x[5] = (g[5] - JH(5, 0) * x[0] - JH(5, 2) * x[2] - JH(5, 3) * x[3] - JH(5, 4) * x[4]) / JH(5, 5);

        // Now obtain the component of delta in the row space of E as delta_h = H*x and assign straight into delta
        delta = H * x;

        // Then, solve for y from W*y = ksi, where matrix W and vector ksi are :
        //
        // W = N'*Omega*N and ksi = -N'*Omega*( r + delta_h );
        Eigen::Matrix<double, 3, 9> NtOmega = N.transpose() * Omega;
        Eigen::Matrix<double, 3, 3> W = NtOmega * N;
        Eigen::Matrix<double, 3, 1> y, rhs = -(NtOmega * (delta + r));

        // solve with LDLt and if it fails, use inverse
        if (AxbSolveLDLt3x3(W, rhs, y))
        {
            Eigen::Matrix<double, 3, 3> Winv;
            InvertSymmetric3x3(W, Winv);
            y = Winv * rhs;
        }

        // Finally, accumulate delta with component in tangent space (delta_n)
        delta += N * y;
    }

    SQPSolution RunSQP(const Eigen::Matrix<double, 9, 9> &Omega, const Eigen::Matrix<double, 9, 1> &r0, const SQPConfig &config)
    {
        Eigen::Matrix<double, 9, 1> r = r0;

        double delta_squared_norm = std::numeric_limits<double>::max();
        Eigen::Matrix<double, 9, 1> delta;
        int step = 0;

        while (delta_squared_norm > config.sqp_squared_tolerance && step++ < config.sqp_max_iteration)
        {
            SolveSQPSystem(Omega, r, delta);
            r += delta;
            delta_squared_norm = delta.squaredNorm();
        }

        SQPSolution solution;
        solution.num_iterations = step;
        solution.r = r;
        // clear the estimate and/or flip the matrix sign if necessary
        double det_r = Determinant9x1(solution.r);
        if (det_r < 0)
        {
            solution.r = -r;
            det_r = -det_r;
        }
        if (det_r > config.sqp_det_threshold)
        {
            config.NearestRotationMatrix(solution.r, solution.r_hat);
        }
        else
        {
            solution.r_hat = solution.r;
        }

        return solution;
    }

} // namespace sqp_engine
