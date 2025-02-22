﻿#include "pr.hpp"
#include <iostream>
#include <memory>

#include <intrin.h>
#define SEN_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }
//#define SEN_ASSERT(ExpectTrue) 

namespace sen
{
    template<bool B, class T, class F>
    struct cond { using type = T; };

    template<class T, class F>
    struct cond<false, T, F> { using type = F; };

    // Example of Mat<3, 2>
    // o, o
    // o, o
    // o, o

    template <int numberOfRows, int numberOfCols>
    struct Storage
    {
        void allocate(int numberOfRows, int numberOfCols) {}

        int rows() const { return numberOfRows; }
        int cols() const { return numberOfCols; }
        int size() const { return numberOfRows * numberOfCols; }

        float& operator[](int i)
        {
            SEN_ASSERT(0 <= i && i < size() && "out of bounds");
            return m_storage[i];
        }
        const float& operator[](int i) const
        {
            SEN_ASSERT(0 <= i && i < size() && "out of bounds");
            return m_storage[i];
        }

        float m_storage[numberOfCols * numberOfRows];
    };

    template <>
    struct Storage<-1, -1>
    {
        Storage() {}
        Storage(const Storage<-1, -1>& rhs)
        {
            allocate(rhs.rows(), rhs.cols());
            for (int i = 0; i < rhs.size(); i++)
            {
                m_storage[i] = rhs[i];
            }
        }
        void operator=(const Storage<-1, -1>& rhs)
        {
            allocate(rhs.rows(), rhs.cols());
            for (int i = 0; i < rhs.size(); i++)
            {
                m_storage[i] = rhs[i];
            }
        }
        void allocate(int M, int N)
        {
            m_numberOfRows = M;
            m_numberOfCols = N;
            delete[] m_storage;
            m_storage = new float[M * N];
        }

        int rows() const { return m_numberOfRows; }
        int cols() const { return m_numberOfCols; }
        int size() const { return m_numberOfRows * m_numberOfCols; }

        float& operator[](int i)
        {
            SEN_ASSERT(0 <= i && i < size() && "out of bounds");
            return m_storage[i];
        }
        const float& operator[](int i) const
        {
            SEN_ASSERT(0 <= i && i < size() && "out of bounds");
            return m_storage[i];
        }

        int m_numberOfRows = 0;
        int m_numberOfCols = 0;
        float* m_storage = 0;
    };

    template <int numberOfRows, int numberOfCols>
    struct Mat
    {
        Mat() {}

        template <int M, int N>
        Mat(const Mat<M, N>& rhs /* static or dynamic */)
        {
            m_storage.allocate(M, N);

            SEN_ASSERT(rows() == rhs.rows() && "dim mismatch");
            SEN_ASSERT(cols() == rhs.cols() && "dim mismatch");

            for (int i = 0; i < rhs.size(); i++)
            {
                m_storage[i] = rhs[i];
            }
        }

        void allocate(int M, int N)
        {
            m_storage.allocate(M, N);
        }

        int rows() const { return m_storage.rows(); }
        int cols() const { return m_storage.cols(); }
        int size() const { return m_storage.size(); }

        float& operator()(int i_col, int i_row)
        {
            SEN_ASSERT(0 <= i_col && i_col < cols() && "out of bounds");
            SEN_ASSERT(0 <= i_row && i_row < rows() && "out of bounds");
            return m_storage[i_col * m_storage.rows() + i_row];
        }
        const float& operator()(int i_col, int i_row) const
        {
            SEN_ASSERT(0 <= i_col && i_col < cols() && "out of bounds");
            SEN_ASSERT(0 <= i_row && i_row < rows() && "out of bounds");
            return m_storage[i_col * m_storage.rows() + i_row];
        }

        float& operator[](int i)
        {
            return m_storage[i];
        }
        const float& operator[](int i) const
        {
            return m_storage[i];
        }

        using ColType = typename cond<numberOfRows == -1 && numberOfCols == -1, Mat<-1, -1>, Mat<numberOfRows, 1>>::type;

        ColType col(int i_col) const
        {
            SEN_ASSERT(0 <= i_col && i_col < cols() && "out of bounds");

            ColType c;
            c.allocate(rows(), 1);
            for (int i = 0; i < rows(); i++)
            {
                c(0, i) = (*this)(i_col, i);
            }
            return c;
        }

        template<int M, int N>
        void set_col(int i_col, const Mat<M, N>& c /* static or dynamic */)
        {
            SEN_ASSERT(0 <= i_col && i_col < cols() && "out of bounds");
            SEN_ASSERT(rows() == c.rows());
            for (int i = 0; i < rows(); i++)
            {
                (*this)(i_col, i) = c(0, i);
            }
        }
        void set_col(int i_col, const Mat<numberOfRows, 1>& c)
        {
            set_col<numberOfRows, 1>(i_col, c);
        }

        //Mat<1, numberOfCols> row(int i_row) const {
        //    SEN_ASSERT(0 <= i_row && i_row < rows() && "");

        //    Mat<1, numberOfCols> r;
        //    for (int i = 0; i < cols(); i++)
        //    {
        //        r(i, 0) = (*this)(i, i_row);
        //    }
        //    return r;
        //}

        float* begin() { return &m_storage[0]; }
        float* end() { return &m_storage[0] + m_storage.size(); }
        const float* begin() const { return m_storage[0]; }
        const float* end() const { return &m_storage[0] + m_storage.size(); }

        Storage<numberOfRows, numberOfCols> m_storage;
    };

    template <int numberOfRows, int numberOfCols>
    struct RowMajorInitializer {
        float xs[numberOfRows * numberOfCols];
        int index;
        RowMajorInitializer() :index(0)
        {
        }
        RowMajorInitializer& operator()(float value) {
            SEN_ASSERT(index < numberOfRows * numberOfCols && "out of bounds");
            xs[index++] = value;
            return *this;
        }
        operator Mat<numberOfRows, numberOfCols>() const {
            SEN_ASSERT(index == numberOfRows * numberOfCols && "initialize failure");

            Mat<numberOfRows, numberOfCols> m;
            for (int i_col = 0; i_col < m.cols(); i_col++)
            for (int i_row = 0; i_row < m.rows(); i_row++)
            {
                m(i_col, i_row) = xs[i_row * numberOfCols + i_col];
            }
            return m;
        }

        operator Mat<-1, -1>() const {
            SEN_ASSERT(index == numberOfRows * numberOfCols && "initialize failure");

            Mat<-1, -1> m;
            m.allocate(numberOfRows, numberOfCols);
            for (int i_col = 0; i_col < m.cols(); i_col++)
            for (int i_row = 0; i_row < m.rows(); i_row++)
            {
                m(i_col, i_row) = xs[i_row * numberOfCols + i_col];
            }
            return m;
        }
    };

    template <int numberOfRows, int numberOfCols>
    RowMajorInitializer<numberOfRows, numberOfCols> mat_of( float v )
    {
        static_assert(0 <= numberOfRows, "");
        static_assert(0 <= numberOfCols, "");
        RowMajorInitializer<numberOfRows, numberOfCols> initializer;
        return initializer(v);
    }

    using MatDyn = Mat<-1, -1>;

    template <int rows, int cols>
    void print(const Mat<rows, cols>& m)
    {
        printf("Mat<%d,%d> %dx%d {\n", rows, cols, m.rows(), m.cols());
        for (int i_row = 0; i_row < m.rows(); i_row++)
        {
            printf("  row[%d]=", i_row);
            for (int i_col = 0; i_col < m.cols(); i_col++)
            {
                printf("%.8f, ", m(i_col, i_row) );
            }
            printf("\n");
        }
        printf("}\n");
    }

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    Mat<lhs_rows, rhs_cols> operator*(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        Mat<lhs_rows, rhs_cols> r;
        static_assert(lhs_cols == rhs_rows, "invalid multiplication");
        SEN_ASSERT(lhs.cols() == rhs.rows() && "invalid multiplication");

        r.allocate(lhs.rows(), rhs.cols());
        
        for (int dst_row = 0; dst_row < r.rows(); dst_row++)
        for (int dst_col = 0; dst_col < r.cols(); dst_col++)
        {
            float value = 0.0f;
            for (int i = 0 ; i < lhs.cols(); i++)
            {
                value += lhs(i, dst_row) * rhs(dst_col, i);
            }
            r(dst_col, dst_row) = value;
        }

        return r;
    }

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    Mat<lhs_rows, rhs_cols> operator-(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        static_assert(lhs_rows == rhs_rows, "invalid substruct");
        static_assert(lhs_cols == rhs_cols, "invalid substruct");
        Mat<lhs_rows, lhs_cols> r;

        r.allocate(lhs.rows(), lhs.cols());

        for (int dst_row = 0; dst_row < r.rows(); dst_row++)
        for (int dst_col = 0; dst_col < r.cols(); dst_col++)
        {
            r(dst_col, dst_row) = lhs(dst_col, dst_row) - rhs(dst_col, dst_row);
        }

        return r;
    }

    template <int rows, int cols>
    Mat<cols, rows> transpose(const Mat<rows, cols>& m)
    {
        Mat<cols, rows> r;

        r.allocate(m.cols(), m.rows());

        for (int i_row = 0; i_row < m.rows(); i_row++)
        {
            for (int i_col = 0; i_col < m.cols(); i_col++)
            {
                r(i_row, i_col ) = m(i_col, i_row);
            }
        }
        return r;
    }
}

// for tests
template <int rows, int cols>
glm::mat<rows, cols, float> toGLM(const sen::Mat<rows, cols>& m)
{
    glm::mat<rows, cols, float> r;
    for (int i_row = 0; i_row < rows; i_row++)
    {
        for (int i_col = 0; i_col < cols; i_col++)
        {
            r[i_col][i_row] = m(i_col, i_row);
        }
    }
    return r;
}

template <int rows, int cols>
sen::Mat<rows, cols> fromGLM(const glm::mat<rows, cols, float>& m)
{
    sen::Mat<rows, cols> r;
    for (int i_row = 0; i_row < rows; i_row++)
    {
        for (int i_col = 0; i_col < cols; i_col++)
        {
            r(i_col, i_row) = m[i_col][i_row];
        }
    }
    return r;
}

template <class T>
inline T ss_max(T x, T y)
{
    return (x < y) ? y : x;
}

template <class T>
inline T ss_min(T x, T y)
{
    return (y < x) ? y : x;
}

void eignValues(float* lambda0, float* lambda1, float* determinant, const glm::mat2& mat)
{
    float mean = (mat[0][0] + mat[1][1]) * 0.5f;
    float det = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
    float d = std::sqrtf(ss_max(mean * mean - det, 0.0f));
    *lambda0 = mean + d;
    *lambda1 = mean - d;
    *determinant = det;
}
void eigen_vectors_of_cov(glm::vec2* eigen0, glm::vec2* eigen1, const glm::mat2& cov, float lambda0, float lambda1)
{
    float s11 = cov[0][0];
    float s22 = cov[1][1];
    float s12 = cov[1][0];
    float s21 = cov[0][1];

    glm::vec2 e0 = {
        s12,
        -(s11 - lambda0)
    };
    glm::vec2 e1 = {
        s12,
        -(s11 - lambda1)
    };
    ////float eps = 1e-15f;
    ////glm::vec2 e0 = glm::normalize(s11 < s22 ? glm::vec2(s12 + eps, lambda0 - s11) : glm::vec2(lambda0 - s22, s12 + eps));
    ////glm::vec2 e1 = { -e0.y, e0.x };
    *eigen0 = glm::normalize(e0);
    *eigen1 = glm::normalize(e1);
}

int main() {
    using namespace pr;

    //{
    //    sen::Storage<-1, -1> d;
    //    sen::Mat<2, 3> A;
    //    sen::Mat<-1, -1> B;
    //}

    {
        sen::Mat<2, 3> A = sen::mat_of<2, 3>
            (1)(2)(3)
            (11)(22)(33);

        sen::MatDyn B;
        sen::Mat<2, 3> C;

        B = A;
        C = B;

        for (float v : A - C) {
            PR_ASSERT(v == 0.0f);
        }
    }
    {
        sen::Mat<2, 3> A = sen::mat_of<2, 3>
            (1  )(2  )(3  )
            (11 )(22 )(33 );
        sen::print(A);
        sen::print(sen::MatDyn(A));
        sen::print(sen::Mat<2, 3>(sen::MatDyn(A)));
        sen::print(A.col(2));
        sen::print(sen::MatDyn(A).col(2));

        A.set_col(0, sen::mat_of<2, 1>(0)(0));
        sen::print(A);

        sen::Mat<2, 3> DynA = sen::MatDyn(A);
        DynA.set_col(0, sen::mat_of<2, 1>(0)(0));
        sen::print(DynA);
        //sen::print(sen::MatDyn(A).row(1));

        sen::MatDyn C = sen::mat_of<2, 3>
            (1)(2)(3)
            (11)(22)(33);
        sen::print(C);

        //sen::Mat<2, 3> B = sen::MatDyn(A);
        //sen::print(B);
        ////sen::MatDyn(sen::mat_of<2, 1>
        ////    (0)
        ////    (0)
        ////);
        //sen::MatDyn DA(A);
        //
        //sen::print(A);

        //printf("");
    }
    {
        PCG rng;
        for (int i = 0; i < 100; i++)
        {
            sen::Mat<3, 3> A;
            sen::Mat<3, 3> B;

            for (float& v : A) { v = rng.uniformf(); }
            for (float& v : B) { v = rng.uniformf(); }
            //sen::print(A);
            //sen::print(B);

            sen::Mat<3, 3> AxB = A * B;
            //sen::print(AxB);

            glm::mat3x3 AxB_ref = toGLM(A) * toGLM(B);
            for (float v : AxB - fromGLM(AxB_ref)) {
                PR_ASSERT(fabs(v) < 1.0e-8f)
            }
        }
    }
    {
        PCG rng;
        for (int i = 0; i < 100; i++)
        {
            sen::Mat<3, 2> A;
            for (float& v : A) { v = rng.uniformf(); }
            // sen::print(A);
            // sen::print(sen::transpose(A));

            // M = (M^T)^T
            for (float v : A - sen::transpose(sen::transpose(A))) {
                PR_ASSERT(v == 0.0f);
            }

            // M == M^T for symmetric
            sen::Mat<3, 3> sym = A * sen::transpose(A);
            for (float v : sym - sen::transpose(sym)) {
                PR_ASSERT(v == 0.0f);
            }
        }
    }

    // [Mul] Dynamic
    {
        PCG rng;
        for (int i = 0; i < 100; i++)
        {
            sen::MatDyn A;
            sen::MatDyn B;

            A.allocate(3, 3);
            B.allocate(3, 3);

            for (float& v : A) { v = rng.uniformf(); }
            for (float& v : B) { v = rng.uniformf(); }
            //sen::print(A);
            //sen::print(B);

            sen::MatDyn AxB = A * B;
            //sen::print(AxB);
            glm::mat3x3 AxB_ref = toGLM(sen::Mat<3,3>(A)) * toGLM(sen::Mat<3, 3>(B));
            for (float v : AxB - sen::MatDyn(fromGLM(AxB_ref))) {
                PR_ASSERT(fabs(v) < 1.0e-8f);
            }
        }
    }

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 0, 0, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = false;

    double e = GetElapsedTime();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        static PCG pcg;

        glm::vec3 b1 = { 0.5, 0.6, -0.2f };
        glm::vec3 b2 = { -2.0f, 1.0f, 0.4f };
        //glm::vec3 b1 = { glm::mix(-2.0f, 2.0f,  pcg.uniformf()), glm::mix(-2.0f, 2.0f,  pcg.uniformf()) ,glm::mix(-2.0f, 2.0f,  pcg.uniformf()) };
        //glm::vec3 b2 = { glm::mix(-2.0f, 2.0f,  pcg.uniformf()), glm::mix(-2.0f, 2.0f,  pcg.uniformf()) ,glm::mix(-2.0f, 2.0f,  pcg.uniformf()) };

        //b2 = glm::vec3(b1.x, b1.z, -b1.y );

        //float X = glm::dot(b1, b2) * 2.0f;
        //float Y = glm::dot(b1, b1) - glm::dot(b2, b2);

        //X = 1;
        //Y = 1;

        static glm::vec3 P = {2, 1, 0.0f};
        ManipulatePosition(camera, &P, 0.5f);

        float X = P.x;
        float Y = P.y;

        float two_theta = atan(Y / X);
        // float two_theta = atan2(Y, X);

        //float cos2Theta = cos(two_theta);
        //float sin2Theta = sin(two_theta);
        //float originalZero =
        //    cos2Theta * Y - sin2Theta * X;
        //printf("originalZero %f\n", originalZero);

        float s = sin(two_theta * 0.5f);
        float c = cos(two_theta * 0.5f);

        DrawSphere({ c, s, 0 }, 0.06, { 255,255,255 });

        // 誤差が大きいが、うごいた
        //{
        //    float Px = glm::dot(b1, b1) - glm::dot(b2, b2);
        //    float Py = 2.0f * glm::dot(b1, b2);
        //    //float Px = P.y;
        //    //float Py = P.x;

        //    glm::vec2 H = { Px + sqrtf(Px * Px + Py * Py), Py };

        //    DrawSphere({ H.x, H.y, 0 }, 0.04, { 0,255,0 });
        //    //float ta = H.y / H.x;
        //    //float k = atan(ta) * 2.0f;
        //    //c = cos(atan(ta));
        //    //s = sin(atan(ta));
        //    H = glm::normalize(H);
        //    c = H.x;
        //    s = H.y;
        //}

        //{
        //    float Px = glm::dot(b1, b1) - glm::dot(b2, b2);
        //    float Py = 2.0f * glm::dot(b1, b2);
        //    //float Px = P.y;
        //    //float Py = P.x;

        //    glm::vec2 H = glm::vec2{ -Py, Px - sqrtf(Px * Px + Py * Py) };

        //    DrawSphere({ H.x, H.y, 0 }, 0.04, { 255,0,0 });

        //    H = glm::normalize(H);
        //    c = H.x;
        //    s = H.y;
        //}
        //{
        //    float tau = 2.0f * glm::dot(b1, b2) / (glm::dot(b1, b1) - glm::dot(b2, b2));
        //    tau = std::min(tau, FLT_MAX);
        //    float tau2 = std::min(tau * tau, FLT_MAX);
        //    glm::vec2 H = glm::vec2( 1.0f + sqrtf( 1.0f + tau2 ), tau );
        //    float L = glm::length(H);
        //    c = H.x / L;
        //    s = H.y / L;
        //}
        //{
        //    float tau = (glm::dot(b1, b1) - glm::dot(b2, b2)) * 0.5f / glm::dot(b1, b2);
        //    tau = sqrt(FLT_MAX);
        //    glm::vec2 H = glm::vec2(tau + sqrtf(1.0f + tau * tau), 1.0f);
        //    float l2 = glm::dot(H, H);
        //    float lll = tau + sqrtf(1.0f + tau * tau);

        //    float L = glm::length(H);
        //    c = H.x / L;
        //    s = H.y / L;
        //}
        //{
        //    float Py = 2.0f * glm::dot(b1, b2); 
        //    float Px = glm::dot(b1, b1) - glm::dot(b2, b2);

        //    float PL = sqrtf( Px * Px + Py * Py );
        //    glm::vec2 H = glm::vec2(Py, PL - Px);
        //    float L = glm::length(H);
        //    c = H.x / L;
        //    s = H.y / L;
        //}
        {
            //float Py = 2.0f * glm::dot(b1, b2);
            //float Px = glm::dot(b1, b1) - glm::dot(b2, b2);
            float Px = X;
            float Py = Y;

            float PL = sqrtf(Px * Px + Py * Py);
            if (PL == 0.0f)
            {
                c = 1.0f;
                s = 0.0f;
            }
            else
            {
                glm::vec2 H;
                //if (Px < 0.0f)
                //{
                //    H = glm::vec2(Py, PL - Px);
                //    //H = { H.y, -H.x };
                //    DrawLine({ 0,0,0 }, { H.x, H.y, 0 }, { 255, 0,0 });
                //}
                ////else
                //{
                //    H = glm::vec2(PL + Px, Py);
                //    //DrawLine({ 0,0,0 }, { H.x, H.y, 0 }, { 0, 255,0 });
                //}
                float sgn = 0.0f < Px ? 1.0f : -1.0f;
                H = glm::vec2(PL + Px * sgn, Py * sgn);
                DrawLine({ 0,0,0 }, { H.x, H.y, 0 }, { 0, 0,255 });

                //float L = glm::length(H);
                //float tanTheta = H.y / H.x;
                //float cosTheta = H.x / L;
                //printf(" c %f %f\n", c, cosTheta);
                //printf(" s %f %f\n", s, cosTheta * tanTheta);
                //DrawSphere({ cosTheta, cosTheta * tanTheta, 0 }, 0.04, { 255,0,0 });

                float L = glm::length(H);
                //printf(" P %f %f\n", Px, Py);
                //printf(" c %f %f\n", c, H.x / L);
                //printf(" s %f %f\n", s, H.y / L);
                c = H.x / L;
                s = H.y / L;

                //float theta = atan(H.x / H.y);
                //printf(" 2 theta %f %f\n", theta * 2, two_theta);

                DrawSphere({ c, s, 0 }, 0.04, { 255,0,0 });
            }
        }


        //DrawSphere({ X, Y, 0.0f }, 0.02f, { 255,0,0 });

        // want to make this zero
        //float zero = X * (c * c - s * s) + c * s * Y * 0.5f;
        //printf("%f\n", zero);

        //float zero = glm::dot(b1, b2) * (c * c - s * s) + c * s * (glm::dot(b2, b2) - glm::dot(b1, b1));
        float zero = 0.5f * Y * (c * c - s * s) + c * s * (-X);
        //printf("%.10f\n", zero);
#if 0
        static float m00 = 0.1f;
        static float m10 = 0.4f;
        static float m01 = -0.1f;
        static float m11 = 1.5f;
        glm::mat2 M = {
            m00, m10, m01, m11
        };

        //float det;
        //float lambda0;
        //float lambda1;
        //eignValues(&lambda0, &lambda1, &det, M);

        //glm::vec2 eigen0, eigen1;
        //eigen_vectors_of_cov(&eigen0, &eigen1, M, lambda0, lambda1);

        //DrawArrow({}, { eigen0.x, eigen0.y, 0.0f }, 0.01f, { 255,0,0 });
        //DrawArrow({}, { eigen1.x, eigen1.y, 0.0f }, 0.01f, { 0,255,0 });

        //eigen0 = M * eigen0;
        //eigen1 = M * eigen1;
        //DrawArrow({}, { eigen0.x, eigen0.y, 0.0f }, 0.01f, { 255,0,0 });
        //DrawArrow({}, { eigen1.x, eigen1.y, 0.0f }, 0.01f, { 0,255,0 });

        int N = 1000;
        CircleGenerator circle(glm::two_pi<float>() / N);
        for (int i = 0; i < N; i++)
        {
            glm::vec2 p = {
                 circle.cos(),
                 circle.sin()
            };
            glm::vec2 q = M * p;
            glm::vec3 color = plasma((float)i / N) * 255.0f;
            DrawPoint({ q.x, q.y, 0.0f }, glm::u8vec3(color), 4);

            circle.step();
        }
#endif

        
        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        //ImGui::SliderFloat("m00", &m00, -2, 2);
        //ImGui::SliderFloat("m10", &m10, -2, 2);
        //ImGui::SliderFloat("m01", &m01, -2, 2);
        //ImGui::SliderFloat("m11", &m11, -2, 2);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
