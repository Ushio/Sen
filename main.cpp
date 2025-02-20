#include "pr.hpp"
#include <iostream>
#include <memory>

#include <intrin.h>
#define SEN_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }

namespace sen
{
    template <int numberOfRows, int numberOfCols>
    struct Mat
    {
        static_assert(0 <= numberOfRows, "");
        static_assert(0 <= numberOfCols, "");

        int rows() const { return numberOfRows; }
        int cols() const { return numberOfCols; }

        float& operator()(int i_col, int i_row) 
        {
            return m_storage[i_col][i_row];
        }
        const float& operator()(int i_col, int i_row) const
        {
            return m_storage[i_col][i_row];
        }

        Mat<numberOfRows, 1> col(int i_col) const {
            Mat<numberOfRows, 1> col;
            for (int i = 0; i < numberOfRows; i++)
            {
                col(0, i) = (*this)(i_col, i);
            }
            return col;
        }

        float* begin() { return &m_storage[0][0]; }
        float* end() { return &m_storage[0][0] + numberOfCols * numberOfRows; }
        const float* begin() const { return m_storage[0][0]; }
        const float* end() const { return &m_storage[0][0] + numberOfCols * numberOfRows; }

        float m_storage[numberOfCols][numberOfRows];
    };

    template <int rows, int cols>
    void allocate_if_needed(Mat<rows, cols>* m, int numberOfRows, int numberOfCols)
    {
    }

    // Dynamic Specialization, only for cpu
    // Do not support any binary operation of (Dynamic, Static)
    // Please use (Dynamic, Dynamic) or (Static, Static)

    template <>
    struct Mat<-1, -1>
    {
        Mat()
        {
        }
        ~Mat()
        {
            delete[] m_storage;
            m_storage = 0;
        }
        void operator=(const Mat& rhs)
        {
            allocate(rhs.rows(), rhs.cols());

            for (int i = 0; i < rhs.rows() * rhs.cols(); i++)
            {
                m_storage[i] = rhs.m_storage[i];
            }
        }
        Mat(const Mat& rhs)
        {
            allocate(rhs.rows(), rhs.cols());

            for (int i = 0; i < rhs.rows() * rhs.cols(); i++)
            {
                m_storage[i] = rhs.m_storage[i];
            }
        }

        // static to dynamic
        template <int M, int N>
        Mat(const Mat<M, N>& rhs)
        {
            allocate(rhs.rows(), rhs.cols());

            for (int i_row = 0; i_row < rhs.rows(); i_row++)
            for (int i_col = 0; i_col < rhs.cols(); i_col++)
            {
                (*this)(i_row, i_col) = rhs(i_row, i_col);
            }
        }

        void allocate(int numberOfRows, int numberOfCols)
        {
            m_numberOfRows = numberOfRows;
            m_numberOfCols = numberOfCols;
            delete[] m_storage;
            m_storage = new float[numberOfRows * numberOfCols];
        }

        // dynamic to static
        template <int M, int N>
        Mat<M, N> asMatMxN()
        {
            SEN_ASSERT(rows() == M && "dim mismatch");
            SEN_ASSERT(cols() == N && "dim mismatch");

            Mat<M, N> r;
            for (int i_row = 0; i_row < rows(); i_row++)
            for (int i_col = 0; i_col < cols(); i_col++)
            {
                r(i_row, i_col) = (*this)(i_row, i_col);
            }
            return r;
        }

        int rows() const { return m_numberOfRows; }
        int cols() const { return m_numberOfCols; }

        float& operator()(int i_col, int i_row)
        {
            return m_storage[i_col * m_numberOfRows + i_row];
        }
        const float& operator()(int i_col, int i_row) const
        {
            return m_storage[i_col * m_numberOfRows + i_row];
        }

        float* begin() { return m_storage; }
        float* end() { return m_storage + m_numberOfCols * m_numberOfRows; }
        const float* begin() const { return m_storage; }
        const float* end() const { return m_storage + m_numberOfCols * m_numberOfRows; }

        int m_numberOfRows = 0;
        int m_numberOfCols = 0;
        float* m_storage = 0;
    };

    using MatDyn = Mat<-1, -1>;

    template <>
    void allocate_if_needed<-1, -1>(MatDyn *m, int numberOfRows, int numberOfCols)
    {
        m->allocate(numberOfRows, numberOfCols);
    }
    

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

        allocate_if_needed(&r, lhs.rows(), rhs.cols());
        
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

        allocate_if_needed(&r, lhs.rows(), lhs.cols());

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

        allocate_if_needed(&r, m.cols(), m.rows());

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
            glm::mat3x3 AxB_ref = toGLM(A.asMatMxN<3, 3>()) * toGLM(B.asMatMxN<3, 3>());
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
