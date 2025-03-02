#include "pr.hpp"
#include <iostream>
#include <memory>
#include "sen.h"

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


    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 0, 0, 18 };
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

        static std::vector<glm::vec2> points = {
            {0, 0},
            {1.5, 1.0},
            {2, 3},
            {4, 3.5},
        };

        PCG colorRng;
        for (int i = 0; i < points.size() - 1; i++)
        {
            glm::vec2 from = points[i];
            glm::vec2 to   = points[i + 1];

            DrawArrow({ from.x, from.y, 0 }, { to.x, to.y, 0 }, 0.02f, 
                { 255 * colorRng.uniformf(),255 * colorRng.uniformf(),255 * colorRng.uniformf() });
        }

        static glm::vec3 ik_terminator = { 4, 3.5, 0 };
        ManipulatePosition(camera, &ik_terminator, 0.4f);
        ik_terminator.z = 0;

        const int N_Joint = points.size() - 1;

        glm::vec2 begining = points[points.size() - 1];
        int N = 1;
        for (int itr = 0; itr < N; itr++)
        {
            glm::vec2 tip = points[points.size() - 1];
            glm::vec2 goal =
                glm::mix(
                    begining,
                    glm::vec2(ik_terminator.x, ik_terminator.y),
                    (float)(itr + 1) / N);

#if 0
            sen::MatDyn A;
            A.allocate(2, N_Joint * 2);

            for (int j = 0; j < N_Joint; j++)
            {
                glm::vec2 X = tip - points[j];

                A(0, j * 2 + 0) = X.x; A(0, j * 2 + 1) = -X.y;
                A(1, j * 2 + 0) = X.y; A(1, j * 2 + 1) = X.x;
            }

            // sen::print(A);

            glm::vec2 deltaX = goal - tip;

            sen::MatDyn pinv = sen::svd_BV(A).pinv();
            sen::MatDyn delta_cs = pinv* sen::Mat<2, 1>().set(
                deltaX.x,
                deltaX.y);

            std::vector<glm::vec2> localDirs(N_Joint);
            for (int i = 0; i < points.size() - 1; i++)
            {
                localDirs[i] = points[i + 1] - points[i];
            }

            for (int i = 0; i < localDirs.size(); i++)
            {
                float delta_c = delta_cs(i * 2,     0);
                float delta_s = delta_cs(i * 2 + 1, 0);

                glm::mat2 R = {
                    1.0f + delta_c, delta_s,
                   -delta_s, 1.0f + delta_c
                };

                glm::vec2 src_dir = localDirs[i];
                glm::vec2 new_dir = R * src_dir;

                // preserve length
                float src_length = glm::length(src_dir);
                localDirs[i] = glm::normalize(new_dir) * src_length;
            }
#else
            sen::MatDyn A;
            A.allocate(2, N_Joint);

            for (int j = 0; j < N_Joint; j++)
            {
                glm::vec2 X = tip - points[j];

                float theta = atan2(X.y, X.x);

                A(0, j) = -sin(theta) * X.x - cos(theta) * X.y;
                A(1, j) = -sin(theta) * X.y + cos(theta) * X.x;
            }

            glm::vec2 deltaX = goal - tip;

            sen::MatDyn pinv = sen::svd_BV(A).pinv(0.1f);
            sen::MatDyn delta_thetas = pinv * sen::Mat<2, 1>().set(
                deltaX.x,
                deltaX.y);

            // sen::print(delta_theta);
            std::vector<glm::vec2> localDirs(N_Joint);
            for (int i = 0; i < points.size() - 1; i++)
            {
                localDirs[i] = points[i + 1] - points[i];
            }

            for (int i = 0; i < localDirs.size(); i++)
            {
                float delta_theta = delta_thetas(i, 0);

                float s = sin(delta_theta);
                float c = cos(delta_theta);

                glm::mat2 R = {
                    c, s,
                   -s, c
                };

                glm::vec2 src_dir = localDirs[i];
                glm::vec2 new_dir = R * src_dir;
                localDirs[i] = new_dir;
            }

            // sen::print(A);
#endif

            glm::vec2 new_tip = { 0, 0 };
            points[0] = new_tip;
            for (int i = 0; i < points.size() - 1; i++)
            {
                new_tip += localDirs[i];
                points[i + 1] = new_tip;
            }
        }

        // sen::print(delta_cs);

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
