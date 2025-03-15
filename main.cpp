#include "pr.hpp"
#include <iostream>
#include <memory>
#include "sen.h"

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

enum
{
    DEMO_INVERSE_KINEMATICS = 0,
    DEMO_PCA,
};

int main() {
    using namespace pr;

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

    int demo = DEMO_PCA;

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        if (demo == DEMO_INVERSE_KINEMATICS)
        {
            static std::vector<glm::vec2> points;
            if (points.empty())
            {
                PCG rng;

                glm::vec2 p = {};
                for (int i = 0; i < 10; i++)
                {
                    points.push_back(p);
                    p.x += 0.3f + glm::mix(-0.2f, 0.2f, rng.uniformf());
                    p.y += 0.3f + glm::mix(-0.2f, 0.2f, rng.uniformf());
                }
            }

            PCG colorRng;
            for (int i = 0; i < points.size() - 1; i++)
            {
                glm::vec2 from = points[i];
                glm::vec2 to   = points[i + 1];

                DrawArrow({ from.x, from.y, 0 }, { to.x, to.y, 0 }, 0.02f, 
                    { 255 * colorRng.uniformf(),255 * colorRng.uniformf(),255 * colorRng.uniformf() });
            }

            static glm::vec3 ik_terminator = { points[points.size()-1].x, points[points.size() - 1].y, 0};
            ManipulatePosition(camera, &ik_terminator, 0.4f);
            ik_terminator.z = 0;

            const int N_Joint = points.size() - 1;

            glm::vec2 begining = points[points.size() - 1];
            int N = 1;
            for (int itr = 0; itr < N; itr++)
            {
                glm::vec2 tip = points[points.size() - 1];
                glm::vec2 goal = glm::vec2(ik_terminator.x, ik_terminator.y);

                sen::MatDyn A;
                A.allocate(2, N_Joint);

                for (int j = 0; j < N_Joint; j++)
                {
                    glm::vec2 X = tip - points[j];

                    float theta = atan2(X.y, X.x);

                    // dx/dtheta, dy/dtheta
                    float dx_dtheta = -X.y;
                    float dy_dtheta = X.x;

                    A(0, j) = dx_dtheta;
                    A(1, j) = dy_dtheta;
                }

                glm::vec2 deltaX = goal - tip;
                sen::MatDyn b;
                b.allocate(2, 1);
                b.set(deltaX.x, deltaX.y);

                // underdetermined, |x| -> min under |Ax = b| 
                // SVD solution
                // sen::MatDyn delta_thetas = sen::pinv(A) * b;

                // QR solution
                sen::MatDyn delta_thetas = sen::solve_qr_underdetermined(A, b);

                std::vector<glm::vec2> localDirs(N_Joint);
                for (int i = 0; i < points.size() - 1; i++)
                {
                    localDirs[i] = points[i + 1] - points[i];
                }

                float p_delta_theta = 0.0f;
                for (int i = 0; i < localDirs.size(); i++)
                {
                    // devatable way to apply rot
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

                glm::vec2 new_tip = { 0, 0 };
                points[0] = new_tip;
                for (int i = 0; i < points.size() - 1; i++)
                {
                    new_tip += localDirs[i];
                    points[i + 1] = new_tip;
                }
            }
        } 
        else if( demo == DEMO_PCA)
        {
            float iTime = GetElapsedTime();
            float timeU = iTime * 0.7f;
            float timeV = -iTime * 0.3f;
            glm::vec2 u = glm::vec2(sinf(timeU), cosf(timeU)) * sin(iTime * 0.3f) * 1.5f;
            glm::vec2 v = glm::vec2(sinf(timeV), cosf(timeV)) * 2.2f;

            glm::mat2 M = { u, v };

            sen::Mat<128, 2> dataMat;

            pr::OnlineMean<float> xm;
            pr::OnlineMean<float> ym;
            PCG rng;
            for (int i = 0; i < 128; i++)
            {
                float u1 = rng.uniformf();
                float u2 = rng.uniformf();
                float d = sqrt(-1.0f * log(1.0f - u1));
                float theta = glm::pi<float>() * 2.0f * rng.uniform();
                float z1 = d * cosf(theta);
                float z2 = d * sinf(theta);

                glm::vec2 p = M * glm::vec2(z1, z2);

                DrawPoint({ p.x, p.y, 0 }, { 255,255,255 }, 2);

                dataMat(i, 0) = p.x;
                dataMat(i, 1) = p.y;
                xm.addSample(p.x);
                ym.addSample(p.y);
            }

            // centered
            for (int i = 0; i < 128; i++)
            {
                dataMat(i, 0) -= xm.mean();
                dataMat(i, 1) -= ym.mean();
            }

            sen::SVD<128, 2> svd = sen::svd_BV(dataMat);
            float scale1 = svd.singular(0) / sqrtf(128.0f);
            float scale2 = svd.singular(1) / sqrtf(128.0f);

            DrawArrow({ 0, 0, 0 }, { svd.V(0, 0) * scale1 * 2.0f, svd.V(1, 0) * scale1 * 2.0f, 0.0f }, 0.005f, { 255,0,0 });
            DrawArrow({ 0, 0, 0 }, { svd.V(0, 1) * scale2 * 2.0f, svd.V(1, 1) * scale2 * 2.0f, 0.0f }, 0.005f, { 0,255,0 });
        }

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());
        ImGui::RadioButton("Inverse Kinematrics", &demo, DEMO_INVERSE_KINEMATICS);
        ImGui::RadioButton("PCA", &demo, DEMO_PCA);

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
