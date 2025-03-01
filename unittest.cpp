#include "catch_amalgamated.hpp"
#include "sen.h"
#include "prp.hpp"
#include <set>

// for tests. MxN = glm::mat<N, M, float> on GLM
template <int rows, int cols>
glm::mat<cols, rows, float> toGLM(const sen::Mat<rows, cols>& m)
{
    glm::mat<cols, rows, float> r;
    for (int i_col = 0; i_col < m.cols(); i_col++)
    for (int i_row = 0; i_row < m.rows(); i_row++)
    {
        r[i_col][i_row] = m(i_row, i_col);
    }
    return r;
}

template <int rows, int cols>
sen::Mat<rows, cols> fromGLM(const glm::mat<cols, rows, float>& m)
{
    sen::Mat<rows, cols> r;
    for (int i_row = 0; i_row < r.rows(); i_row++)
    for (int i_col = 0; i_col < r.cols(); i_col++)
    {
        r(i_row, i_col) = m[i_col][i_row];
    }
    return r;
}

TEST_CASE("Index", "") {
    sen::Mat<3, 4> A;
    A.set(
        1,   2,   3,   4,
        11,  22,  33,  44,
        111, 222, 333, 444
    );

    REQUIRE(A(0, 1) == 2);
    REQUIRE(A(2, 3) == 444);
}

TEST_CASE("Col") {
    sen::Mat<3, 4> A;
    A.set(
        1,   2,   3,   4,
        11,  22,  33,  44,
        111, 222, 333, 444
    );

    //sen::print(A.col(2));
    //sen::print(sen::MatDyn(A).col(2));
    REQUIRE(A.col(2) == sen::MatDyn(A).col(2));
    
    A.set_col(2, sen::Mat<3, 1>().set(1, 1, 1));

    sen::Mat<3, 4> ref_c;
    ref_c.set(
        1,   2,   1, 4,
        11,  22,  1, 44,
        111, 222, 1, 444
    );

    REQUIRE(A == ref_c);

    //sen::print(A.row(2));
    //sen::print(sen::MatDyn(A).row(2));
    REQUIRE(A.row(2) == sen::MatDyn(A).row(2));

    A.set_row(1, sen::Mat<1, 4>().set(1,1,1,1));

    sen::Mat<3, 4> ref_r;
    ref_r.set(
        1,   2,   1, 4,
        1,   1,   1, 1,
        111, 222, 1, 444);

    REQUIRE(A == ref_r);
}

TEST_CASE("Copy", "") {
    sen::Mat<2, 3> A;
    A.set(
        1, 2, 3,
        11, 22, 33);

    sen::MatDyn B;
    sen::Mat<2, 3> C;

    B = A;
    C = B;

    for (float v : A - C) {
        REQUIRE(v == 0.0f);
    }
}

TEST_CASE("Sub") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<4, 5> A;
        for (float& v : A) { v = rng.uniformf(); }
        
        sen::MatDyn B = A;

        for (float v : B - A) {
            REQUIRE(v == 0.0f);
        }
        for (float v : A - B) {
            REQUIRE(v == 0.0f);
        }
    }
}

TEST_CASE("Mul") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<3, 2> A;
        sen::Mat<2, 3> B;

        for (float& v : A) { v = rng.uniformf(); }
        for (float& v : B) { v = rng.uniformf(); }
        //sen::print(A);
        //sen::print(B);

        sen::Mat<3, 3> AxB = A * B;
        //sen::print(AxB);

        glm::mat<3, 2, float>  ag = toGLM(A);
        glm::mat<3, 3, float> AxB_ref = toGLM(A) * toGLM(B);

        for (float v : AxB - fromGLM(AxB_ref)) {
            SEN_ASSERT(fabs(v) < 1.0e-8f);
        }
    }
}
TEST_CASE("Mul Dynamic") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::MatDyn A;
        sen::MatDyn B;

        A.allocate(3, 4);
        B.allocate(4, 3);

        for (float& v : A) { v = rng.uniformf(); }
        for (float& v : B) { v = rng.uniformf(); }
        //sen::print(A);
        //sen::print(B);

        sen::MatDyn AxB = A * B;
        //sen::print(AxB);
        glm::mat3x3 AxB_ref = toGLM(sen::Mat<3, 4>(A)) * toGLM(sen::Mat<4, 3>(B));
        for (float v : AxB - sen::MatDyn(fromGLM(AxB_ref))) {
            REQUIRE(fabs(v) < 1.0e-8f);
        }
    }

    for (int i = 0; i < 100; i++)
    {
        sen::MatDyn A;
        sen::Mat<4, 3> B;

        A.allocate(3, 4);
        B.allocate(4, 3);

        for (float& v : A) { v = rng.uniformf(); }
        for (float& v : B) { v = rng.uniformf(); }
        //sen::print(A);
        //sen::print(B);

        sen::MatDyn AxB = A * B;
        //sen::print(AxB);
        glm::mat3x3 AxB_ref = toGLM(sen::Mat<3, 4>(A)) * toGLM(sen::Mat<4, 3>(B));
        for (float v : AxB - sen::MatDyn(fromGLM(AxB_ref))) {
            REQUIRE(fabs(v) < 1.0e-8f);
        }
    }
}

TEST_CASE("Transpose") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<3, 2> A;
        for (float& v : A) { v = rng.uniformf(); }
        // sen::print(A);
        // sen::print(sen::transpose(A));

        // M = (M^T)^T
        REQUIRE(A == sen::transpose(sen::transpose(A)));

        // M == M^T for symmetric
        sen::Mat<3, 3> sym = A * sen::transpose(A);
        REQUIRE(sym == sen::transpose(sym));
    }

    for (int i = 0; i < 100; i++)
    {
        sen::MatDyn A;
        A.allocate(1 + rng.uniform() % 100, 1 + rng.uniform() % 100);
        for (float& v : A) { v = rng.uniformf(); }
        // sen::print(A);
        // sen::print(sen::transpose(A));

        // M = (M^T)^T
        REQUIRE(A == sen::transpose(sen::transpose(A)));

        // M == M^T for symmetric
        sen::MatDyn sym = A * sen::transpose(A);
        REQUIRE(sym == sen::transpose(sym));
    }
}

TEST_CASE("scaler", "") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<2, 3> A;
        float s = rng.uniformf();
        for (float& v : A) { v = rng.uniformf(); }

        //print(A);
        //print(A * 10.0f);
        REQUIRE(A * s == s * A);
    }

    for (int i = 0; i < 100; i++)
    {
        sen::MatDyn A;
        A.allocate(1 + rng.uniform() % 100, 1 + rng.uniform() % 100);
        float s = rng.uniformf();
        for (float& v : A) { v = rng.uniformf(); }
        REQUIRE(A * s == s * A);
    }

    for (int i = 0; i < 100; i++)
    {
        sen::Mat<2, 3> A;
        float s = 1 + rng.uniform() % 100;
        for (float& v : A) { v = rng.uniform() % 128; }

        REQUIRE(A * s / s == A);
    }
}

TEST_CASE("identity", "") {

    {
        sen::Mat<10, 10> I;
        I.set_identity();
        REQUIRE(I * I == I);
    }

    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        int dim = 1 + rng.uniform() % 100;
        sen::MatDyn I;
        I.allocate(dim, dim);
        I.set_identity();
        REQUIRE(I * I == I);
    }
}

TEST_CASE("badinverse", "") {
    pr::PCG rng;

    {
        // | Ax - b | -> min answer
        sen::Mat<2, 2> A;
        A.set(
            2,4,
            1,2);
        sen::Mat<2, 1> b;
        b.set(
            2, 
            3);
        sen::SVD<2, 2> svd = sen::svd_BV(A);
        auto pInv = svd.pinv();
        auto best_x = pInv * b;
        sen::Mat<2, 1> best_b = A * best_x;
        sen::Mat<1, 1> min_cost = sen::transpose(best_b - b) * (best_b - b);

        for (int i = 0; i < 64; i++)
        {
            sen::Mat<2, 1> x = best_x;
            x(0, 0) += glm::mix(-0.05f, 0.05f, rng.uniformf());
            x(1, 0) += glm::mix(-0.05f, 0.05f, rng.uniformf());

            sen::Mat<2, 1> not_best_b = A * x;
            sen::Mat<1, 1> cost = sen::transpose(not_best_b - b) * (not_best_b - b);

            // printf("%f\n", cost(0, 0));
            REQUIRE(min_cost(0, 0) <= cost(0, 0));
        }
    }

    //{
    //    // | x | -> min answer
    //    // x1 + 2 * x2 = 1

    //    sen::Mat<2, 2> A = sen::mat_of<2, 2>
    //        (2)(4)
    //        (1)(2);
    //    sen::Mat<2, 1> b = sen::mat_of<2, 1>
    //        (2)
    //        (1);
    //    sen::SVD<2, 2> svd = sen::svd_BV(A);
    //    auto pInv = svd.pinv();
    //    auto best_x = pInv * b;
    //    sen::Mat<2, 1> best_b = A * best_x;
    //    sen::Mat<1, 1> min_cost = sen::transpose(best_x) * (best_x);
    //    sen::Mat<1, 1> sq = sen::transpose(best_b - b) * (best_b - b);

    //    REQUIRE(fabs(sq(0, 0)) <= 0.00001f );

    //    for (int i = 0; i < 64; i++)
    //    {
    //        // x1 = 1 - 2 * x2
    //        float x2 = glm::mix(-2.0f, 2.0f, rng.uniformf());
    //        float x1 = 1.0f - 2.0f * x2;
    //        sen::Mat<2, 1> x = sen::mat_of<2, 1>
    //            (x1)
    //            (x2);

    //        best_b = A * x;
    //        sen::Mat<1, 1> sq = sen::transpose(best_b - b) * (best_b - b);
    //        REQUIRE(fabs(sq(0, 0)) <= 0.00001f);

    //        sen::Mat<1, 1> cost = sen::transpose(best_x) * (best_x);

    //        printf("%f\n", cost(0, 0));
    //        REQUIRE(min_cost(0, 0) <= cost(0, 0));
    //    }
    //}
}

TEST_CASE("4x4 inverse", "") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<4, 4> A;
        for (auto& v : A) { v = rng.uniformf(); }

        sen::SVD<4, 4> svd = sen::svd_BV(A);
        sen::Mat<4, 4> invA = svd.pinv();
        
        //print(A * invA);
        //print(invA * A);

        sen::Mat<4, 4> I;
        I.set_identity();

        for (auto v : I - A * invA) {
            REQUIRE(fabs(v) < 1.0e-4f);
        }

        for (auto v : I - invA * A) {
            REQUIRE(fabs(v) < 1.0e-4f);
        }
    }
}
//
TEST_CASE("pseudo inverse(overdetermined)", "") {
    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        //sen::Mat<5, 2> A;
        //for (float& v : A) { v = rng.uniformf(); }

        //sen::Mat<5, 1> b;
        //for (float& v : b) { v = rng.uniformf(); }

        //sen::Mat<2, 5> pinvA = sen::inverse(sen::transpose(A) * A) * sen::transpose(A);

        //sen::Mat<2, 1> best_x = pinvA * b;
        //sen::Mat<5, 1> best_b = A * best_x;
        //sen::Mat<1, 1> min_cost = sen::transpose(best_b - b) * (best_b - b);

        //// any offsetted x can't be better than best_x
        //for (int i = 0; i < 64; i++)
        //{
        //    sen::Mat<2, 1> x = best_x;
        //    x(0, 0) += glm::mix(-0.05f, 0.05f, rng.uniformf());
        //    x(1, 0) += glm::mix(-0.05f, 0.05f, rng.uniformf());

        //    sen::Mat<5, 1> not_best_b = A * x;
        //    sen::Mat<1, 1> cost = sen::transpose(not_best_b - b) * (not_best_b - b);

        //    REQUIRE(min_cost(0, 0) <= cost(0, 0));
        //}

        //sen::SVD<5, 2> svd = sen::svd_unordered(A);

        //for (auto& s : svd.sigma)
        //{
        //    if (s != 0.0f)
        //        s = 1.0f / s;
        //}
        //sen::Mat<2, 5> pinvA_svd = sen::transpose(svd.V_transposed) * svd.sigma * sen::transpose(svd.U);
        
        //sen::print(pinvA);
        //sen::print(pinvA_svd);
    }
}

TEST_CASE("cyclic by row", "") 
{
    for (int i = 2; i < 10; i++)
    {
        int N = i;

        std::set<std::pair<int, int>> xs;

        CYCLIC_BY_ROW(N, a, b)
        {
            REQUIRE(a < b);
            xs.insert({ a, b });
        }

        REQUIRE(xs.size() == N * (N - 1) / 2);
    }
}

TEST_CASE("SVD", "") {

    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<4, 4> A;
        for (auto& v : A) { v = glm::mix(-5.0f, 5.0f, rng.uniformf()); }

        sen::SVD<4, 4> svd = sen::svd_BV(A);
        sen::Mat<4, 4> A_composed = svd.B * sen::transpose(svd.V);

        //sen::print(A);
        //sen::print(A_composed);

        for (auto v : A - A_composed) {
            REQUIRE(fabs(v) < 1.0e-5f);
        }
    }

    for (int i = 0; i < 100; i++)
    {
        int rows = 2 + rng.uniform() % 10;
        int cols = 2 + rng.uniform() % 10;
        sen::MatDyn A;
        A.allocate(rows, cols);
        for (auto& v : A) { v = glm::mix(-1.0f, 1.0f, rng.uniformf()); }

        sen::SVDDyn svd = sen::svd_BV(A);
        sen::MatDyn A_composed = svd.B * sen::transpose(svd.V);

        //sen::print(A);
        //sen::print(A_composed);

        for (auto v : A - A_composed) {
            REQUIRE(fabs(v) < 1.0e-5f);
        }
    }
}