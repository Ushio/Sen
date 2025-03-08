#include "catch_amalgamated.hpp"

//#define SEN_ENABLE_ASSERTION
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

TEST_CASE("bad mat", "") {
    // no solution
    // x1 + x2 = 1
    // x1 + x2 = 2
    sen::Mat<2, 2> A;
    A.set(
        1, 1,
        1, 1);
    sen::Mat<2, 1> b;
    b.set(
        1,
        2);
    sen::SVD<2, 2> svd = sen::svd_BV(A);
    auto pInv = svd.pinv();
    auto best_x = pInv * b;
    // print(best_x);

    REQUIRE(best_x(0, 0) == 0.75f);
    REQUIRE(best_x(1, 0) == 0.75f);
}

TEST_CASE("4x4 inverse", "") {
    pr::PCG rng;
    for (int i = 0; i < 10000; i++)
    {
        sen::Mat<4, 4> A;
        for (auto& v : A) { v = rng.uniformf(); }

        sen::SVD<4, 4> svd = sen::svd_BV(A);
        sen::Mat<4, 4> invA = svd.pinv();
        
        bool regular = true;
        for (int j = 0; j < svd.nSingulars(); j++)
        {
            if (svd.singular(j) < 0.01f)
            {
                regular = false;
            }
        }
        if (regular == false)
        {
            continue;
        }
        sen::Mat<4, 4> A_composed = svd.B * sen::transpose(svd.V);

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

TEST_CASE("Multiple Linear Regression", "") {
    pr::PCG rng;

    // reference function(x) = k + a*x1 + b*x2 + c*x3
    float k = 12.0f;
    float a = -0.4f;
    float b = 2.0f;
    float c = 0.5f;

    sen::Mat<4, 1> x_ref;
    x_ref.set(
        k, 
        a, 
        b, 
        c);

    enum {
        N_Samples = 512
    };

    sen::Mat<N_Samples, 4> A;
    for (int i = 0; i < A.rows(); i++)
    {
        A(i, 0) = 1;

        for (int j = 1; j < 4; j++)
        {
            A(i, j) = rng.uniformf();
        }
    }

    sen::Mat<N_Samples, 1> errors;
    for (auto& v : errors) { v = glm::mix(-0.1f, 0.1f, rng.uniformf()); }

    sen::Mat<N_Samples, 1> measurements = A * x_ref + errors;

    sen::SVD<N_Samples, 4> svd = sen::svd_BV(A);
    sen::Mat<4, 1> x_resolved = svd.pinv() * measurements;

    // sen::print(x_resolved);

    REQUIRE(fabs(x_resolved(0, 0) - k) < 0.05f);
    REQUIRE(fabs(x_resolved(1, 0) - a) < 0.05f);
    REQUIRE(fabs(x_resolved(2, 0) - b) < 0.05f);
    REQUIRE(fabs(x_resolved(3, 0) - c) < 0.05f);

    REQUIRE(sen::svd_BV(sen::MatDyn(A)).pinv() == svd.pinv()); // dynamic equality check
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

    BENCHMARK("SVD static") {
        pr::PCG rng;
        float s = 0;
        for (int i = 0; i < 10000; i++)
        {
            sen::Mat<8, 8> A;
            for (auto& v : A) { v = glm::mix(-5.0f, 5.0f, rng.uniformf()); }

            sen::SVD<8, 8> svd = sen::svd_BV(A);
            sen::Mat<8, 8> A_composed = svd.B * sen::transpose(svd.V);

            s += A_composed(0, 0);
        }
        return s;
    };
    BENCHMARK("SVD dynamic") {
        pr::PCG rng;
        float s = 0;
        for (int i = 0; i < 10000; i++)
        {
            sen::MatDyn A;
            A.allocate(8, 8);
            for (auto& v : A) { v = glm::mix(-5.0f, 5.0f, rng.uniformf()); }

            sen::SVDDyn svd = sen::svd_BV(A);
            sen::MatDyn A_composed = svd.B * sen::transpose(svd.V);

            s += A_composed(0, 0);
        }
        return s;
    };
}

TEST_CASE("cholesky", "") {
    pr::PCG rng;
    for (int i = 0; i < 100000; i++)
    {
        sen::Mat<10, 5> data;
        for (auto& v : data) { v = glm::mix(-5.0f, 5.0f, rng.uniformf()); }

        sen::Mat<5, 5> A = sen::transpose(data) * data / 10.0f;
        sen::Mat<5, 5> L = sen::cholesky_decomposition(A);
        //sen::print(A);
        //sen::print(L * sen::transpose(L));

        for (auto v : A - L * sen::transpose(L)) {
            REQUIRE(fabs(v) < 1.0e-5f);
        }
    }
}

//TEST_CASE("overdetermined", "") {
//    sen::Mat<3, 2> A;
//    A.set(
//        1, 1,
//        0, 2,
//        1, 0);
//
//    sen::Mat<3, 1> b;
//    b.set(
//        1,
//        2,
//        1
//    );
//
//    // Normal Equation
//    // A^T * A x = A^T b
//    sen::Mat<2, 3> AT = sen::transpose(A);
//    sen::Mat<2, 1> x = solve_cholesky(AT * A, AT * b);
//    print(x);
//
//    pr::PCG rng;
//    for (int i = 0; i < 1000; i++)
//    {
//        sen::Mat<4, 3> A;
//        sen::Mat<4, 1> b;
//        for (auto& v : A) { v = glm::mix(-1.0f, 1.0f, rng.uniformf()); }
//        for (auto& v : b) { v = glm::mix(-1.0f, 1.0f, rng.uniformf()); }
//
//        sen::Mat<3, 4> AT = sen::transpose(A);
//        sen::Mat<3, 1> x = solve_cholesky(AT * A, AT * b);
//
//        print(x);
//        print(sen::pinv(A) * b);
//
//        //sen::SVD<4, 3> svd = sen::svd_BV(A);
//        //for (int j = 0; j < svd.nSingulars(); j++)
//        //{
//        //    printf("singular %f\n", svd.singular(j));
//        //}
//
//        //for (auto v : x - sen::pinv(A) * b ) {
//        //    REQUIRE(fabs(v) < 1.0e-2f);
//        //}
//    }
//}

TEST_CASE("qr", "") {
    //sen::Mat<3, 2> A;
    //A.set(
    //    1, 1,
    //    0, 2,
    //    1, 0);

    //sen::Mat<3, 1> b;
    //b.set(
    //    1,
    //    2,
    //    1
    //);
    //sen::Mat<3, 3> I;
    //I.set_identity();
    //sen::QR<3, 2> qr = sen::qr_decomposition(A, I);
    //sen::print(sen::transpose(qr.Q_transposed) * qr.R);

    //unsigned int current_word = 0;
    //_controlfp_s( &current_word, _EM_ZERODIVIDE | _EM_OVERFLOW | _EM_UNDERFLOW | _EM_INEXACT, _MCW_EM );
    //

    pr::PCG rng;
    for (int i = 0; i < 100; i++)
    {
        sen::Mat<4, 4> A;
        for (auto& v : A) { v = glm::mix(-5.0f, 5.0f, rng.uniformf()); }

        sen::Mat<4, 4> I;
        I.set_identity();
        sen::QR<4, 4> qr = sen::qr_decomposition(A, I);
        sen::Mat<4, 4> A_composed = sen::transpose(qr.Q_transposed) * qr.R;

        //sen::print(A);
        //sen::print(A_composed);

        for (auto v : I - qr.Q_transposed * sen::transpose(qr.Q_transposed)) {
            REQUIRE(fabs(v) < 1.0e-5f);
        }

        for (auto v : A - A_composed) {
            REQUIRE(fabs(v) < 1.0e-5f);
        }

        sen::QR<-1, -1> qr_dynamic = sen::qr_decomposition(sen::MatDyn(A), sen::MatDyn(I));
        REQUIRE(qr.Q_transposed == qr_dynamic.Q_transposed);
        REQUIRE(qr.R == qr_dynamic.R);
    }

    for (int i = 0; i < 100; i++)
    {
        int rows = 2 + rng.uniform() % 10;
        int cols = 2 + rng.uniform() % 10;
        sen::MatDyn A;
        A.allocate(rows, cols);
        for (auto& v : A) { v = glm::mix(-1.0f, 1.0f, rng.uniformf()); }

        sen::MatDyn I;
        I.allocate(A.rows(), A.rows());
        I.set_identity();
        sen::QR<-1, -1> qr = sen::qr_decomposition(A, I);
        sen::MatDyn A_composed = sen::transpose(qr.Q_transposed) * qr.R;

        for (int i_col = 0; i_col < qr.R.cols(); i_col++)
        {
            for (int i_row = i_col + 1; i_row < qr.R.rows(); i_row++)
            {
                REQUIRE(qr.R(i_row, i_col) == 0.0f);
            }
        }

        //sen::print(A);
        //sen::print(A_composed);

        for (auto v : A - A_composed) {
            REQUIRE(fabs(v) < 1.0e-4f);
        }
    }

    BENCHMARK("QR static") {
        pr::PCG rng;
        float s = 0;
        for (int i = 0; i < 10000; i++)
        {
            sen::Mat<8, 8> A;
            for (auto& v : A) { v = glm::mix(-5.0f, 5.0f, rng.uniformf()); }

            sen::Mat<8, 8> I;
            I.set_identity();
            sen::QR<8, 8> qr = sen::qr_decomposition(A, I);
            sen::Mat<8, 8> A_composed = sen::transpose(qr.Q_transposed) * qr.R;

            s += A_composed(0, 0);
        }
        return s;
    };
}