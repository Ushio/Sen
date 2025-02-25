#pragma once

#include <intrin.h>
#define SEN_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }
//#define SEN_ASSERT(ExpectTrue) 

namespace sen
{
    template<bool B, class T, class F>
    struct cond { using type = T; };

    template<class T, class F>
    struct cond<false, T, F> { using type = F; };

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

    // Example of Mat<3 /*out*/, 2 /*in*/>
    // | m(0,0), m(0,1) |
    // | m(1,0), m(1,1) |
    // | m(2,0), m(2,1) |
    template <int numberOfRows, int numberOfCols>
    struct Mat
    {
        Mat() {}

        template <int M, int N>
        Mat(const Mat<M, N>& rhs /* static or dynamic */)
        {
            m_storage.allocate(M, N);
            
            static_assert(numberOfRows == -1 /*ignore dynamic case*/  || M == -1 /*ignore dynamic case*/ || (numberOfRows == M && numberOfCols == N));
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
        void set_zero()
        {
            for (int i = 0; i < size(); i++)
            {
                m_storage[i] = 0.0f;
            }
        }
        void set_identity()
        {
            SEN_ASSERT(rows() == cols() && "dim mismatch");

            for (int i_col = 0; i_col < cols(); i_col++)
            for (int i_row = 0; i_row < rows(); i_row++)
            {
                (*this)(i_row, i_col) = i_row == i_col ? 1.0f : 0.0f;
            }
        }

        int rows() const { return m_storage.rows(); }
        int cols() const { return m_storage.cols(); }
        int size() const { return m_storage.size(); }

        float& operator()(int i_row, int i_col)
        {
            SEN_ASSERT(0 <= i_col && i_col < cols() && "out of bounds");
            SEN_ASSERT(0 <= i_row && i_row < rows() && "out of bounds");
            return m_storage[i_col * m_storage.rows() + i_row];
        }
        const float& operator()(int i_row, int i_col) const
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
        using RowType = typename cond<numberOfRows == -1 && numberOfCols == -1, Mat<-1, -1>, Mat<1, numberOfCols>>::type;

        ColType col(int i_col) const
        {
            SEN_ASSERT(0 <= i_col && i_col < cols() && "out of bounds");

            ColType c;
            c.allocate(rows(), 1);
            for (int i = 0; i < rows(); i++)
            {
                c(i, 0 ) = (*this)(i, i_col);
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
                (*this)(i, i_col) = c(i, 0);
            }
        }
        void set_col(int i_col, const Mat<numberOfRows, 1>& c)
        {
            set_col<numberOfRows, 1>(i_col, c);
        }

        RowType row(int i_row) const 
        {
            SEN_ASSERT(0 <= i_row && i_row < rows() && "");
            RowType r;
            r.allocate(1, cols());
            for (int i = 0; i < cols(); i++)
            {
                r(0, i) = (*this)(i_row, i);
            }
            return r;
        }
        template<int M, int N>
        void set_row(int i_row, const Mat<M, N>& r /* static or dynamic */)
        {
            SEN_ASSERT(0 <= i_row && i_row < rows() && "");
            SEN_ASSERT(cols() == r.cols());
            for (int i = 0; i < cols(); i++)
            {
                (*this)(i_row, i) = r(0, i);
            }
        }
        void set_row(int i_row, const Mat<1, numberOfCols>& r)
        {
            set_row<1, numberOfCols>(i_row, r);
        }

        float* begin() { return &m_storage[0]; }
        float* end() { return &m_storage[0] + m_storage.size(); }
        const float* begin() const { return &m_storage[0]; }
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
                m(i_row, i_col) = xs[i_row * numberOfCols + i_col];
            }
            return m;
        }

        operator Mat<-1, -1>() const {
            SEN_ASSERT(index == numberOfRows * numberOfCols && "initialize failure");
            return this->operator sen::Mat<numberOfRows, numberOfCols>();
        }
    };

    template <int numberOfRows, int numberOfCols>
    RowMajorInitializer<numberOfRows, numberOfCols> mat_of(float v)
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
                printf("%.8f, ", m(i_row, i_col));
            }
            printf("\n");
        }
        printf("}\n");
    }

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    bool operator==(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        static_assert(lhs_cols == -1 || rhs_cols == -1 /*ignore dynamic*/ || lhs_rows == rhs_rows, "invalid comparison");
        static_assert(lhs_cols == -1 || rhs_cols == -1 /*ignore dynamic*/ || lhs_cols == rhs_cols, "invalid comparison");
        SEN_ASSERT(lhs.cols() == rhs.cols() && "invalid comparison");
        SEN_ASSERT(lhs.rows() == rhs.rows() && "invalid comparison");

        for (int i = 0; i < lhs.size(); i++)
        {
            if (lhs[i] != rhs[i])
            {
                return false;
            }
        }
        return true;
    }
    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    bool operator!=(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        return !(lhs == rhs);
    }

    template <int rows, int cols>
    struct ConservativelyDynamic
    {
        using type = typename cond<rows == -1 || cols == -1, Mat<-1, -1>, Mat<rows, cols>>::type;
    };

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    typename ConservativelyDynamic<lhs_rows, rhs_cols>::type operator*(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        typename ConservativelyDynamic<lhs_rows, rhs_cols>::type r;
        static_assert(lhs_cols == -1 || rhs_cols == -1 /*ignore dynamic*/ || lhs_cols == rhs_rows, "invalid multiplication");
        SEN_ASSERT(lhs.cols() == rhs.rows() && "invalid multiplication");

        r.allocate(lhs.rows(), rhs.cols());

        for (int dst_row = 0; dst_row < r.rows(); dst_row++)
        for (int dst_col = 0; dst_col < r.cols(); dst_col++)
        {
            float value = 0.0f;
            for (int i = 0; i < lhs.cols(); i++)
            {
                value += lhs(dst_row, i) * rhs(i, dst_col);
            }
            r(dst_row, dst_col) = value;
        }

        return r;
    }

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols, class T>
    typename ConservativelyDynamic<lhs_rows, rhs_cols>::type element_wise_op_binary(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs, T f)
    {
        static_assert(lhs_cols == -1 || rhs_cols == -1 /*ignore dynamic*/ || lhs_rows == rhs_rows, "invalid substruct");
        static_assert(lhs_cols == -1 || rhs_cols == -1 /*ignore dynamic*/ || lhs_cols == rhs_cols, "invalid substruct");

        typename ConservativelyDynamic<lhs_rows, rhs_cols>::type r;

        r.allocate(lhs.rows(), lhs.cols());

        for (int i = 0; i < lhs.size(); i++)
        {
            r[i] = f(lhs[i], rhs[i]);
        }

        return r;
    }

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    typename ConservativelyDynamic<lhs_rows, rhs_cols>::type operator-(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        return element_wise_op_binary(lhs, rhs, [](float a, float b) { return a - b; });
    }
    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    typename ConservativelyDynamic<lhs_rows, rhs_cols>::type operator+(const Mat<lhs_rows, lhs_cols>& lhs, const Mat<rhs_rows, rhs_cols>& rhs)
    {
        return element_wise_op_binary(lhs, rhs, [](float a, float b) { return a + b; });
    }

    // Mat vs scaler
    template <int rows, int cols>
    Mat<rows, cols> operator*(const Mat<rows, cols>& m, float x)
    {
        Mat<rows, cols> r;
        r.allocate(m.cols(), m.rows());
        for (int i = 0; i < m.size(); i++)
        {
            r[i] = m[i] * x;
        }
        return r;
    }
    template <int rows, int cols>
    Mat<rows, cols> operator*(float x, const Mat<rows, cols>& m)
    {
        return m * x;
    }

    template <int rows, int cols>
    Mat<rows, cols> operator/(const Mat<rows, cols>& m, float x)
    {
        Mat<rows, cols> r;
        r.allocate(m.cols(), m.rows());
        for (int i = 0; i < m.size(); i++)
        {
            r[i] = m[i] / x;
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
                r(i_col, i_row) = m(i_row, i_col);
            }
        }
        return r;
    }

    template <int lhs_rows, int lhs_cols, int rhs_rows, int rhs_cols>
    float v_dot(const Mat<lhs_rows, lhs_cols>& a, const Mat<rhs_rows, rhs_cols>& b)
    {
        return ( transpose(a) * b )( 0, 0 );
    }

    template <int rows, int cols>
    float v_length(const Mat<rows, cols>& a)
    {
        return sqrtf(v_dot(a, a));
    }

    // all combinations of 0 to n - 1
#define CYCLIC_BY_ROW(n, a, b) \
    for (int a = 0; a < (n); a++) \
    for (int b = a + 1; b < (n); b++)

    // Economy SVD
    template <int rows, int cols>
    struct SVD
    {
        Mat<rows, cols> U;
        Mat<cols, cols> sigma;
        Mat<cols, cols> V_transposed;
    };

    template <int rows, int cols>
    SVD<rows, cols> svd_unordered(const Mat<rows, cols>& A)
    {
        static_assert(cols <= rows, "use svd_underdetermined_unordered()");

        Mat<rows, cols> B = A;
        Mat<cols, cols> V;
        V.set_identity();

        float convergence_previous = FLT_MAX;
        for (;;)
        {
            float convergence = 0.0f;

            CYCLIC_BY_ROW(cols, index_b1, index_b2)
            {
                Mat<rows, 1> b1 = B.col(index_b1);
                Mat<rows, 1> b2 = B.col(index_b2);

                float Py = 2.0f * v_dot(b1, b2);
                float Px = v_dot(b1, b1) - v_dot(b2, b2);
                float PL = sqrtf(Px * Px + Py * Py);

                convergence = ss_max(convergence, fabs(Py));

                if (PL == 0.0f || Py == 0.0f )
                {
                    continue; // no rotation
                }

                float sgn = 0.0f < Px ? 1.0f : -1.0f;
                float Hx = PL + Px * sgn;
                float Hy = Py * sgn;
                float L = sqrtf(Hx * Hx + Hy * Hy);
                float c = Hx / L;
                float s = Hy / L;

                // Equivalent to:
                //float two_theta = atan(Py / Px);
                //float s = sin(two_theta * 0.5f);
                //float c = cos(two_theta * 0.5f);

                B.set_col(index_b1, +c * b1 + s * b2);
                B.set_col(index_b2, -s * b1 + c * b2);

                auto b1_v = V.col(index_b1);
                auto b2_v = V.col(index_b2);
                V.set_col(index_b1, +c * b1_v + s * b2_v);
                V.set_col(index_b2, -s * b1_v + c * b2_v);
            }

            if (convergence < convergence_previous && convergence != 0.0f)
            {
                convergence_previous = convergence;
                continue;
            }
            break;
        }

        SVD<rows, cols> svd;

        // B = UA
        svd.sigma.set_zero();

        svd.U = B;
        for (int i_col = 0; i_col < cols; i_col++)
        {
            auto col = B.col(i_col);
            float sigma_i = v_length(col);
            svd.sigma(i_col, i_col) = sigma_i;
            svd.U.set_col(i_col, col / sigma_i); // need zero check?
        }

        Mat<cols, cols> inv_sigma = svd.sigma;
        for (int i = 0; i < inv_sigma.size(); i++)
        {
            if (inv_sigma[i] != 0.0f)
            {
                inv_sigma[i] = 1.0f / inv_sigma[i];
            }
        }
        
        svd.V_transposed = transpose(V);

        return svd;
    }

    template <int rows, int cols>
    struct SVD_underdetermined
    {
        Mat<rows, rows> U;
        Mat<rows, rows> sigma;
        Mat<rows, cols> V_transposed;
    };

    template <int rows, int cols>
    SVD_underdetermined<rows, cols> svd_unordered_underdetermined(const Mat<rows, cols>& A)
    {
        static_assert(rows < cols, "use svd_underdetermined()");

        SVD<cols, rows> svd_transposed = svd_unordered(transpose(A));
        return {
            transpose(svd_transposed.V_transposed),
            svd_transposed.sigma,
            transpose(svd_transposed.U)
        };
    }

    template <int dim>
    Mat<dim, dim> inverse(const Mat<dim, dim>& A)
    {
        //static_assert(rows == cols, "must be square");
        //static_assert(0, "not implemented");

        SVD<dim, dim> svd = svd_unordered(A);

        // TODO, refactor
        for (auto& s : svd.sigma)
        {
            if (s != 0.0f)
                s = 1.0f / s;
        }
        return sen::transpose(svd.V_transposed) * svd.sigma * sen::transpose(svd.U);
    }
    template <int rows, int cols>
    float det(const Mat<rows, cols>& A)
    {
        static_assert(rows == cols, "must be square");
        static_assert(0, "not implemented");
    }
    template <>
    float det(const Mat<2, 2>& A)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }

    template <>
    Mat<2, 2> inverse(const Mat<2, 2>& A)
    {
        Mat<2, 2> r = mat_of<2, 2>
            (A(1, 1))(-A(0, 1))
            (-A(1, 0))(A(0, 0));
        return r / det(A);
    }
}