#pragma once

#include <intrin.h>

#if defined(SEN_ENABLE_ASSERTION)
#define SEN_ASSERT(ExpectTrue) if((ExpectTrue) == 0) { __debugbreak(); }
#else
#define SEN_ASSERT(ExpectTrue) 
#endif

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
            m_storage = std::unique_ptr<float[]>(new float[M * N]);
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
        std::unique_ptr<float[]> m_storage;
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

    private:
        template <class... Tail>
        void set_resolve(int i, float x, Tail... tail)
        {
            this->operator()( i / cols(), i % cols()) = x;
            set_resolve( i + 1, tail...);
        }
        void set_resolve(int i, float x) {
            this->operator()(i / cols(), i % cols()) = x;
            SEN_ASSERT(i + 1 == rows() * cols() && "element mismatch");
        }
    public:
        template <class... Args>
        Mat<numberOfRows, numberOfCols>& set(Args... args)
        {
            set_resolve( 0, args...);
            return *this;
        }

        Mat<numberOfRows, numberOfCols>& set_zero()
        {
            for (int i = 0; i < size(); i++)
            {
                m_storage[i] = 0.0f;
            }
            return *this;
        }
        Mat<numberOfRows, numberOfCols>& set_identity()
        {
            SEN_ASSERT(rows() == cols() && "dim mismatch");

            for (int i_col = 0; i_col < cols(); i_col++)
            for (int i_row = 0; i_row < rows(); i_row++)
            {
                (*this)(i_row, i_col) = i_row == i_col ? 1.0f : 0.0f;
            }
            return *this;
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

        float* begin() { return &m_storage[0]; }
        float* end() { return &m_storage[0] + m_storage.size(); }
        const float* begin() const { return &m_storage[0]; }
        const float* end() const { return &m_storage[0] + m_storage.size(); }

        Storage<numberOfRows, numberOfCols> m_storage;
    };

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
        r.allocate(m.rows(), m.cols());
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
        r.allocate(m.rows(), m.cols());
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

    template <int rows, int cols>
    float column_dot(const Mat<rows, cols>& m, int col_0, int col_1)
    {
        int col_0_head = col_0 * m.rows();
        int col_1_head = col_1 * m.rows();
        float r = 0.0f;
        for (int i_row = 0; i_row < m.rows(); i_row++)
        {
            r += m[col_0_head + i_row] * m[col_1_head + i_row];
        }
        return r;
    }

    // all combinations of 0 to n - 1
#define CYCLIC_BY_ROW(n, a, b) \
    for (int a = 0; a < (n); a++) \
    for (int b = a + 1; b < (n); b++)

    // Economy SVD
    // A = B x transpose(V)
    template <int rows, int cols>
    struct SVD
    {
        Mat<cols, cols> V;
        Mat<rows, cols> B; // B = A x J1 x J2 ... JN, where J is rotation

        float singular(int i) const
        {
            return sqrtf(column_dot(B, i, i));
        }
        int nSingulars() const
        {
            return B.cols();
        }
        Mat<cols, rows> pinv( float tol = 1e-7f ) const
        {
            auto B_reg = B;
            for (int i_col = 0; i_col < B_reg.cols(); i_col++)
            {
                float sigma2 = column_dot(B_reg, i_col, i_col);
                for (int i_row = 0; i_row < B_reg.rows(); i_row++)
                {
                    B_reg(i_row, i_col) = sigma2 < tol * tol ? 0.0f : B_reg(i_row, i_col) / sigma2;
                }
            }
            return V * transpose(B_reg);
        }
    };
    using SVDDyn = SVD<-1, -1>;

    template <int rows, int cols>
    SVD<rows, cols> svd_BV(const Mat<rows, cols>& A)
    {
        Mat<rows, cols> B = A;
        Mat<cols, cols> V;
        V.allocate(A.cols(), A.cols());
        V.set_identity();

        const float tol = 1.e-06f; // a bit larger than machine eps

        bool converged = false;
        while( !converged )
        {
            converged = true;

            CYCLIC_BY_ROW(B.cols(), index_b1, index_b2)
            {
                float non_diag = column_dot(B, index_b1, index_b2);
                float diag1 = column_dot(B, index_b1, index_b1);
                float diag2 = column_dot(B, index_b2, index_b2);
                float Py = 2.0f * non_diag;
                float Px = diag1 - diag2;
                float PL = sqrtf(Px * Px + Py * Py);
                
                // A stopping criterion
                // "4 One-sided Jacobi", Jameset al. el, Jacobi's method is more accurate than QR
                // note that diag1 and diag2 can be zero in some bad-conditioned matrix but need to support them.
                if( diag1 * diag2 <= FLT_MIN || non_diag * non_diag <= ( tol * tol ) * diag1 * diag2 )
                {
                    continue;
                }
                if (PL == 0.0f || Py == 0.0f )
                {
                    continue; // no rotation
                }

                converged = false;

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

                for (int i_row = 0; i_row < B.rows(); i_row++)
                {
                    float b1 = B(i_row, index_b1);
                    float b2 = B(i_row, index_b2);
                    B(i_row, index_b1) = +c * b1 + s * b2;
                    B(i_row, index_b2) = -s * b1 + c * b2;
                }

                for (int i_row = 0; i_row < V.rows(); i_row++)
                {
                    float b1 = V(i_row, index_b1);
                    float b2 = V(i_row, index_b2);
                    V(i_row, index_b1) = +c * b1 + s * b2;
                    V(i_row, index_b2) = -s * b1 + c * b2;
                }
            }
        }

        return { V, B };
    }

    template <int rows, int cols>
    Mat<cols, rows> pinv(const Mat<rows, cols>& A)
    {
        auto svd = sen::svd_BV(A);
        return svd.pinv();
    }

    template <int s>
    Mat<s, s> cholesky_decomposition(const Mat<s, s>& A)
    {
        Mat<s, s> L;
        L.allocate(A.rows(), A.cols());
        L.set_zero();
        for (int i_col = 0; i_col < L.cols(); i_col++)
        {
            for (int i_row = i_col; i_row < L.rows(); i_row++)
            {
                float sum = 0.0f;
                for (int i = 0; i < i_col; i++)
                {
                    sum += L(i_row, i) * L(i_col, i);
                }
                float v = A(i_row, i_col) - sum;
                float l = i_col == i_row ? sqrtf(v) : v / L(i_col, i_col);
                if (isfinite(l) == false)
                {
                    L.set_zero();
                    return L;
                }
                L(i_row, i_col) = l;
            }
        }

        return L;
    }

    template <int s, int t /*1 or - 1*/>
    inline sen::Mat<s, t> solve_cholesky( const Mat<s, s>& A, const sen::Mat<s, t>& b )
    {
        static_assert(t == 1 || t == -1, "invalid b");
        Mat<s, s> L = cholesky_decomposition(A);

        print(L);

        // solve Lb'=b
        sen::Mat<s, t> bp;
        bp.allocate(b.rows(), 1);
        for (int i_row = 0; i_row < b.rows(); i_row++)
        {
            float sum = 0.0f;
            for (int i = 0; i < i_row; i++)
            {
                sum += L(i_row, i) * bp(i, 0);
            }
            bp(i_row, 0) = (b(i_row, 0) - sum) / L(i_row, i_row);
        }

        print(bp);

        sen::Mat<s, t> x;
        x.allocate(b.rows(), 1);
        for (int i_row = b.rows() - 1; 0 <= i_row; i_row--)
        {
            float sum = 0.0f;
            for (int i = i_row + 1; i < b.rows(); i++)
            {
                sum += L(i, i_row) * x(i, 0);
            }
            x(i_row, 0) = (bp(i_row, 0) - sum) / L(i_row, i_row);
        }

        print(x);

        return x;
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
}