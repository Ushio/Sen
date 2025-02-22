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

    // Example of Mat<3, 2>
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
                m(i_row, i_col) = xs[i_row * numberOfCols + i_col];
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
                m(i_row, i_col) = xs[i_row * numberOfCols + i_col];
            }
            return m;
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
            for (int i = 0; i < lhs.cols(); i++)
            {
                value += lhs(dst_row, i) * rhs(i, dst_col);
            }
            r(dst_row, dst_col) = value;
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
            r(dst_row, dst_col) = lhs(dst_row, dst_col) - rhs(dst_row, dst_col);
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
}