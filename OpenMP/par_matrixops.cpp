/**
 * Serial implementation of the matrix operations for the assignments for Parallel Computing.
 * You can use a copy of this file as a starting point for each of your parallel implementations.
 */

#include <boost/numeric/ublas/matrix.hpp>
#include <omp.h> 

namespace ublas = boost::numeric::ublas;

void fullTimesDiagonal(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t N = result.size1();
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
            for (size_t j = 0; j < N; ++j)
            {
                result(i, j) = left(i, j) * right(j, j);
            }                      
        }
}

void fullTimesFull(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t N = result.size1();
    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            result(i, j) = 0.0;
            for (size_t k = 0; k < N; ++k)
            {
                result(i, j) += left(i, k) * right(k, j);
            }
        }
    }
}

void fullTimesFullBlocked(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t const blocksize = 50;
    size_t N = result.size1();

    for (size_t i = 0; i < N / blocksize; ++i)
    {
        for (size_t j = 0; j < N / blocksize; ++j)
        {

            for (size_t i_block = 0; i_block < blocksize; ++i_block)
            {
                for (size_t j_block = 0; j_block < blocksize; ++j_block)
                {
                    result(i * blocksize + i_block, j * blocksize + j_block) = 0.0;
                }
            }
            for (size_t k = 0; k < N / blocksize; ++k)
            {
                #pragma omp parallel for
                for (size_t i_block = 0; i_block < blocksize; ++i_block)
                {
                    for (size_t j_block = 0; j_block < blocksize; ++j_block)
                    {
                        for (size_t k_block = 0; k_block < blocksize; ++k_block)
                        {
                            result(i * blocksize + i_block, j * blocksize + j_block) += left(i * blocksize + i_block, k * blocksize + k_block) * right(k * blocksize + k_block, j * blocksize + j_block);
                        }
                    }
                }
            }
        }
    }
}

void triangularTimesFull(ublas::matrix<double> &left, ublas::matrix<double> &right, ublas::matrix<double> &result)
{
    size_t N = result.size1();
    #pragma omp prallelize for collapse(2) schedule(dynamic, 16)
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            result(i, j) = 0.0;
            for (size_t k = i; k < N; ++k)
            {
                result(i, j) += left(i, k) * right(k, j);
            }
        }
    }
}