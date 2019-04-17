#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>

static const int SIZE = 1200;
static const int R = SIZE / 4;

float matrix_a[SIZE][SIZE]; // lewy operand
float matrix_b[SIZE][SIZE]; // prawy operand
float matrix_r[SIZE][SIZE]; // wynik

void initialize_matrices()
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix_a[i][j] = (float)rand() / RAND_MAX;
            matrix_b[i][j] = (float)rand() / RAND_MAX;
            matrix_r[i][j] = 0.0;
        }
    }
}

void clear_result_matrix()
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrix_r[i][j] = 0.0;
        }
    }
}

void multiply_matrices_IJK()
{
#pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
}

void multiply_matrices_IJK_IJK()
{
#pragma omp parallel for
    for (int i = 0; i < SIZE; i += R)
        for (int j = 0; j < SIZE; j += R)
            for (int k = 0; k < SIZE; k += R)
                for (int ii = i; ii < i + R; ii++)
                    for (int jj = j; jj < j + R; jj++) {
                        for (int kk = k; kk < k + R; kk++)
                            matrix_r[ii][jj] += matrix_a[ii][kk] * matrix_b[kk][jj];
                    }
}

void multiply_matrices_IKJ_sequential()
{
    for (int i = 0; i < SIZE; i++)
        for (int k = 0; k < SIZE; k++)
            for (int j = 0; j < SIZE; j++)
                matrix_r[i][j] += matrix_a[i][k] * matrix_b[k][j];
}

int main()
{
    std::ofstream file;
    file.open("results.txt", std::ofstream::out | std::ofstream::app);
    omp_set_num_threads(4);

    std::cout << "Matrix size: " << SIZE << "\tR: " << R;
    file << "Matrix size: " << SIZE << "\tR: " << R;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            std::cout << "\tNum of threads: " << omp_get_num_threads() << std::endl
                      << std::endl;
            file << "\tNum of threads: " << omp_get_num_threads() << std::endl
                 << std::endl;
        }
    }

    initialize_matrices();
    {
        auto start_chrono = std::chrono::high_resolution_clock::now();

        multiply_matrices_IKJ_sequential();

        auto stop_chrono = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop_chrono - start_chrono);

        std::cout << "IKJ_seq:     " << time_span.count() << " seconds." << std::endl;
        file << "IKJ_seq:     " << time_span.count() << " seconds." << std::endl;
    }

    clear_result_matrix();
    {
        auto start_chrono = std::chrono::high_resolution_clock::now();

        multiply_matrices_IJK();

        auto stop_chrono = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop_chrono - start_chrono);

        std::cout << "IJK:         " << time_span.count() << " seconds." << std::endl;
        file << "IJK:         " << time_span.count() << " seconds." << std::endl;
    }

    clear_result_matrix();
    {
        auto start_chrono = std::chrono::high_resolution_clock::now();

        multiply_matrices_IJK_IJK();

        auto stop_chrono = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stop_chrono - start_chrono);

        std::cout << "IJK_IJK:     " << time_span.count() << " seconds." << std::endl;
        file << "IJK_IJK:     " << time_span.count() << " seconds." << std::endl;
    }

    file << "\n---------------------------------------------------------------\n\n";
    file.close();

    return 0;
}
