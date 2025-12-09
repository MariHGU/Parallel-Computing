/**
 * Serial implementatin of the game of life for the course Parallel Computing.
 * Emil Loevbak (emil.loevbak@cs.kuleuven.be)
 * First implementation: November 2019
 * Updated to ublas datastructures in December 2022
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <boost/numeric/ublas/matrix.hpp>
#include <ctime>
#include <mpi.h>
#include <stdio.h>

namespace ublas = boost::numeric::ublas;

int const globalBufferLength = 50;

void initializeBoard(ublas::matrix<bool> &board, int processID)
{
    int deadCellMultiplyer = 2;

    // do random seed per process, to ensure difference
    unsigned int seed = time(0) * processID;
    srand(seed);
    for (auto row = board.begin1()+1; row != board.end1()-1; ++row)
    {
        for (auto element = row.begin(); element != row.end(); ++element)
        {
            *element = (rand() % (deadCellMultiplyer + 1) == 0);
        }
    }
}

void updateBoard(ublas::matrix<bool> &board)
{
    const size_t totalRows = board.size1();
    const size_t cols = board.size2();
    ublas::matrix<int> liveNeighbors(totalRows, cols, 0);

    //Count live neigbors
    for (size_t i = 0; i < totalRows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            if (board(i, j))
            {
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        int ni = i + di;
                        int nj = j + dj;

                        nj = (nj + cols) % cols;

                        //Periodic boundary conditions
                        if (ni >= 0 && ni < totalRows){
                            liveNeighbors(ni,nj)++;
                        }
                    }
                }
                liveNeighbors(i, j)--; //Correction so that a cell does not concider itself as a live neighbor
            }
        }
    }

    //Update board
    // Change to i = 1 and rows -1, cols-1
    for (size_t i = 1; i < totalRows-1; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            board(i, j) = ((liveNeighbors(i, j) == 3) || (board(i, j) && liveNeighbors(i, j) == 2));
        }
    }
}

void writeBoardToFile(ublas::matrix<bool> &board, size_t firstRow, size_t lastRow, size_t firstCol, size_t lastCol, std::string fileName, int iteration, unsigned int processID)
{
    //Open file
    std::ofstream outputFile(fileName + "_" + std::to_string(iteration) + "_" + std::to_string(processID) + ".gol");
    //Write metadata
    outputFile << std::to_string(firstRow) << " " << std::to_string(lastRow) << std::endl;
    outputFile << std::to_string(firstCol) << " " << std::to_string(lastCol) << std::endl;
    //Write data
    std::ostream_iterator<bool> output_iteratable(outputFile, "\t");
    // maybe skip ghost lines?
    for (auto row = board.begin1(); row != board.end1(); ++row)
    {
        copy(row.begin(), row.end(), output_iteratable);
        outputFile << std::endl;
    }
    
    outputFile.close();
}


std::string setUpProgram(size_t rows, size_t cols, int iteration_gap, int iterations, int processes)
{
    //Generate progam name based on current time, all threads should use the same name!
    time_t rawtime;
    struct tm *timeInfo;
    char buffer[globalBufferLength];
    time(&rawtime);
    timeInfo = localtime(&rawtime);
    strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M-%S", timeInfo);
    std::string programName(buffer);

    //Generate main file
    std::ofstream outputFile(programName + ".gol");
    outputFile << std::to_string(rows) << " " << std::to_string(cols) << " " << std::to_string(iteration_gap) << " " << std::to_string(iterations) << " " << std::to_string(processes) << std::endl;
    outputFile.close();

    return programName;
}

void broadcastProgram(std::string &programName, int processID){
    char buffer[globalBufferLength];

    if (processID == 0){
        // only process 0 should broadcast
        // copies name into buffer
        std::snprintf(buffer, globalBufferLength, "%s", programName.c_str());
    }

    MPI_Bcast(buffer, globalBufferLength, MPI_CHAR, 0, MPI_COMM_WORLD);

    programName = std::string(buffer);
}

void exchangeGhostRows(ublas::matrix<unsigned char> &board, int rows, int cols, int processID, int processes){
    int up = (processID -1 + processes) % processes;
    int down = (processID + 1) % processes;

    // ublas is row-major -> row elements of row i are contiguous

    MPI_Sendrecv(
        &board(1,0), cols, MPI_UNSIGNED_CHAR, down, 0, // send first real row
        &board(rows+1, 0), cols, MPI_UNSIGNED_CHAR, down, 0, // recieve bottom ghost row
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        &board(rows, 0), cols, MPI_UNSIGNED_CHAR, up, 1, // send bottom real row
        &board(0, 0), cols, MPI_UNSIGNED_CHAR, up, 1,// recieve first ghost row
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    ); 
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "This program should be called with four arguments! \nThese should be, the total number of rows; the total number of columns; the gap between saved iterations and the total number of iterations, in that order." << std::endl;
        return 1;
    } 
    int iteration_gap, iterations;
    size_t rows, cols;
    try
    {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        iteration_gap = atoi(argv[3]);
        iterations = atoi(argv[4]);
    }
    catch (std::exception const &exc)
    {
        std::cout << "One or more program arguments are invalid!" << std::endl;
        return 1;
    }

    int processes, processID;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &processID);
    
    std::string programName;
    if (processID == 0){
        programName = setUpProgram(rows, cols, iteration_gap, iterations, processes);
    }
    
    broadcastProgram(programName, processID);
    
    //Build board
    int row_width = rows/processes; // assuming divisability
    // ublas::matrix<bool> board(lastRow - firstRow + 1, lastCol - firstCol + 1); global board
    
    ublas::matrix<bool> board(row_width+2, cols); // local board
    size_t localFirstRow = processID * row_width;
    size_t localLastRow = localFirstRow + row_width - 1, firstCol = 0, lastCol = cols - 1;
    //size_t firstRow = 0, lastRow = rows - 1, firstCol = 0, lastCol = cols - 1;
    
    // Initialize local boards
    initializeBoard(board, processID);
    //Do iteration    initializeBoard(board);

    writeBoardToFile(board, localFirstRow, localLastRow, firstCol, lastCol, programName, 0, processID);
    for (int i = 1; i <= iterations; ++i)
    {
        exchangeGhostRows(board, row_width, cols, processID, processes);
        updateBoard(board);
        if (i % iteration_gap == 0)
        {
            writeBoardToFile(board, localFirstRow, localLastRow, firstCol, lastCol, programName, i, processID);
        }
    }
    MPI_Finalize();
}
