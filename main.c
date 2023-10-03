#include <stdio.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>
#include <string.h>

#define X 240
#define Y 240

#define MAX_NUM_ITERATIONS (4 * X + 5)

const char surviveRule[9] = {0, 0, 1, 1, 0, 0, 0, 0, 0};
const char birthRule[9] = {0, 0, 0, 1, 0, 0, 0, 0, 0};

static inline int calc_num_elems(int sizeX, int sizeY, int rank, int numProcess) {
	return (ceil((double)sizeX / numProcess * (rank + 1)) - ceil((double)sizeX / numProcess * rank)) * sizeY;
}

static inline int calc_neighbors(const char* arr, int sizeY, int posX, int posY) {
	int res = 0;
	res += arr[(posX + 1) * sizeY + (posY)];
	res += arr[(posX - 1) * sizeY + (posY)];
	res += arr[(posX) * sizeY + (posY + 1) % sizeY];
	res += arr[(posX) * sizeY + (posY - 1 + sizeY) % sizeY];
	res += arr[(posX + 1) * sizeY + (posY + 1) % sizeY];
	res += arr[(posX - 1) * sizeY + (posY + 1) % sizeY];
	res += arr[(posX + 1) * sizeY + (posY - 1 + sizeY) % sizeY];
	res += arr[(posX - 1) * sizeY + (posY - 1 + sizeY) % sizeY];
	return res;
}

static inline int calc_neighbors_upper_bound(const char* arr, const char* upperLine, int sizeY, int posY) {
	int res = 0;
	res += arr[Y + (posY)];
	res += upperLine[posY];
	res += arr[(posY + 1) % sizeY];
	res += arr[(posY - 1 + sizeY) % sizeY];
	res += arr[Y + (posY + 1) % sizeY];
	res += upperLine[(posY + 1) % sizeY];
	res += arr[Y + (posY - 1 + sizeY) % sizeY];
	res += upperLine[(posY - 1 + sizeY) % sizeY];
	return res;
}

static inline int calc_neighbors_lower_bound(const char* arr, const char* lowerLine, int sizeY, int sizeX, int posY) {
	int res = 0;
	res += lowerLine[posY];
	res += arr[(sizeX - 2) * Y + posY];
	res += arr[(sizeX - 1) * sizeY + (posY + 1) % sizeY];
	res += arr[(sizeX - 1) * sizeY + (posY - 1 + sizeY) % sizeY];
	res += lowerLine[(posY + 1) % sizeY];
	res += arr[(sizeX - 2) * sizeY + (posY + 1) % sizeY];
	res += lowerLine[(posY - 1 + sizeY) % sizeY];
	res += arr[(sizeX - 2) * sizeY + (posY - 1 + sizeY) % sizeY];
	return res;
}

static inline char calc_state(int numNeighbors, char prevState) {
	return prevState * surviveRule[numNeighbors] + !prevState * birthRule[numNeighbors];
}

static inline char arr_comp(const char* arr1, const char* arr2, int size) {
	for (int i = 0; i < size; ++i) {
		if (arr1[i] != arr2[i]) {
			return 0;
		}
	}
	return 1;
}

void fill_arr_with_glider(char* arr, int sizeY) {
	arr[1] = 1;
	arr[sizeY + 2] = 1;
	arr[2 * sizeY + 0] = 1;
	arr[2 * sizeY + 1] = 1;
	arr[2 * sizeY + 2] = 1;
}

int main(int argc, char** argv) {
	double begin, end;
	MPI_Init(&argc, &argv);
	begin = MPI_Wtime();
	int num_process;
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int cellArr_p_Size = calc_num_elems(X, Y, rank, num_process);
	char** cellArrs = malloc((MAX_NUM_ITERATIONS + 2) * sizeof(char*));
	for (int i = 0; i < MAX_NUM_ITERATIONS + 2; ++i) {
		cellArrs[i] = malloc(cellArr_p_Size);
	}
	char* prevLine = malloc(Y);
	char* nextLine = malloc(Y);
	memset(prevLine, 0, Y);
	memset(nextLine, 0, Y);

	memset(cellArrs[0], 0, cellArr_p_Size);
	if (rank == 0) {
		fill_arr_with_glider(cellArrs[0], Y);
	}

	char* sumFlagsVector = malloc(MAX_NUM_ITERATIONS + 1);
	char* flagsVector = malloc(MAX_NUM_ITERATIONS + 1);

	int iterNum;
	for (iterNum = 1; iterNum < MAX_NUM_ITERATIONS; ++iterNum) {
		int prevProc = (rank - 1 + num_process) % num_process;
		int nextProc = (rank + 1) % num_process;

		MPI_Request nextProcSend, nextProcRecv, prevProcSend, prevProcRecv, flagsVectorReduce;

		MPI_Isend(cellArrs[iterNum - 1], Y, MPI_CHAR, prevProc, 'f', MPI_COMM_WORLD, &prevProcSend);
		MPI_Isend(
				cellArrs[iterNum - 1] + cellArr_p_Size - Y, Y, MPI_CHAR, nextProc, 'l', MPI_COMM_WORLD, &nextProcSend);
		MPI_Irecv(prevLine, Y, MPI_CHAR, prevProc, 'l', MPI_COMM_WORLD, &prevProcRecv);
		MPI_Irecv(nextLine, Y, MPI_CHAR, nextProc, 'f', MPI_COMM_WORLD, &nextProcRecv);

		for (int i = 0; i < (iterNum - 1); ++i) {
			flagsVector[i] = arr_comp(cellArrs[iterNum - 1], cellArrs[i], cellArr_p_Size);
		}
		MPI_Iallreduce(flagsVector, sumFlagsVector, iterNum - 1, MPI_CHAR, MPI_SUM, MPI_COMM_WORLD, &flagsVectorReduce);

		for (int i = 1; i < cellArr_p_Size / Y - 1; ++i) {
			for (int j = 0; j < Y; ++j) {
				int numNeighbors = calc_neighbors(cellArrs[iterNum - 1], Y, i, j);
				cellArrs[iterNum][i * Y + j] = calc_state(numNeighbors, cellArrs[iterNum - 1][i * Y + j]);
			}
		}

		MPI_Status status;
		MPI_Wait(&prevProcSend, &status);
		MPI_Wait(&prevProcRecv, &status);

		for (int i = 0; i < Y; ++i) {
			int numNeighbors = calc_neighbors_upper_bound(cellArrs[iterNum - 1], prevLine, Y, i);
			cellArrs[iterNum][i] = calc_state(numNeighbors, cellArrs[iterNum - 1][i]);
		}

		MPI_Wait(&nextProcSend, &status);
		MPI_Wait(&nextProcRecv, &status);

		for (int i = 0; i < Y; ++i) {
			int numNeighbors = calc_neighbors_lower_bound(cellArrs[iterNum - 1], nextLine, Y, cellArr_p_Size / Y, i);
			cellArrs[iterNum][cellArr_p_Size - Y + i] = calc_state(numNeighbors, cellArrs[iterNum - 1][cellArr_p_Size -
			                                                                                           Y + i]);
		}

		char exitFlag = 0;
		MPI_Wait(&flagsVectorReduce, &status);
		for (int i = 0; i < iterNum - 1; ++i) {
			if (sumFlagsVector[i] == num_process) {
				exitFlag = 1;
				break;
			}
		}

		if (exitFlag) {
			break;
		}
	}

	end = MPI_Wtime();

	if (rank == 0) {
		printf("Time = %f, iterNum = %d\n", (end - begin), iterNum);
	}

	for (int i = 0; i < MAX_NUM_ITERATIONS + 2; ++i) {
		free(cellArrs[i]);
	}
	free(flagsVector);
	free(sumFlagsVector);
	free(cellArrs);
	free(prevLine);
	free(nextLine);
	MPI_Finalize();
	return 0;
}
