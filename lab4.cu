#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cuda_fp16.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <bitset>
#include <chrono>

#define silent false 	//no stderr except actual errors
#define verbal true		 
#define visual false		 
#define debug  false	//do printf in kernel

#define gridsize 32

#define INDEX_ERROR 800

#define EPS 1e-7

#define CSC(call)  																											\
do {																														\
	cudaError_t err = call;																									\
	if (err != cudaSuccess) {																								\
		std::cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << ". Message: " << cudaGetErrorString(err) << "\n";		\
		exit(0);																											\
	}																														\
} while(0)

//вариант 6
//Нахождение ранга матрицы

__global__ void kernel_swap_rows(double* elements, int row_1, int row_2, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset_x = blockDim.x * gridDim.x;

	double swp;
	while(idx < m) {
		swp = elements[idx*n + row_1];
		elements[idx*n + row_1] = elements[idx*n + row_2];
		elements[idx*n + row_2] = swp;
		
		if (debug) printf("KERNEL: swaping %lf and %lf at idx %d\n",elements[idx*n + row_1], elements[idx*n + row_2], idx);

		idx += offset_x;
	}
}

__device__ bool dev_close_to_zero(double val) {
	if(fabs(val) < EPS) {
		return true;
	}
	else {
		return false;
	}
} 

__global__ void kernel_gaussian_step(double* elements, int n, int m, int start_row_index, int active_colomn) {
	//n - количество строк (элементов в столбце)
	//m - количество столбцов (элементов в строке)

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int cur_row = idx + start_row_index + 1;
	int cur_col = idy + active_colomn + 1; 
	int in_row_offset = blockDim.y * gridDim.y;
	int other_row_offset = blockDim.x * gridDim.x; 

	__shared__ double coef[32][32];

	while(cur_row < n) {
		
		coef[threadIdx.x][threadIdx.y] = - elements[active_colomn*n + cur_row] / elements[active_colomn*n + start_row_index];

		while(cur_col < m) {
			elements[cur_col*n + cur_row] = elements[cur_col*n + cur_row] + coef[threadIdx.x][threadIdx.y]*elements[cur_col*n + start_row_index];
			cur_col += in_row_offset;
		}
		cur_row += other_row_offset;
		cur_col = idy + active_colomn + 1;
	}
}

bool close_to_zero(double val) {
	if(fabs(val) < EPS) {
		return true;
	}
	return false;
}

struct Compare {
    __host__ __device__ bool operator()(double num1, double num2) {
        return fabs(num1) < fabs(num2);
    }
};


class matrix{
	int n;
	int m;
	double* array;
	double* device_matrix;
	
	double* max_elm_val; //нужно для одной функции

	

	Compare cmp;
	thrust::device_ptr<double> device_ptr;

public:
	matrix(int n_, int m_, double* array_) {
		n = n_;
		m = m_;
		array = array_;
		CSC(cudaMalloc (&device_matrix, sizeof(double)*m*n));
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
		device_ptr = thrust::device_pointer_cast(device_matrix);

		max_elm_val = (double*)malloc(sizeof(double));
	}

	matrix(int n_, int m_){ //считывание матрицы с stdin
		n = n_; m = m_;
		//n - количество строк (элементов в столбце)
		//m - количество столбцов (элементов в строке)
		double* arr_all = (double*)malloc(sizeof(double)*m*n);

		for (int i = 0; i < n; ++i){ //проход по строкам
			for (int j = 0; j < m; ++j){ //проход по столбцам
				double elm = 0;
				std::cin >> elm;
				arr_all[j*n + i] = elm;
			}
		}

		array = arr_all;

		CSC(cudaMalloc (&device_matrix, sizeof(double)*m*n));
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
		device_ptr = thrust::device_pointer_cast(device_matrix);

		max_elm_val = (double*)malloc(sizeof(double));
	}

	~matrix() {
		free(array);
		free(max_elm_val);
		CSC(cudaFree(device_matrix));
	}

	void update_host_matrix() {
		CSC(cudaMemcpy (array, device_matrix, sizeof(double)*m*n, cudaMemcpyDeviceToHost));
	}

	void update_device_matrix() {
		CSC(cudaMemcpy (device_matrix, array, sizeof(double)*m*n, cudaMemcpyHostToDevice));
	}

	void print() {

		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				std::cerr << array[i*n + j] << " ";
			}
			//std::cerr << "\n---\n";
			std::cerr << "\n";
		}
	}

	void printf() {

		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				
				if(array[i*n + j] >= 0) {
					std::printf(" ");
				}
				std::printf("%.2lf ",array[i*n + j]);

			}
			std::printf("\n");
		}
	}

	int find_max_elm(int array_start,int array_size) {
		//находит максимальный элемент массива и возвращает его индекс
		//если максимальный элемент = 0, то возвращает -1

		int max_elm_pos = thrust::max_element(device_ptr + array_start, device_ptr + array_start + array_size, cmp) - device_ptr - array_start;
		
		CSC(cudaMemcpy (max_elm_val, &device_matrix[array_start + max_elm_pos], sizeof(double), cudaMemcpyDeviceToHost));

		if(close_to_zero(max_elm_val[0])) {
			return -1;
		}

		return max_elm_pos;
	}

	void swap_rows(int lhs,int rhs) {
		//меняет местами строки lhs и rhs

		if(lhs >= n || rhs >= n) {
			throw INDEX_ERROR;
		}

		if(lhs == rhs) {
			return;
		}

		kernel_swap_rows<<<1024,1024>>>(device_matrix, lhs, rhs, n, m);

	}

	//1 - определить ведущий элемент в столбце i (thrust)
	//2 - переставить строки местами (О(m)) (параллельно, одномерной сеткой)
	//3 - вычислить коэфециенты для каждой строки (O(n)) ДЕЛАЕТСЯ ВНУТРИ ЯДРА
	//4 - записать коэфициенты в разд. память. 
	//		Каждый блок работает с одной из строк, ему нужен только один коэфициент
	//		Первый варп каждого блока считает коэфициент для соответствующей строки, следующие вары извлекают его из разделяемой памяти
	//		У каждого блока должен быть поток-лидер, который помещает нужный элемент в разделяемую память, остальные потоки должны начать работу только после завершения перемещения
	//		варп потока-лидера будет работать неоптимально
	//5 - преобразовать строки (O(n*n)) (параллельно, двумерной сеткой)
	
	int rank() {
		cudaEvent_t start, stop;
		float time_summary = 0;
		float time_max = 0;
		float time_swap = 0;
		float milliseconds = 0;
		CSC(cudaEventCreate(&start));
		CSC(cudaEventCreate(&stop));
		std::chrono::steady_clock::time_point max_begin;
		std::chrono::steady_clock::time_point max_end;
		float max_time = 0;

		int rank = -1;
		int active_element_idx = -1;

		std::chrono::steady_clock::time_point cc_begin = std::chrono::steady_clock::now();
		for (int i = 0; i < n - 1; ++i) { //i - текущая строка.
		
			active_element_idx += 1;

			if(active_element_idx == m) {
				rank = i;
				break;
			}


			CSC(cudaEventRecord(start));
			max_begin = std::chrono::steady_clock::now();
			int max_elm_idx = find_max_elm(active_element_idx*n + i,n - i) + i;

			while(max_elm_idx - i == -1 && active_element_idx + 1 < m) {//максимальный элемент строки равен нулю
				active_element_idx += 1;
				max_elm_idx = find_max_elm(active_element_idx*n + i,n - i) + i;
			}
			max_end = std::chrono::steady_clock::now();

			CSC(cudaEventRecord(stop));
			CSC(cudaEventSynchronize(stop));
			milliseconds = 0;
			CSC(cudaEventElapsedTime(&milliseconds, start, stop));
			time_max += milliseconds;
			max_time += std::chrono::duration_cast<std::chrono::microseconds>(max_end - max_begin).count();

			if(active_element_idx + 1 == m && max_elm_idx - i == -1) {
				rank = i;
				break;
			}


			if(i != max_elm_idx) {
				CSC(cudaEventRecord(start));
				swap_rows(i,max_elm_idx);
				CSC(cudaEventRecord(stop));
				CSC(cudaEventSynchronize(stop));
				milliseconds = 0;
				CSC(cudaEventElapsedTime(&milliseconds, start, stop));
				time_swap += milliseconds;
			}
									
			CSC(cudaEventRecord(start));
			kernel_gaussian_step<<<dim3(gridsize,gridsize),dim3(gridsize,gridsize)>>> (device_matrix, n, m, i, active_element_idx);
			CSC(cudaEventRecord(stop));
			CSC(cudaEventSynchronize(stop));
			milliseconds = 0;
			CSC(cudaEventElapsedTime(&milliseconds, start, stop));
			time_summary += milliseconds;


			if(active_element_idx + 1 == m) {
				rank = i + 1;
				break;
			}
		}
		std::chrono::steady_clock::time_point cc_end = std::chrono::steady_clock::now();
		std::cerr << "Finished cycle\n";
		std::cout << "total cc time: " << std::chrono::duration_cast<std::chrono::microseconds>(cc_end - cc_begin).count() / 1000 << " ms \n";
		std::cout << "total max cpu time: " << max_time / 1000 << " ms \n";

		update_host_matrix();
		//после завершения цикла нужно определить, занулилась ли последняя строка и, тем самым, понять, каков ранг
		//если цикл завершился до последней строки, то ранг уже вычислен
		if(rank == -1) {
			//надо определить, есть ли среди последних m - n элементов ненулевые
			rank = n - 1;
			for (int i = n - 1; i < m; ++i) { //было n-1
				std::cerr << array[i*n + (n-1)] << "\n";
				if(!close_to_zero(array[i*n + (n-1)])) {
					rank = n;
					break;
				}
			}
		}

		if (verbal) {
			std::cerr << rank << "\n";
			//printf();
			std::cerr << "total main kernel time: " << time_summary << " ms (" << time_summary / 1000 << " s)\n";
			std::cerr << "total max kernel time: " << time_max << " ms (" << time_max / 1000 << " s)\n";
			std::cerr << "total swap kernel time: " << time_swap << " ms (" << time_swap / 1000 << " s)\n";
		}

		return rank;

	}

};

int main() {
	
	try{ 
		int n,m;
		//std::cin >> n >> m;
		n = 2000;
		m = 10000;
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

		double* arr = (double*)malloc(m*n*sizeof(double));
		for(int i = 0; i < n*m; ++i) {
			arr[i] = std::rand()%100;
		}

		matrix matr(n,m,arr);

		std::cerr << "shape: "<< n << " " << m << "\n";

		if(visual) {
			std::cerr << "--\n";
			matr.print();
			std::cerr << "\n";
			matr.printf();
			std::cerr << "\n";
		}

		std::chrono::steady_clock::time_point rank_begin = std::chrono::steady_clock::now();
		int rank = matr.rank();
		std::chrono::steady_clock::time_point rank_end = std::chrono::steady_clock::now();

		if(visual) {
			std::cerr << "-- RANK: " << rank << " --\n";
		}
		std::cout.precision(10);
		std::cout << rank << std::fixed << "\n";
		
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "total cpu time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << " ms \n";
		std::cout << "rank calc cpu time: " << std::chrono::duration_cast<std::chrono::microseconds>(rank_end - rank_begin).count() / 1000 << " ms \n";
	}
	catch(int err) {
		if (err == 101) {
			std::cerr << "error opening file\n";
		} else
		if (err == 105){
			std::cerr << "error new length\n";
		} else 
		if (err == 800) {
			std::cerr << "error index\n";
		} else{
			std::cerr << "unknown error detected\n";
		}
	}

	return 0;
}


