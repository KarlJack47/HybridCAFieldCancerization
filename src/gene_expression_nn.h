#ifndef __GENE_EXPRESSION_NN_H__
#define __GENE_EXPRESSION_NN_H__

__device__ void hidden_layer_activation(unsigned int idx, double *input, double alpha, double *output) {
	/*  Computes the value of the inverse square root unit: f(x) = x/sqrt(1 + alpha*x^2).
            Inputs:
            idx: current element being computed
	    input: array, the input array
	    alpha: parameter that controls range of output, that being (-1/sqrt(alpha), 1/sqrt(alpha))
            output: array, the results of the computation are to be stored here
	*/

	output[idx] = input[idx] / sqrtf(1.0f + alpha*input[idx]*input[idx]);
}

__device__ void output_layer_activation(unsigned int idx, double *input, double alpha, double *output) {
	/*  Computes the value of the inverse square root unit: f(x) = abs(x/sqrt(1 + alpha*x^2)).
            Inputs:
            idx: current element being computed
	    input: array, the input array
	    alpha: parameter that controls range of output, that being (0, 1/sqrt(alpha))
            output: array, the results of the computation are to be stored here
	*/

	output[idx] = fabsf(input[idx] / sqrtf(1.0f + alpha*input[idx]*input[idx]));
}

__device__ double* dot(unsigned int idx, double *m1, double *m2, double *output, unsigned int m1_rows , unsigned int m1_columns, unsigned int m2_columns) {
	/*  Computes the product of two matrices: m1 x m2.
	    Inputs:
	    m1: array, left matrix of size m1_rows x m1_columns
	    m2: array, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
	        must be equal to the number of the columns in the left one)
	    output: array, the results of the computation are to be stored here
	    m1_rows: int, number of rows in the left matrix m1
	    m1_columns: int, number of columns in the left matrix m1
	    m2_columns: int, number of columns in the right matrix m2
	*/

	unsigned int i, k;

	for (i = idx; i < m1_rows*m2_columns; i += m2_columns*m1_rows) {
		int r = i / m2_columns;
		int c = i % m2_columns;
		double t_output = 0.0f;
		for(k = 0; k < m1_columns; k++)
			t_output += m1[r*m1_columns+k] * m2[k*m2_columns+c];
		output[i] = t_output;
	}

	return output;
}

__device__ double* matrixAddMatrix(unsigned int idx, double *m1, double *m2, double *output) {
	/* Computes the (elementwise) addition between two arrays
	   Inputs:
	   m1: array
	   m2: array
	   output: array, the results of the computation are to be stored here
	*/

	output[idx] = m1[idx] + m2[idx];

	return output;
}

__device__ void feedforward(double *input, double *W_in, double *hidden, double *W_out, double *b_out, double *output,
			    int n_input, int n_hidden, int n_output) {

	unsigned int i;

	for (i = 0; i < n_hidden; i++)
		hidden_layer_activation(i, dot(i, W_in, input, hidden, n_hidden, n_input, 1), ALPHA, hidden);

	for (i = 0; i < n_output; i++)
		output_layer_activation(i, matrixAddMatrix(i, dot(i, W_out, hidden, output, n_hidden, n_output, 1), b_out, output), ALPHA, output);
}

struct GeneExpressionNN {
	int device;
	double *input;
	double *output;
	double *hidden;
	double *W_in;
	double *W_out;
	double *b_out;

	unsigned int n_input;
	unsigned int n_hidden;
	unsigned int n_output;

	GeneExpressionNN(unsigned int n_in, unsigned int n_out) {
		n_input = n_in;
		n_hidden = n_out;
		n_output = n_out;
	}

	void memory_allocate(double *W_x, double *W_y, double *b_y, int dev) {
		device = dev;

		CudaSafeCall(cudaMallocManaged((void**)&input, n_input*sizeof(double)));
		memset(input, 0.0f, n_input*sizeof(double));
		CudaSafeCall(cudaMallocManaged((void**)&output, n_output*sizeof(double)));
		memset(output, 0.0f, n_output*sizeof(double));
		CudaSafeCall(cudaMallocManaged((void**)&W_in, n_hidden*n_input*sizeof(double)));
		CudaSafeCall(cudaMallocManaged((void**)&hidden, n_hidden*sizeof(double)));
		memset(hidden, 0.0f, n_hidden*sizeof(double));
		CudaSafeCall(cudaMallocManaged((void**)&W_out, n_hidden*n_output*sizeof(double)));
		CudaSafeCall(cudaMallocManaged((void**)&b_out, n_output*sizeof(double)));

		memcpy(W_in, W_x, n_hidden*n_input*sizeof(double));
		memcpy(W_out, W_y, n_hidden*n_output*sizeof(double));
		memcpy(b_out, b_y, n_output*sizeof(double));
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(input));
		CudaSafeCall(cudaFree(output));
		CudaSafeCall(cudaFree(W_in));
		CudaSafeCall(cudaFree(hidden));
		CudaSafeCall(cudaFree(W_out));
		CudaSafeCall(cudaFree(b_out));
	}

	void prefetch_nn_params(int loc) {
		int location = loc;
		if (loc == -1) location = cudaCpuDeviceId;
		CudaSafeCall(cudaMemPrefetchAsync(input, n_input*sizeof(double), location, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(output, n_output*sizeof(double), location, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(hidden, n_hidden*sizeof(double), location, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(W_in, n_hidden*n_input*sizeof(double), location, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(W_out, n_hidden*n_output*sizeof(double), location, NULL));
		CudaSafeCall(cudaMemPrefetchAsync(b_out, n_output*sizeof(double), location, NULL));
	}

	__device__ void evaluate(void) {
		feedforward(input, W_in, hidden, W_out, b_out, output, n_input, n_hidden, n_output);
	}

	__device__ void mutate(double *gene_expressions) {
		for (unsigned int i = 0; i < NUM_GENES; i++) {
			if (b_out[i] <= FLT_EPSILON && !(fabsf(gene_expressions[i*2] - gene_expressions[i*2+1]) <= FLT_EPSILON)
			    && (gene_expressions[i*2] >= MUT_THRESHOLD || gene_expressions[i*2+1] >= MUT_THRESHOLD))
				b_out[i] = BIAS;
			else
				b_out[i] = 0.0f;
		}
	}
};

#endif // __GENE_EXPRESSION_NN_H__
