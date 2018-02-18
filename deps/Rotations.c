#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_threat_num() 0
    #define omp_get_num_procs() 8
#endif

void julia_apply_givens(double * A, const double * snm, const double * cnm, const int N, const int M);
void julia_apply_givens_t(double * A, const double * snm, const double * cnm, const int N, const int M);

int main(void) {
    return 0;
}

void julia_apply_givens(double * A, const double * snm, const double * cnm, const int N, const int M) {
    #pragma omp parallel for schedule(dynamic)
    for (int m = M/2; m > 1; m--) {
        double s, c;
        double a1, a2, a3, a4;
        for (int j = m; j > 1; j = j-2) {
            for (int l = N-1-j; l >= 0; l--){
                s = snm[l+(j-2)*(2*N+3-j)/2];
                c = cnm[l+(j-2)*(2*N+3-j)/2];
                a1 = A[l+N*(2*m-1)];
                a2 = A[l+2+N*(2*m-1)];
                a3 = A[l+N*(2*m)];
                a4 = A[l+2+N*(2*m)];
                A[l+N*(2*m-1)] = c*a1 + s*a2;
                A[l+2+N*(2*m-1)] = c*a2 - s*a1;
                A[l+N*(2*m)] = c*a3 + s*a4;
                A[l+2+N*(2*m)] = c*a4 - s*a3;
            }
        }
    }
    return;
}

void julia_apply_givens_t(double * A, const double * snm, const double * cnm, const int N, const int M) {
    #pragma omp parallel for schedule(dynamic)
    for (int m = M/2; m > 1; m--) {
        double s, c;
        double a1, a2, a3, a4;
        for (int j = 2+m%2; j <= m; j = j+2) {
            for (int l = 0; l <= N-1-j; l++){
                s = snm[l+(j-2)*(2*N+3-j)/2];
                c = cnm[l+(j-2)*(2*N+3-j)/2];
                a1 = A[l+N*(2*m-1)];
                a2 = A[l+2+N*(2*m-1)];
                a3 = A[l+N*(2*m)];
                a4 = A[l+2+N*(2*m)];
                A[l+N*(2*m-1)] = c*a1 - s*a2;
                A[l+2+N*(2*m-1)] = c*a2 + s*a1;
                A[l+N*(2*m)] = c*a3 - s*a4;
                A[l+2+N*(2*m)] = c*a4 + s*a3;
            }
        }
    }
    return;
}
