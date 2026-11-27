/* kpeig4: eig probe 6 kernels with an explicit leading dimension.
 * Calls MATLAB's own BLAS (libmwblas, MKL, ILP64). Build:
 *   mex -R2018a kpeig4.c -lmwblas
 *
 *   B = kpeig4('trmm_ld', side, uplo, transa, diag, alpha, A, B, ldb)
 *       dtrmm with B (m x n) placed in a buffer with leading dimension ldb >= m (the words between
 *       rows m+1..ldb are NaN and are not read); returns the m x n result.
 *   C = kpeig4('gemm_ld', ta, tb, alpha, A, B, beta, C, lda, ldb, ldc)
 *       dgemm with each matrix placed in a buffer with the given leading dimension (>= its rows).
 */
#include "mex.h"
#include <stddef.h>
#include <string.h>
#include <math.h>
typedef ptrdiff_t I;
extern void dgemm_(const char *, const char *, const I *, const I *, const I *, const double *, const double *, const I *,
                   const double *, const I *, const double *, double *, const I *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const I *, const I *, const double *,
                   const double *, const I *, double *, const I *);

static int is(const char *a, const char *b) { return !strcmp(a, b); }
static void str(const mxArray *a, char *buf, size_t n, const char *what) {
    if (!mxIsChar(a) || mxGetString(a, buf, (mwSize)n)) mexErrMsgIdAndTxt("kpeig4:arg", "%s must be a short char", what);
}
static double num(const mxArray *a, const char *what) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("kpeig4:arg", "%s must be a real scalar", what);
    return mxGetScalar(a);
}
static void real(const mxArray *a, const char *what) {
    if (!mxIsDouble(a) || mxIsComplex(a)) mexErrMsgIdAndTxt("kpeig4:arg", "%s must be real double", what);
}
/* copy an r x c matrix into a NaN-filled buffer with leading dimension ld */
static double *place(const mxArray *a, I ld, const char *what) {
    real(a, what);
    I r = (I)mxGetM(a), c = (I)mxGetN(a);
    if (ld < r || ld < 1) mexErrMsgIdAndTxt("kpeig4:arg", "%s: leading dimension below its rows", what);
    double *buf = mxMalloc((size_t)ld * (size_t)(c > 0 ? c : 1) * sizeof(double));
    for (I t = 0; t < ld * (c > 0 ? c : 1); t++) buf[t] = NAN;
    const double *s = mxGetPr(a);
    for (I j = 0; j < c; j++) memcpy(buf + j * ld, s + j * r, (size_t)r * sizeof(double));
    return buf;
}
static mxArray *unplace(const double *buf, I r, I c, I ld) {
    mxArray *o = mxCreateDoubleMatrix((mwSize)r, (mwSize)c, mxREAL);
    double *d = mxGetPr(o);
    for (I j = 0; j < c; j++) memcpy(d + j * r, buf + j * ld, (size_t)r * sizeof(double));
    return o;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    char mode[32];
    if (nrhs < 1) mexErrMsgIdAndTxt("kpeig4:arg", "mode required");
    str(prhs[0], mode, sizeof mode, "mode");
#define NEED(k) do { if (nrhs < (k)) mexErrMsgIdAndTxt("kpeig4:arg", "%s needs %d arguments", mode, (k) - 1); } while (0)
    (void)nlhs;
    if (is(mode, "trmm_ld")) {
        NEED(9);
        char side[4], uplo[4], ta[4], dg[4];
        str(prhs[1], side, sizeof side, "side"); str(prhs[2], uplo, sizeof uplo, "uplo");
        str(prhs[3], ta, sizeof ta, "transa"); str(prhs[4], dg, sizeof dg, "diag");
        double alpha = num(prhs[5], "alpha");
        const mxArray *Am = prhs[6], *Bm = prhs[7];
        real(Am, "A");
        I m = (I)mxGetM(Bm), n = (I)mxGetN(Bm), lda = mxGetM(Am) > 0 ? (I)mxGetM(Am) : 1, ldb = (I)num(prhs[8], "ldb");
        double *B = place(Bm, ldb, "B");
        dtrmm_(side, uplo, ta, dg, &m, &n, &alpha, mxGetPr(Am), &lda, B, &ldb);
        plhs[0] = unplace(B, m, n, ldb);
        mxFree(B);
        return;
    }
    if (is(mode, "gemm_ld")) {
        NEED(11);
        char ta[4], tb[4];
        str(prhs[1], ta, sizeof ta, "transa"); str(prhs[2], tb, sizeof tb, "transb");
        double alpha = num(prhs[3], "alpha"), beta = num(prhs[6], "beta");
        const mxArray *Am = prhs[4], *Bm = prhs[5], *Cm = prhs[7];
        I lda = (I)num(prhs[8], "lda"), ldb = (I)num(prhs[9], "ldb"), ldc = (I)num(prhs[10], "ldc");
        I m = (I)mxGetM(Cm), n = (I)mxGetN(Cm), k = (ta[0] == 'N') ? (I)mxGetN(Am) : (I)mxGetM(Am);
        double *A = place(Am, lda, "A"), *B = place(Bm, ldb, "B"), *C = place(Cm, ldc, "C");
        dgemm_(ta, tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
        plhs[0] = unplace(C, m, n, ldc);
        mxFree(A); mxFree(B); mxFree(C);
        return;
    }
    mexErrMsgIdAndTxt("kpeig4:mode", "unknown mode %s", mode);
}
