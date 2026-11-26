/* kpeig2: stage and kernel probe for MATLAB's symmetric eig path
 * (dsyevd 'U' = dsytrd + dstedc + dormtr). Calls MATLAB's own LAPACK/BLAS
 * (libmwlapack/libmwblas, MKL, ILP64). Build:
 *   mex -R2018a kpeig2.c -lmwlapack -lmwblas
 *
 *   v               = kpeig2('ilaenv', ispec, name, opts, n1, n2, n3, n4)
 *   [d, e, tau, Ar] = kpeig2('sytd2', A, uplo)
 *   [Ar, e, tau, W] = kpeig2('latrd', A, nb, uplo)
 *   C               = kpeig2('ormtr', Ar, tau, C, uplo)       side L, trans N
 *   y               = kpeig2('symv', alpha, A, x, uplo)        beta = 0, y = 0
 *   A               = kpeig2('syr2', alpha, x, y, A, uplo)
 *   C               = kpeig2('syr2k', alpha, A, B, beta, C, uplo, trans)
 *   y               = kpeig2('gemv', trans, alpha, A, x, beta, y)
 *   s               = kpeig2('dot', x, y)
 *   s               = kpeig2('nrm2', x)
 *   y               = kpeig2('axpy', alpha, x, y)
 *   [beta, v, tau]  = kpeig2('larfg', alpha, x)                 n = numel(x) + 1
 *   R               = kpeig2('fprev_symv', n, rows, uplo)
 *   r               = kpeig2('fprev_dot', n)
 *
 * fprev_*: FPRev summation-tree measurement. For every pair p < q (1-based,
 * p ascending, then q ascending) the summands are all 1 except +M at p and -M
 * at q, with M = 2^70 (so any partial sum of at most 400 ones added to M is
 * absorbed). The result is the number of ones added after +M and -M met,
 * i.e. n minus the size of the subtree where p and q first join.
 *   fprev_symv: A = all ones (both triangles), x as above, alpha = 1, beta = 0;
 *     R(pair, r) = y(rows(r)). The summands of y(j) are A(j,k) x(k), k = 1..n.
 *   fprev_dot: x as above, y = all ones; r(pair) = ddot(n, x, 1, y, 1).
 */
#include "mex.h"
#include <stddef.h>
#include <string.h>
#include <math.h>
typedef ptrdiff_t I;
extern I ilaenv_(const I *, const char *, const char *, const I *, const I *, const I *, const I *, size_t, size_t);
extern void dsytd2_(const char *, const I *, double *, const I *, double *, double *, double *, I *);
extern void dlatrd_(const char *, const I *, const I *, double *, const I *, double *, double *, double *, const I *);
extern void dormtr_(const char *, const char *, const char *, const I *, const I *, const double *, const I *,
                    const double *, double *, const I *, double *, const I *, I *);
extern void dsymv_(const char *, const I *, const double *, const double *, const I *, const double *, const I *,
                   const double *, double *, const I *);
extern void dsyr2_(const char *, const I *, const double *, const double *, const I *, const double *, const I *, double *, const I *);
extern void dsyr2k_(const char *, const char *, const I *, const I *, const double *, const double *, const I *,
                    const double *, const I *, const double *, double *, const I *);
extern void dgemv_(const char *, const I *, const I *, const double *, const double *, const I *, const double *, const I *,
                   const double *, double *, const I *);
extern double ddot_(const I *, const double *, const I *, const double *, const I *);
extern double dnrm2_(const I *, const double *, const I *);
extern void daxpy_(const I *, const double *, const double *, const I *, double *, const I *);
extern void dlarfg_(const I *, double *, double *, const I *, double *);

static const I ONE = 1;

static void str(const mxArray *a, char *buf, size_t n, const char *what) {
    if (!mxIsChar(a) || mxGetString(a, buf, (mwSize)n)) mexErrMsgIdAndTxt("kpeig2:arg", "%s must be a short char", what);
}
static double num(const mxArray *a, const char *what) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("kpeig2:arg", "%s must be a real scalar", what);
    return mxGetScalar(a);
}
static mxArray *dup(const mxArray *a, const char *what) {
    if (!mxIsDouble(a) || mxIsComplex(a)) mexErrMsgIdAndTxt("kpeig2:arg", "%s must be real double", what);
    return mxDuplicateArray(a);
}
static mxArray *square(const mxArray *a, I *n, const char *what) {
    if (mxGetM(a) != mxGetN(a)) mexErrMsgIdAndTxt("kpeig2:arg", "%s must be square", what);
    *n = (I)mxGetM(a);
    return dup(a, what);
}
static mxArray *col(I n, const double *src) {
    mxArray *o = mxCreateDoubleMatrix((mwSize)(n > 0 ? n : 0), 1, mxREAL);
    if (n > 0 && src) memcpy(mxGetPr(o), src, (size_t)n * sizeof(double));
    return o;
}
static void chk(I info, const char *who) {
    if (info != 0) mexErrMsgIdAndTxt("kpeig2:info", "%s returned info = %lld", who, (long long)info);
}
static int is(const char *a, const char *b) { return !strcmp(a, b); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    char mode[16], uplo[4] = "U", trans[4] = "N";
    I n, info = 0;
    if (nrhs < 1) mexErrMsgIdAndTxt("kpeig2:arg", "kpeig2(mode, ...)");
    str(prhs[0], mode, sizeof mode, "mode");
#define NEED(k) do { if (nrhs < (k)) mexErrMsgIdAndTxt("kpeig2:arg", "%s needs %d arguments", mode, (k) - 1); } while (0)

    if (is(mode, "ilaenv")) {
        NEED(8);
        char name[16], opts[8];
        str(prhs[2], name, sizeof name, "name");
        str(prhs[3], opts, sizeof opts, "opts");
        I ispec = (I)num(prhs[1], "ispec"), n1 = (I)num(prhs[4], "n1"), n2 = (I)num(prhs[5], "n2"),
          n3 = (I)num(prhs[6], "n3"), n4 = (I)num(prhs[7], "n4");
        I v = ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4, strlen(name), strlen(opts));
        plhs[0] = mxCreateDoubleScalar((double)v);
        return;
    }
    if (is(mode, "sytd2")) {
        NEED(2);
        if (nrhs >= 3) str(prhs[2], uplo, sizeof uplo, "uplo");
        mxArray *A = square(prhs[1], &n, "A");
        I lda = n > 0 ? n : 1;
        double *d = mxCalloc(lda, sizeof(double)), *e = mxCalloc(lda, sizeof(double)), *tau = mxCalloc(lda, sizeof(double));
        dsytd2_(uplo, &n, mxGetPr(A), &lda, d, e, tau, &info); chk(info, "dsytd2");
        plhs[0] = col(n, d);
        if (nlhs > 1) plhs[1] = col(n - 1, e);
        if (nlhs > 2) plhs[2] = col(n - 1, tau);
        if (nlhs > 3) plhs[3] = A; else mxDestroyArray(A);
        return;
    }
    if (is(mode, "latrd")) {
        NEED(3);
        if (nrhs >= 4) str(prhs[3], uplo, sizeof uplo, "uplo");
        mxArray *A = square(prhs[1], &n, "A");
        I nb = (I)num(prhs[2], "nb"), lda = n > 0 ? n : 1;
        if (nb < 1 || nb > n) mexErrMsgIdAndTxt("kpeig2:arg", "nb must be in 1..n");
        double *e = mxCalloc(lda, sizeof(double)), *tau = mxCalloc(lda, sizeof(double));
        mxArray *W = mxCreateDoubleMatrix((mwSize)n, (mwSize)nb, mxREAL);
        dlatrd_(uplo, &n, &nb, mxGetPr(A), &lda, e, tau, mxGetPr(W), &lda);
        plhs[0] = A;
        if (nlhs > 1) plhs[1] = col(n - 1, e);
        if (nlhs > 2) plhs[2] = col(n - 1, tau);
        if (nlhs > 3) plhs[3] = W; else mxDestroyArray(W);
        return;
    }
    if (is(mode, "ormtr")) {
        NEED(4);
        if (nrhs >= 5) str(prhs[4], uplo, sizeof uplo, "uplo");
        I m, lwork = -1; double wq;
        mxArray *Ar = square(prhs[1], &m, "Ar");
        mxArray *C = dup(prhs[3], "C");
        I nc = (I)mxGetN(C), lda = m > 0 ? m : 1;
        if ((I)mxGetM(C) != m) mexErrMsgIdAndTxt("kpeig2:arg", "C must have size(Ar,1) rows");
        dormtr_("L", uplo, "N", &m, &nc, mxGetPr(Ar), &lda, mxGetPr(prhs[2]), mxGetPr(C), &lda, &wq, &lwork, &info); chk(info, "dormtr query");
        lwork = (I)wq; double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
        dormtr_("L", uplo, "N", &m, &nc, mxGetPr(Ar), &lda, mxGetPr(prhs[2]), mxGetPr(C), &lda, work, &lwork, &info); chk(info, "dormtr");
        mxDestroyArray(Ar);
        plhs[0] = C;
        return;
    }
    if (is(mode, "symv")) {
        NEED(4);
        if (nrhs >= 5) str(prhs[4], uplo, sizeof uplo, "uplo");
        double alpha = num(prhs[1], "alpha"), beta = 0.0;
        mxArray *A = square(prhs[2], &n, "A");
        if ((I)mxGetNumberOfElements(prhs[3]) != n) mexErrMsgIdAndTxt("kpeig2:arg", "x must have n entries");
        I lda = n > 0 ? n : 1;
        mxArray *y = mxCreateDoubleMatrix((mwSize)n, 1, mxREAL);
        dsymv_(uplo, &n, &alpha, mxGetPr(A), &lda, mxGetPr(prhs[3]), &ONE, &beta, mxGetPr(y), &ONE);
        mxDestroyArray(A);
        plhs[0] = y;
        return;
    }
    if (is(mode, "syr2")) {
        NEED(5);
        if (nrhs >= 6) str(prhs[5], uplo, sizeof uplo, "uplo");
        double alpha = num(prhs[1], "alpha");
        mxArray *A = square(prhs[4], &n, "A");
        if ((I)mxGetNumberOfElements(prhs[2]) != n || (I)mxGetNumberOfElements(prhs[3]) != n)
            mexErrMsgIdAndTxt("kpeig2:arg", "x and y must have n entries");
        I lda = n > 0 ? n : 1;
        dsyr2_(uplo, &n, &alpha, mxGetPr(prhs[2]), &ONE, mxGetPr(prhs[3]), &ONE, mxGetPr(A), &lda);
        plhs[0] = A;
        return;
    }
    if (is(mode, "syr2k")) {
        NEED(6);
        if (nrhs >= 7) str(prhs[6], uplo, sizeof uplo, "uplo");
        if (nrhs >= 8) str(prhs[7], trans, sizeof trans, "trans");
        double alpha = num(prhs[1], "alpha"), beta = num(prhs[4], "beta");
        mxArray *C = square(prhs[5], &n, "C");
        I k = (trans[0] == 'N') ? (I)mxGetN(prhs[2]) : (I)mxGetM(prhs[2]);
        I lda = (I)mxGetM(prhs[2]), ldb = (I)mxGetM(prhs[3]), ldc = n > 0 ? n : 1;
        if (lda < 1) lda = 1;
        if (ldb < 1) ldb = 1;
        dsyr2k_(uplo, trans, &n, &k, &alpha, mxGetPr(prhs[2]), &lda, mxGetPr(prhs[3]), &ldb, &beta, mxGetPr(C), &ldc);
        plhs[0] = C;
        return;
    }
    if (is(mode, "gemv")) {
        NEED(7);
        str(prhs[1], trans, sizeof trans, "trans");
        double alpha = num(prhs[2], "alpha"), beta = num(prhs[5], "beta");
        I m = (I)mxGetM(prhs[3]), nn = (I)mxGetN(prhs[3]), lda = m > 0 ? m : 1;
        mxArray *y = dup(prhs[6], "y");
        dgemv_(trans, &m, &nn, &alpha, mxGetPr(prhs[3]), &lda, mxGetPr(prhs[4]), &ONE, &beta, mxGetPr(y), &ONE);
        plhs[0] = y;
        return;
    }
    if (is(mode, "dot")) {
        NEED(3);
        n = (I)mxGetNumberOfElements(prhs[1]);
        if ((I)mxGetNumberOfElements(prhs[2]) != n) mexErrMsgIdAndTxt("kpeig2:arg", "x and y must match");
        plhs[0] = mxCreateDoubleScalar(ddot_(&n, mxGetPr(prhs[1]), &ONE, mxGetPr(prhs[2]), &ONE));
        return;
    }
    if (is(mode, "nrm2")) {
        NEED(2);
        n = (I)mxGetNumberOfElements(prhs[1]);
        plhs[0] = mxCreateDoubleScalar(dnrm2_(&n, mxGetPr(prhs[1]), &ONE));
        return;
    }
    if (is(mode, "axpy")) {
        NEED(4);
        double alpha = num(prhs[1], "alpha");
        n = (I)mxGetNumberOfElements(prhs[2]);
        if ((I)mxGetNumberOfElements(prhs[3]) != n) mexErrMsgIdAndTxt("kpeig2:arg", "x and y must match");
        mxArray *y = dup(prhs[3], "y");
        daxpy_(&n, &alpha, mxGetPr(prhs[2]), &ONE, mxGetPr(y), &ONE);
        plhs[0] = y;
        return;
    }
    if (is(mode, "larfg")) {
        NEED(3);
        double alpha = num(prhs[1], "alpha"), tau = 0.0;
        mxArray *x = dup(prhs[2], "x");
        I m = (I)mxGetNumberOfElements(x) + 1;
        dlarfg_(&m, &alpha, mxGetPr(x), &ONE, &tau);
        plhs[0] = mxCreateDoubleScalar(alpha);
        if (nlhs > 1) plhs[1] = x; else mxDestroyArray(x);
        if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(tau);
        return;
    }
    if (is(mode, "fprev_symv") || is(mode, "fprev_dot")) {
        NEED(2);
        n = (I)num(prhs[1], "n");
        if (n < 2 || n > 400) mexErrMsgIdAndTxt("kpeig2:arg", "fprev needs 2 <= n <= 400");
        const double M = ldexp(1.0, 70);
        I pairs = n * (n - 1) / 2, nrows = 1;
        const double *rows = NULL;
        if (is(mode, "fprev_symv")) {
            NEED(3);
            if (nrhs >= 4) str(prhs[3], uplo, sizeof uplo, "uplo");
            rows = mxGetPr(prhs[2]);
            nrows = (I)mxGetNumberOfElements(prhs[2]);
            for (I r = 0; r < nrows; r++)
                if (rows[r] < 1 || rows[r] > n) mexErrMsgIdAndTxt("kpeig2:arg", "rows must be in 1..n");
        }
        mxArray *R = mxCreateDoubleMatrix((mwSize)pairs, (mwSize)nrows, mxREAL);
        double *res = mxGetPr(R);
        double *A = mxMalloc((size_t)(n * n) * sizeof(double)), *x = mxMalloc((size_t)n * sizeof(double)),
               *ones = mxMalloc((size_t)n * sizeof(double)), *y = mxMalloc((size_t)n * sizeof(double));
        for (I i = 0; i < n * n; i++) A[i] = 1.0;
        for (I i = 0; i < n; i++) { x[i] = 1.0; ones[i] = 1.0; }
        double alpha = 1.0, beta = 0.0;
        I t = 0;
        for (I p = 0; p < n; p++) {
            for (I q = p + 1; q < n; q++, t++) {
                x[p] = M; x[q] = -M;
                if (rows) {
                    for (I i = 0; i < n; i++) y[i] = 0.0;
                    dsymv_(uplo, &n, &alpha, A, &n, x, &ONE, &beta, y, &ONE);
                    for (I r = 0; r < nrows; r++) res[t + r * pairs] = y[(I)rows[r] - 1];
                } else {
                    res[t] = ddot_(&n, x, &ONE, ones, &ONE);
                }
                x[p] = 1.0; x[q] = 1.0;
            }
        }
        mxFree(A); mxFree(x); mxFree(ones); mxFree(y);
        plhs[0] = R;
        return;
    }
    mexErrMsgIdAndTxt("kpeig2:mode", "unknown mode %s", mode);
}
