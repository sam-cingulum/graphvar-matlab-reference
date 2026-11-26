/* kpeig3: eig probe 3 (blocked dsytrd, dormtr's blocked back-transform, dstedc's dgemm).
 * Calls MATLAB's own LAPACK/BLAS (libmwlapack/libmwblas, MKL, ILP64). Build:
 *   mex -R2018a kpeig3.c -lmwlapack -lmwblas
 *
 * Composition checks (reference LAPACK 3.12.1 control flow, every kernel call goes to MKL):
 *   [d, e, tau, Ar, snapA, snapW] = kpeig3('sytrd_composed', A, wantSnaps)
 *       DSYTRD('U') blocked loop: DLATRD + DSYR2K('U','N') panels, then DSYTD2 on the leading kk.
 *       nb, nx from MKL ilaenv. snapA(:,:,2p-1) = A after panel p's DLATRD, snapA(:,:,2p) = A after
 *       its DSYR2K (before the e restore); snapW(:,:,p) = that panel's W (n x nb).
 *   [Ar, e, tau, W] = kpeig3('latrd_composed', A, nb, variant)   DLATRD('U') from MKL dgemv/dlarfg/dsymv/dscal/ddot/daxpy
 *       (variant 0 = reference; 1 = first column's DSYMV with alpha = tau, no DSCAL)
 *   [C, Ts]  = kpeig3('ormtr_composed', Ar, tau, C, nb, v2l)  DORMTR('L','U','N') -> DORMQL('L','N') with MKL
 *       (nb > 0 overrides ilaenv; v2l = -1 MKL dorm2l, 0/1 composed DLARF/DLARF1L for the unblocked path;
 *       cfb = 1 uses the composed reference DLARFB). [C, Ts, Cs]: Cs(:,:,b) = C after block b.
 *       dlarft + dlarfb per block (or dorm2l when MKL's nb says unblocked). Ts = the T of each block,
 *       stacked (nb x nb x nblocks, zero padded).
 *   C = kpeig3('larfb_composed', V, T, C, k)            reference DLARFB('L','N','B','C') from MKL
 *       dcopy/dtrmm/dgemm; V is m x k (V2 = its last k rows, unit upper), T k x k lower.
 *   C = kpeig3('larfb', V, T, C, k)                     MKL dlarfb('L','N','B','C') itself
 *   T = kpeig3('larft', V, tau)                         MKL dlarft('B','C')
 *   C = kpeig3('orm2l', V, tau, C)                      MKL dorm2l('L','N')
 *   C = kpeig3('orm2l_composed', V, tau, C, variant)    reference DORM2L with MKL kernels: variant 0 = DLARF
 *                                                       (LAPACK <= 3.12.0), 1 = DLARF1L (3.12.1)
 *   C = kpeig3('larf', v, tau, C)                       MKL dlarf('L')
 *   C = kpeig3('larf_composed', v, tau, C)              reference DLARF('L') (lastv/lastc scan) from MKL dgemv + dger
 *   [d, Z] = kpeig3('steqr', d, e)                      MKL dsteqr('I')
 * Kernels (MKL as called):
 *   C = kpeig3('gemm', ta, tb, alpha, A, B, beta, C)    lda/ldb/ldc = the matrices' row counts
 *   B = kpeig3('trmm', side, uplo, transa, diag, alpha, A, B)
 *   C = kpeig3('syr2k', alpha, A, B, beta, C)           uplo 'U', trans 'N'
 *   y = kpeig3('gemv', trans, alpha, A, x, beta, y, incx)   x read with stride incx (x holds incx*len words)
 *   A = kpeig3('syr2', alpha, x, y, A)                  uplo 'U'
 *   y = kpeig3('symv', alpha, A, x)                     uplo 'U', beta 0
 *   s = kpeig3('dot', x, y)
 *   A = kpeig3('ger', alpha, x, y, A)
 *   x = kpeig3('scal', alpha, x)
 * FPRev summation trees (M = 2^70; each value = summands minus the size of the smallest subtree
 * holding both masked summands; pairs p < q, p ascending then q ascending):
 *   R = kpeig3('fprev_gemm', ta, tb, m, n, L, rows, cols)   C = op(A) op(B), A = ones, B's mask along L
 *                                                           (same mask in every column), beta = 0
 *   R = kpeig3('fprev_gemv', trans, m, n, rows)             trans 'T': summands = the m rows of x, all n outputs.
 *                                                           trans 'N': summands = y0 (index 1) and the n columns.
 *   R = kpeig3('fprev_trmm', uplo, transa, diag, m, k, rows) side 'R': B(i,j) = sum_l B(i,l) op(A)(l,j)
 *   R = kpeig3('fprev_syr2k', n, k, d)                      entries (i, i+d) for i with floor((i-1)/d) even;
 *       summand 1 = C(i,j), 2..k+1 = A(i,l) B(j,l), k+2..2k+1 = B(i,l) A(j,l). R(pair, entry).
 */
#include "mex.h"
#include <stddef.h>
#include <string.h>
#include <math.h>
typedef ptrdiff_t I;
extern I ilaenv_(const I *, const char *, const char *, const I *, const I *, const I *, const I *, size_t, size_t);
extern void dsytd2_(const char *, const I *, double *, const I *, double *, double *, double *, I *);
extern void dlatrd_(const char *, const I *, const I *, double *, const I *, double *, double *, double *, const I *);
extern void dsyr2k_(const char *, const char *, const I *, const I *, const double *, const double *, const I *,
                    const double *, const I *, const double *, double *, const I *);
extern void dgemm_(const char *, const char *, const I *, const I *, const I *, const double *, const double *, const I *,
                   const double *, const I *, const double *, double *, const I *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const I *, const I *, const double *,
                   const double *, const I *, double *, const I *);
extern void dgemv_(const char *, const I *, const I *, const double *, const double *, const I *, const double *, const I *,
                   const double *, double *, const I *);
extern void dsymv_(const char *, const I *, const double *, const double *, const I *, const double *, const I *,
                   const double *, double *, const I *);
extern void dsyr2_(const char *, const I *, const double *, const double *, const I *, const double *, const I *, double *, const I *);
extern void dger_(const I *, const I *, const double *, const double *, const I *, const double *, const I *, double *, const I *);
extern double ddot_(const I *, const double *, const I *, const double *, const I *);
extern void daxpy_(const I *, const double *, const double *, const I *, double *, const I *);
extern void dscal_(const I *, const double *, double *, const I *);
extern void dcopy_(const I *, const double *, const I *, double *, const I *);
extern void dlarfg_(const I *, double *, double *, const I *, double *);
extern void dlarft_(const char *, const char *, const I *, const I *, const double *, const I *, const double *, double *, const I *);
extern void dlarfb_(const char *, const char *, const char *, const char *, const I *, const I *, const I *,
                    const double *, const I *, const double *, const I *, double *, const I *, double *, const I *);
extern void dlarf_(const char *, const I *, const I *, const double *, const I *, const double *, double *, const I *, double *);
extern void dorm2l_(const char *, const char *, const I *, const I *, const I *, double *, const I *, const double *,
                    double *, const I *, double *, I *);
extern void dsteqr_(const char *, const I *, double *, double *, double *, const I *, double *, I *);

static const I ONE = 1;
static const double D1 = 1.0, DM1 = -1.0, D0 = 0.0;

static void str(const mxArray *a, char *buf, size_t n, const char *what) {
    if (!mxIsChar(a) || mxGetString(a, buf, (mwSize)n)) mexErrMsgIdAndTxt("kpeig3:arg", "%s must be a short char", what);
}
static double num(const mxArray *a, const char *what) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1) mexErrMsgIdAndTxt("kpeig3:arg", "%s must be a real scalar", what);
    return mxGetScalar(a);
}
static mxArray *dup(const mxArray *a, const char *what) {
    if (!mxIsDouble(a) || mxIsComplex(a)) mexErrMsgIdAndTxt("kpeig3:arg", "%s must be real double", what);
    return mxDuplicateArray(a);
}
static I rowsof(const mxArray *a) { return (I)mxGetM(a); }
static I colsof(const mxArray *a) { return (I)mxGetN(a); }
static I ldof(const mxArray *a) { return mxGetM(a) > 0 ? (I)mxGetM(a) : 1; }
static mxArray *col(I n, const double *src) {
    mxArray *o = mxCreateDoubleMatrix((mwSize)(n > 0 ? n : 0), 1, mxREAL);
    if (n > 0 && src) memcpy(mxGetPr(o), src, (size_t)n * sizeof(double));
    return o;
}
static void chk(I info, const char *who) {
    if (info != 0) mexErrMsgIdAndTxt("kpeig3:info", "%s returned info = %lld", who, (long long)info);
}
static int is(const char *a, const char *b) { return !strcmp(a, b); }
static I imax(I a, I b) { return a > b ? a : b; }
static I imin(I a, I b) { return a < b ? a : b; }
static I ilaenv4(I ispec, const char *name, const char *opts, I n1, I n2, I n3, I n4) {
    return ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4, strlen(name), strlen(opts));
}

static I iladlc(I m, I n, const double *C, I ldc);

/* reference DLATRD('U') with MKL kernels; A n x n (lda), W n x nb (ldw) */
/* variant 0: reference order (DSYMV alpha 1, then DSCAL by tau); variant 1: DSYMV with alpha = tau and no
   DSCAL when i = n (the dgemv corrections, when present, are scaled by DSCAL as in variant 0 otherwise) */
static void latrd_composed(I n, I nb, double *A, I lda, double *e, double *tau, double *W, I ldw, int variant) {
#define a(i, j) A[((i) - 1) + ((j) - 1) * lda]
#define w(i, j) W[((i) - 1) + ((j) - 1) * ldw]
    for (I i = n; i >= n - nb + 1; i--) {
        I iw = i - n + nb;
        if (i < n) {
            I m = i, k = n - i;
            dgemv_("N", &m, &k, &DM1, &a(1, i + 1), &lda, &w(i, iw + 1), &ldw, &D1, &a(1, i), &ONE);
            dgemv_("N", &m, &k, &DM1, &w(1, iw + 1), &ldw, &a(i, i + 1), &lda, &D1, &a(1, i), &ONE);
        }
        if (i > 1) {
            I im1 = i - 1, k = n - i;
            dlarfg_(&im1, &a(i - 1, i), &a(1, i), &ONE, &tau[i - 2]);
            e[i - 2] = a(i - 1, i);
            a(i - 1, i) = 1.0;
            int fold = variant == 1 && i == n;
            dsymv_("U", &im1, fold ? &tau[i - 2] : &D1, A, &lda, &a(1, i), &ONE, &D0, &w(1, iw), &ONE);
            if (i < n) {
                dgemv_("T", &im1, &k, &D1, &w(1, iw + 1), &ldw, &a(1, i), &ONE, &D0, &w(i + 1, iw), &ONE);
                dgemv_("N", &im1, &k, &DM1, &a(1, i + 1), &lda, &w(i + 1, iw), &ONE, &D1, &w(1, iw), &ONE);
                dgemv_("T", &im1, &k, &D1, &a(1, i + 1), &lda, &a(1, i), &ONE, &D0, &w(i + 1, iw), &ONE);
                dgemv_("N", &im1, &k, &DM1, &w(1, iw + 1), &ldw, &w(i + 1, iw), &ONE, &D1, &w(1, iw), &ONE);
            }
            if (!fold) dscal_(&im1, &tau[i - 2], &w(1, iw), &ONE);
            double alpha = -0.5 * tau[i - 2] * ddot_(&im1, &w(1, iw), &ONE, &a(1, i), &ONE);
            daxpy_(&im1, &alpha, &a(1, i), &ONE, &w(1, iw), &ONE);
        }
    }
#undef a
#undef w
}

/* reference DLARFB('L','N','B','C'): C (m x n) := H C with V m x k (V2 = rows m-k+1..m), T k x k */
static void larfb_composed(I m, I n, I k, const double *V, I ldv, const double *T, I ldt, double *C, I ldc, double *Wk, I ldwork) {
    if (m <= 0 || n <= 0) return;
    for (I j = 1; j <= k; j++) dcopy_(&n, &C[(m - k + j) - 1], &ldc, &Wk[(j - 1) * ldwork], &ONE);
    dtrmm_("R", "U", "N", "U", &n, &k, &D1, &V[m - k], &ldv, Wk, &ldwork);
    if (m > k) {
        I mk = m - k;
        dgemm_("T", "N", &n, &k, &mk, &D1, C, &ldc, V, &ldv, &D1, Wk, &ldwork);
    }
    dtrmm_("R", "L", "T", "N", &n, &k, &D1, T, &ldt, Wk, &ldwork);
    if (m > k) {
        I mk = m - k;
        dgemm_("N", "T", &mk, &n, &k, &DM1, V, &ldv, Wk, &ldwork, &D1, C, &ldc);
    }
    dtrmm_("R", "U", "T", "U", &n, &k, &D1, &V[m - k], &ldv, Wk, &ldwork);
    for (I j = 1; j <= k; j++)
        for (I i = 1; i <= n; i++) C[(m - k + j - 1) + (i - 1) * ldc] -= Wk[(i - 1) + (j - 1) * ldwork];
}

/* reference DLARF('L') with lastv/lastc scan (ILADLR/ILADLC logic), MKL dgemv + dger */
static void larf_composed(I m, I n, const double *v, double tau, double *C, I ldc, double *work) {
    I lastv = 0, lastc = 0;
    if (tau != 0.0) {
        lastv = m;
        while (lastv > 0 && v[lastv - 1] == 0.0) lastv--;
        lastc = iladlc(lastv, n, C, ldc);
    }
    if (lastv > 0) {
        dgemv_("T", &lastv, &lastc, &D1, C, &ldc, v, &ONE, &D0, work, &ONE);
        double mt = -tau;
        dger_(&lastv, &lastc, &mt, v, &ONE, work, &ONE, C, &ldc);
    }
}

/* ILADLC(m, n, C): last column of C(1:m, 1:n) with a non-zero entry (0 if none) */
static I iladlc(I m, I n, const double *C, I ldc) {
    if (n == 0) return 0;
    if (m > 0 && (C[(n - 1) * ldc] != 0.0 || C[(m - 1) + (n - 1) * ldc] != 0.0)) return n;
    for (I j = n; j >= 1; j--)
        for (I i = 1; i <= m; i++)
            if (C[(i - 1) + (j - 1) * ldc] != 0.0) return j;
    return 0;
}

/* reference DLARF1L('L') (LAPACK 3.12.1; v(m) is an implicit 1, leading zeros of v skipped), MKL kernels */
static void larf1l_composed(I m, I n, const double *v, double tau, double *C, I ldc, double *work) {
    I firstv = 1, lastv = m, lastc = 0, i = 1;
    if (tau != 0.0) {
        while (lastv > firstv && v[i - 1] == 0.0) { firstv++; i++; }
        lastc = iladlc(lastv, n, C, ldc);
    }
    if (lastc == 0 || lastv <= 0) return;
    if (lastv == firstv) {
        double s = 1.0 - tau;
        dscal_(&lastc, &s, &C[firstv - 1], &ldc);
        return;
    }
    I len = lastv - firstv;
    dgemv_("T", &len, &lastc, &D1, &C[firstv - 1], &ldc, &v[i - 1], &ONE, &D0, work, &ONE);
    daxpy_(&lastc, &D1, &C[lastv - 1], &ldc, work, &ONE);
    double mt = -tau;
    daxpy_(&lastc, &mt, work, &ONE, &C[lastv - 1], &ldc);
    dger_(&len, &lastc, &mt, &v[i - 1], &ONE, work, &ONE, &C[firstv - 1], &ldc);
}

static mxArray *fprev_out(I pairs, I npos) { return mxCreateDoubleMatrix((mwSize)pairs, (mwSize)npos, mxREAL); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    char mode[32];
    if (nrhs < 1) mexErrMsgIdAndTxt("kpeig3:arg", "mode required");
    str(prhs[0], mode, sizeof mode, "mode");
#define NEED(k) do { if (nrhs < (k)) mexErrMsgIdAndTxt("kpeig3:arg", "%s needs %d arguments", mode, (k) - 1); } while (0)
    I info = 0;
    const double M = ldexp(1.0, 70);

    if (is(mode, "sytrd_composed")) {
        NEED(2);
        int snaps = nrhs >= 3 && num(prhs[2], "wantSnaps") != 0;
        mxArray *Am = dup(prhs[1], "A");
        I n = rowsof(Am), lda = imax(1, n);
        if (colsof(Am) != n) mexErrMsgIdAndTxt("kpeig3:arg", "A must be square");
        double *A = mxGetPr(Am);
        double *d = mxCalloc(imax(1, n), sizeof(double)), *e = mxCalloc(imax(1, n), sizeof(double)), *tau = mxCalloc(imax(1, n), sizeof(double));
        I nx = n, nb = ilaenv4(1, "DSYTRD", "U", n, -1, -1, -1);
        if (nb > 1 && nb < n) {
            nx = imax(nb, ilaenv4(3, "DSYTRD", "U", n, -1, -1, -1));
            if (nx >= n) nb = 1;
        } else nb = 1;
        if (nb == 1) nx = n;
        I kk = n - ((n - nx + nb - 1) / nb) * nb;
        I npan = 0;
        for (I i = n - nb + 1; i >= kk + 1; i -= nb) npan++;
        I ldw = lda;
        double *W = mxCalloc((size_t)ldw * (size_t)imax(1, nb), sizeof(double));
        mxArray *SA = NULL, *SW = NULL;
        if (snaps && npan > 0) {
            mwSize dA[3] = {(mwSize)n, (mwSize)n, (mwSize)(2 * npan)}, dW[3] = {(mwSize)n, (mwSize)nb, (mwSize)npan};
            SA = mxCreateNumericArray(3, dA, mxDOUBLE_CLASS, mxREAL);
            SW = mxCreateNumericArray(3, dW, mxDOUBLE_CLASS, mxREAL);
        }
        I p = 0;
        for (I i = n - nb + 1; i >= kk + 1; i -= nb, p++) {
            I nn = i + nb - 1, im1 = i - 1;
            memset(W, 0, (size_t)ldw * (size_t)nb * sizeof(double));
            dlatrd_("U", &nn, &nb, A, &lda, e, tau, W, &ldw);
            if (SA) {
                memcpy(mxGetPr(SA) + (size_t)(2 * p) * n * n, A, (size_t)n * n * sizeof(double));
                memcpy(mxGetPr(SW) + (size_t)p * n * nb, W, (size_t)n * nb * sizeof(double));
            }
            dsyr2k_("U", "N", &im1, &nb, &DM1, &A[(i - 1) * lda], &lda, W, &ldw, &D1, A, &lda);
            if (SA) memcpy(mxGetPr(SA) + (size_t)(2 * p + 1) * n * n, A, (size_t)n * n * sizeof(double));
            for (I j = i; j <= i + nb - 1; j++) {
                A[(j - 2) + (j - 1) * lda] = e[j - 2];
                d[j - 1] = A[(j - 1) + (j - 1) * lda];
            }
        }
        dsytd2_("U", &kk, A, &lda, d, e, tau, &info); chk(info, "dsytd2");
        plhs[0] = col(n, d);
        if (nlhs > 1) plhs[1] = col(n - 1, e);
        if (nlhs > 2) plhs[2] = col(n - 1, tau);
        if (nlhs > 3) plhs[3] = Am; else mxDestroyArray(Am);
        if (nlhs > 4) plhs[4] = SA ? SA : mxCreateDoubleMatrix(0, 0, mxREAL);
        if (nlhs > 5) plhs[5] = SW ? SW : mxCreateDoubleMatrix(0, 0, mxREAL);
        mxFree(d); mxFree(e); mxFree(tau); mxFree(W);
        return;
    }
    if (is(mode, "latrd_composed")) {
        NEED(3);
        mxArray *Am = dup(prhs[1], "A");
        I n = rowsof(Am), lda = imax(1, n), nb = (I)num(prhs[2], "nb");
        double *e = mxCalloc(imax(1, n), sizeof(double)), *tau = mxCalloc(imax(1, n), sizeof(double));
        mxArray *Wm = mxCreateDoubleMatrix((mwSize)n, (mwSize)nb, mxREAL);
        latrd_composed(n, nb, mxGetPr(Am), lda, e, tau, mxGetPr(Wm), lda, nrhs >= 4 ? (int)num(prhs[3], "variant") : 0);
        plhs[0] = Am;
        if (nlhs > 1) plhs[1] = col(n - 1, e);
        if (nlhs > 2) plhs[2] = col(n - 1, tau);
        if (nlhs > 3) plhs[3] = Wm; else mxDestroyArray(Wm);
        mxFree(e); mxFree(tau);
        return;
    }
    if (is(mode, "ormtr_composed")) {
        NEED(4);
        const mxArray *Am = prhs[1];
        I n = rowsof(Am), lda = imax(1, n);
        mxArray *Cm = dup(prhs[3], "C");
        I nc = colsof(Cm), ldc = ldof(Cm);
        const double *A = mxGetPr(Am), *tau = mxGetPr(prhs[2]);
        double *C = mxGetPr(Cm);
        /* DORMTR('L','U','N'): DORMQL('L','N', m = n-1, nc, k = n-1, A(1,2), lda, tau, C, ldc) */
        I m = n - 1, k = n - 1, nq = m;
        char opts[3] = "LN";
        I nb = imin(64, ilaenv4(1, "DORMQL", opts, m, nc, k, -1));
        if (nrhs >= 5 && num(prhs[4], "nb") > 0) nb = (I)num(prhs[4], "nb");   /* override, for tests */
        int v2l = nrhs >= 6 ? (int)num(prhs[5], "orm2l variant") : -1;     /* -1: MKL dorm2l */
        int cfb = nrhs >= 7 && num(prhs[6], "composed larfb") != 0;         /* reference DLARFB from MKL kernels */
        mxArray *Cs = NULL;
        I nbmin = 2;
        mxArray *Ts = NULL;
        if (m > 0 && nc > 0 && k > 0) {
            double *Aq = mxMalloc((size_t)lda * (size_t)imax(1, n) * sizeof(double));
            memcpy(Aq, A, (size_t)lda * n * sizeof(double));
            double *V = Aq + lda; /* A(1,2) */
            if (nb < nbmin || nb >= k) {
                double *work = mxCalloc((size_t)imax(1, nc), sizeof(double));
                if (v2l < 0) { dorm2l_("L", "N", &m, &nc, &k, V, &lda, tau, C, &ldc, work, &info); chk(info, "dorm2l"); }
                else for (I i = 1; i <= k; i++) {
                    I mi = m - k + i;
                    double *vi = &V[(i - 1) * lda], aii = vi[mi - 1];
                    if (v2l == 1) larf1l_composed(mi, nc, vi, tau[i - 1], C, ldc, work);
                    else { vi[mi - 1] = 1.0; larf_composed(mi, nc, vi, tau[i - 1], C, ldc, work); vi[mi - 1] = aii; }
                }
                mxFree(work);
            } else {
                I nblk = (k + nb - 1) / nb, ldt = nb, ldwork = imax(1, nc);
                mwSize dT[3] = {(mwSize)nb, (mwSize)nb, (mwSize)nblk};
                Ts = mxCreateNumericArray(3, dT, mxDOUBLE_CLASS, mxREAL);
                if (nlhs > 2) { mwSize dC[3] = {(mwSize)ldc, (mwSize)nc, (mwSize)nblk}; Cs = mxCreateNumericArray(3, dC, mxDOUBLE_CLASS, mxREAL); }
                double *T = mxCalloc((size_t)ldt * nb, sizeof(double)), *Wk = mxCalloc((size_t)ldwork * nb, sizeof(double));
                I b = 0;
                for (I i = 1; i <= k; i += nb, b++) {
                    I ib = imin(nb, k - i + 1), nv = nq - k + i + ib - 1, mi = m - k + i + ib - 1;
                    memset(T, 0, (size_t)ldt * nb * sizeof(double));
                    dlarft_("B", "C", &nv, &ib, &V[(i - 1) * lda], &lda, &tau[i - 1], T, &ldt);
                    for (I jj = 0; jj < ib; jj++)
                        for (I ii = 0; ii < ib; ii++) mxGetPr(Ts)[(size_t)b * nb * nb + ii + jj * nb] = T[ii + jj * ldt];
                    if (cfb) larfb_composed(mi, nc, ib, &V[(i - 1) * lda], lda, T, ldt, C, ldc, Wk, ldwork);
                    else dlarfb_("L", "N", "B", "C", &mi, &nc, &ib, &V[(i - 1) * lda], &lda, T, &ldt, C, &ldc, Wk, &ldwork);
                    if (Cs) memcpy(mxGetPr(Cs) + (size_t)b * ldc * nc, C, (size_t)ldc * nc * sizeof(double));
                }
                mxFree(T); mxFree(Wk);
            }
            mxFree(Aq);
        }
        plhs[0] = Cm;
        if (nlhs > 1) plhs[1] = Ts ? Ts : mxCreateDoubleMatrix(0, 0, mxREAL);
        if (nlhs > 2) plhs[2] = Cs ? Cs : mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
    }
    if (is(mode, "larfb_composed") || is(mode, "larfb")) {
        NEED(5);
        const mxArray *Vm = prhs[1], *Tm = prhs[2];
        mxArray *Cm = dup(prhs[3], "C");
        I k = (I)num(prhs[4], "k"), m = rowsof(Cm), n = colsof(Cm), ldv = ldof(Vm), ldt = ldof(Tm), ldc = ldof(Cm), ldwork = imax(1, n);
        double *Wk = mxCalloc((size_t)ldwork * imax(1, k), sizeof(double));
        if (is(mode, "larfb")) dlarfb_("L", "N", "B", "C", &m, &n, &k, mxGetPr(Vm), &ldv, mxGetPr(Tm), &ldt, mxGetPr(Cm), &ldc, Wk, &ldwork);
        else larfb_composed(m, n, k, mxGetPr(Vm), ldv, mxGetPr(Tm), ldt, mxGetPr(Cm), ldc, Wk, ldwork);
        mxFree(Wk);
        plhs[0] = Cm;
        return;
    }
    if (is(mode, "larft")) {
        NEED(3);
        const mxArray *Vm = prhs[1];
        I nv = rowsof(Vm), k = colsof(Vm), ldv = ldof(Vm), ldt = imax(1, k);
        mxArray *Tm = mxCreateDoubleMatrix((mwSize)k, (mwSize)k, mxREAL);
        dlarft_("B", "C", &nv, &k, mxGetPr(Vm), &ldv, mxGetPr(prhs[2]), mxGetPr(Tm), &ldt);
        plhs[0] = Tm;
        return;
    }
    if (is(mode, "orm2l") || is(mode, "orm2l_composed")) {
        NEED(4);
        mxArray *Vm = dup(prhs[1], "V"), *Cm = dup(prhs[3], "C");
        I m = rowsof(Cm), nc = colsof(Cm), k = colsof(Vm), ldv = ldof(Vm), ldc = ldof(Cm);
        const double *tau = mxGetPr(prhs[2]);
        double *V = mxGetPr(Vm), *C = mxGetPr(Cm), *work = mxCalloc((size_t)imax(1, nc), sizeof(double));
        if (is(mode, "orm2l")) { dorm2l_("L", "N", &m, &nc, &k, V, &ldv, tau, C, &ldc, work, &info); chk(info, "dorm2l"); }
        else {
            /* reference DORM2L('L','N'): i = 1..k, mi = m-k+i, H(i) applied to C(1:mi, :).
               variant 0: LAPACK <= 3.12.0 (A(mi,i) set to 1, DLARF); variant 1: 3.12.1 (DLARF1L) */
            int variant = nrhs >= 5 ? (int)num(prhs[4], "variant") : 0;
            for (I i = 1; i <= k; i++) {
                I mi = m - k + i;
                double *vi = &V[(i - 1) * ldv], aii = vi[mi - 1];
                if (variant == 1) larf1l_composed(mi, nc, vi, tau[i - 1], C, ldc, work);
                else {
                    vi[mi - 1] = 1.0;
                    larf_composed(mi, nc, vi, tau[i - 1], C, ldc, work);
                    vi[mi - 1] = aii;
                }
            }
        }
        mxFree(work); mxDestroyArray(Vm);
        plhs[0] = Cm;
        return;
    }
    if (is(mode, "larf") || is(mode, "larf_composed")) {
        NEED(4);
        mxArray *Cm = dup(prhs[3], "C");
        I m = rowsof(Cm), n = colsof(Cm), ldc = ldof(Cm);
        double tau = num(prhs[2], "tau"), *work = mxCalloc((size_t)imax(1, n), sizeof(double));
        if (is(mode, "larf")) dlarf_("L", &m, &n, mxGetPr(prhs[1]), &ONE, &tau, mxGetPr(Cm), &ldc, work);
        else larf_composed(m, n, mxGetPr(prhs[1]), tau, mxGetPr(Cm), ldc, work);
        mxFree(work);
        plhs[0] = Cm;
        return;
    }
    if (is(mode, "steqr")) {
        NEED(3);
        mxArray *dm = dup(prhs[1], "d");
        I n = (I)mxGetNumberOfElements(dm), ldz = imax(1, n);
        double *e = mxCalloc((size_t)imax(1, n), sizeof(double));
        if (n > 1) memcpy(e, mxGetPr(prhs[2]), (size_t)(n - 1) * sizeof(double));
        mxArray *Zm = mxCreateDoubleMatrix((mwSize)n, (mwSize)n, mxREAL);
        double *work = mxCalloc((size_t)imax(1, 2 * n - 2), sizeof(double));
        dsteqr_("I", &n, mxGetPr(dm), e, mxGetPr(Zm), &ldz, work, &info); chk(info, "dsteqr");
        mxFree(e); mxFree(work);
        plhs[0] = dm;
        if (nlhs > 1) plhs[1] = Zm; else mxDestroyArray(Zm);
        return;
    }
    if (is(mode, "gemm")) {
        NEED(8);
        char ta[4], tb[4];
        str(prhs[1], ta, sizeof ta, "transa"); str(prhs[2], tb, sizeof tb, "transb");
        double alpha = num(prhs[3], "alpha"), beta = num(prhs[6], "beta");
        const mxArray *Am = prhs[4], *Bm = prhs[5];
        mxArray *Cm = dup(prhs[7], "C");
        I m = rowsof(Cm), n = colsof(Cm), k = (ta[0] == 'N') ? colsof(Am) : rowsof(Am), lda = ldof(Am), ldb = ldof(Bm), ldc = ldof(Cm);
        dgemm_(ta, tb, &m, &n, &k, &alpha, mxGetPr(Am), &lda, mxGetPr(Bm), &ldb, &beta, mxGetPr(Cm), &ldc);
        plhs[0] = Cm;
        return;
    }
    if (is(mode, "trmm")) {
        NEED(8);
        char side[4], uplo[4], ta[4], dg[4];
        str(prhs[1], side, sizeof side, "side"); str(prhs[2], uplo, sizeof uplo, "uplo");
        str(prhs[3], ta, sizeof ta, "transa"); str(prhs[4], dg, sizeof dg, "diag");
        double alpha = num(prhs[5], "alpha");
        mxArray *Bm = dup(prhs[7], "B");
        I m = rowsof(Bm), n = colsof(Bm), lda = ldof(prhs[6]), ldb = ldof(Bm);
        dtrmm_(side, uplo, ta, dg, &m, &n, &alpha, mxGetPr(prhs[6]), &lda, mxGetPr(Bm), &ldb);
        plhs[0] = Bm;
        return;
    }
    if (is(mode, "syr2k")) {
        NEED(6);
        double alpha = num(prhs[1], "alpha"), beta = num(prhs[4], "beta");
        mxArray *Cm = dup(prhs[5], "C");
        I n = rowsof(Cm), k = colsof(prhs[2]), lda = ldof(prhs[2]), ldb = ldof(prhs[3]), ldc = ldof(Cm);
        dsyr2k_("U", "N", &n, &k, &alpha, mxGetPr(prhs[2]), &lda, mxGetPr(prhs[3]), &ldb, &beta, mxGetPr(Cm), &ldc);
        plhs[0] = Cm;
        return;
    }
    if (is(mode, "gemv")) {
        NEED(8);
        char tr[4];
        str(prhs[1], tr, sizeof tr, "trans");
        double alpha = num(prhs[2], "alpha"), beta = num(prhs[5], "beta");
        I incx = (I)num(prhs[7], "incx");
        const mxArray *Am = prhs[3];
        mxArray *ym = dup(prhs[6], "y");
        I m = rowsof(Am), n = colsof(Am), lda = ldof(Am);
        dgemv_(tr, &m, &n, &alpha, mxGetPr(Am), &lda, mxGetPr(prhs[4]), &incx, &beta, mxGetPr(ym), &ONE);
        plhs[0] = ym;
        return;
    }
    if (is(mode, "syr2")) {
        NEED(5);
        double alpha = num(prhs[1], "alpha");
        mxArray *Am = dup(prhs[4], "A");
        I n = rowsof(Am), lda = imax(1, n);
        dsyr2_("U", &n, &alpha, mxGetPr(prhs[2]), &ONE, mxGetPr(prhs[3]), &ONE, mxGetPr(Am), &lda);
        plhs[0] = Am;
        return;
    }
    if (is(mode, "symv")) {
        NEED(4);
        double alpha = num(prhs[1], "alpha");
        I n = rowsof(prhs[2]), lda = imax(1, n);
        mxArray *ym = mxCreateDoubleMatrix((mwSize)n, 1, mxREAL);
        dsymv_("U", &n, &alpha, mxGetPr(prhs[2]), &lda, mxGetPr(prhs[3]), &ONE, &D0, mxGetPr(ym), &ONE);
        plhs[0] = ym;
        return;
    }
    if (is(mode, "dot")) {
        NEED(3);
        I n = (I)mxGetNumberOfElements(prhs[1]);
        plhs[0] = mxCreateDoubleScalar(ddot_(&n, mxGetPr(prhs[1]), &ONE, mxGetPr(prhs[2]), &ONE));
        return;
    }
    if (is(mode, "ger")) {
        NEED(5);
        double alpha = num(prhs[1], "alpha");
        mxArray *Am = dup(prhs[4], "A");
        I m = rowsof(Am), n = colsof(Am), lda = ldof(Am);
        dger_(&m, &n, &alpha, mxGetPr(prhs[2]), &ONE, mxGetPr(prhs[3]), &ONE, mxGetPr(Am), &lda);
        plhs[0] = Am;
        return;
    }
    if (is(mode, "scal")) {
        NEED(3);
        double alpha = num(prhs[1], "alpha");
        mxArray *xm = dup(prhs[2], "x");
        I n = (I)mxGetNumberOfElements(xm);
        dscal_(&n, &alpha, mxGetPr(xm), &ONE);
        plhs[0] = xm;
        return;
    }
    if (is(mode, "fprev_gemm")) {
        NEED(8);
        char ta[4], tb[4];
        str(prhs[1], ta, sizeof ta, "transa"); str(prhs[2], tb, sizeof tb, "transb");
        I m = (I)num(prhs[3], "m"), n = (I)num(prhs[4], "n"), L = (I)num(prhs[5], "L");
        I nr = (I)mxGetNumberOfElements(prhs[6]), ncl = (I)mxGetNumberOfElements(prhs[7]);
        const double *rows = mxGetPr(prhs[6]), *cols = mxGetPr(prhs[7]);
        /* op(A) is m x L, op(B) is L x n */
        I ar = ta[0] == 'N' ? m : L, ac = ta[0] == 'N' ? L : m, br = tb[0] == 'N' ? L : n, bc = tb[0] == 'N' ? n : L;
        I lda = imax(1, ar), ldb = imax(1, br), ldc = imax(1, m);
        double *A = mxMalloc((size_t)lda * imax(1, ac) * sizeof(double)), *B = mxMalloc((size_t)ldb * imax(1, bc) * sizeof(double)),
               *C = mxMalloc((size_t)ldc * imax(1, n) * sizeof(double));
        for (I t = 0; t < lda * ac; t++) A[t] = 1.0;
        for (I t = 0; t < ldb * bc; t++) B[t] = 1.0;
        I pairs = L * (L - 1) / 2, t = 0;
        mxArray *R = fprev_out(pairs, nr * ncl);
        double *res = mxGetPr(R);
#define BSET(l, v) do { for (I j = 0; j < n; j++) { if (tb[0] == 'N') B[(l) + j * ldb] = (v); else B[j + (l) * ldb] = (v); } } while (0)
        for (I p = 0; p < L; p++)
            for (I q = p + 1; q < L; q++, t++) {
                BSET(p, M); BSET(q, -M);
                dgemm_(ta, tb, &m, &n, &L, &D1, A, &lda, B, &ldb, &D0, C, &ldc);
                for (I c = 0; c < ncl; c++)
                    for (I r = 0; r < nr; r++) res[t + (r + c * nr) * pairs] = C[((I)rows[r] - 1) + ((I)cols[c] - 1) * ldc];
                BSET(p, 1.0); BSET(q, 1.0);
            }
#undef BSET
        mxFree(A); mxFree(B); mxFree(C);
        plhs[0] = R;
        return;
    }
    if (is(mode, "fprev_gemv")) {
        NEED(5);
        char tr[4];
        str(prhs[1], tr, sizeof tr, "trans");
        I m = (I)num(prhs[2], "m"), n = (I)num(prhs[3], "n"), lda = imax(1, m);
        I nr = (I)mxGetNumberOfElements(prhs[4]);
        const double *rows = mxGetPr(prhs[4]);
        double *A = mxMalloc((size_t)lda * imax(1, n) * sizeof(double));
        for (I t = 0; t < lda * n; t++) A[t] = 1.0;
        if (tr[0] == 'T') {
            /* y(j) = sum_i A(i,j) x(i): summands = x, m of them; outputs rows = which j */
            double *x = mxMalloc((size_t)imax(1, m) * sizeof(double)), *y = mxMalloc((size_t)imax(1, n) * sizeof(double));
            for (I i = 0; i < m; i++) x[i] = 1.0;
            I pairs = m * (m - 1) / 2, t = 0;
            mxArray *R = fprev_out(pairs, nr);
            for (I p = 0; p < m; p++)
                for (I q = p + 1; q < m; q++, t++) {
                    x[p] = M; x[q] = -M;
                    for (I j = 0; j < n; j++) y[j] = 0.0;
                    dgemv_("T", &m, &n, &D1, A, &lda, x, &ONE, &D0, y, &ONE);
                    for (I r = 0; r < nr; r++) mxGetPr(R)[t + r * pairs] = y[(I)rows[r] - 1];
                    x[p] = 1.0; x[q] = 1.0;
                }
            mxFree(x); mxFree(y);
            plhs[0] = R;
        } else {
            /* y(i) = y0(i) + sum_j A(i,j) x(j), beta = 1: summand 1 = y0, 2..n+1 = the columns */
            double *x = mxMalloc((size_t)imax(1, n) * sizeof(double)), *y = mxMalloc((size_t)imax(1, m) * sizeof(double));
            I s = n + 1, pairs = s * (s - 1) / 2, t = 0;
            mxArray *R = fprev_out(pairs, nr);
            for (I p = 0; p < s; p++)
                for (I q = p + 1; q < s; q++, t++) {
                    for (I j = 0; j < n; j++) x[j] = 1.0;
                    double y0 = 1.0;
                    if (p == 0) y0 = M; else x[p - 1] = M;
                    x[q - 1] = -M;
                    for (I i = 0; i < m; i++) y[i] = y0;
                    dgemv_("N", &m, &n, &D1, A, &lda, x, &ONE, &D1, y, &ONE);
                    for (I r = 0; r < nr; r++) mxGetPr(R)[t + r * pairs] = y[(I)rows[r] - 1];
                }
            mxFree(x); mxFree(y);
            plhs[0] = R;
        }
        mxFree(A);
        return;
    }
    if (is(mode, "fprev_trmm")) {
        NEED(7);
        char uplo[4], ta[4], dg[4];
        str(prhs[1], uplo, sizeof uplo, "uplo"); str(prhs[2], ta, sizeof ta, "transa"); str(prhs[3], dg, sizeof dg, "diag");
        I m = (I)num(prhs[4], "m"), k = (I)num(prhs[5], "k"), ldb = imax(1, m), lda = imax(1, k);
        I nr = (I)mxGetNumberOfElements(prhs[6]);
        const double *rows = mxGetPr(prhs[6]);
        double *A = mxMalloc((size_t)lda * imax(1, k) * sizeof(double)), *B = mxMalloc((size_t)ldb * imax(1, k) * sizeof(double));
        for (I t = 0; t < lda * k; t++) A[t] = 1.0;
        I pairs = k * (k - 1) / 2, t = 0;
        mxArray *R = fprev_out(pairs, nr * k);
        for (I p = 0; p < k; p++)
            for (I q = p + 1; q < k; q++, t++) {
                for (I l = 0; l < k; l++) {
                    double v = l == p ? M : l == q ? -M : 1.0;
                    for (I i = 0; i < m; i++) B[i + l * ldb] = v;
                }
                dtrmm_("R", uplo, ta, dg, &m, &k, &D1, A, &lda, B, &ldb);
                for (I j = 0; j < k; j++)
                    for (I r = 0; r < nr; r++) mxGetPr(R)[t + (r + j * nr) * pairs] = B[((I)rows[r] - 1) + j * ldb];
            }
        mxFree(A); mxFree(B);
        plhs[0] = R;
        return;
    }
    if (is(mode, "fprev_syr2k")) {
        NEED(4);
        I n = (I)num(prhs[1], "n"), k = (I)num(prhs[2], "k"), dd = (I)num(prhs[3], "d"), ld = imax(1, n);
        I ne = 0;
        for (I i = 1; i + dd <= n; i++) if (((i - 1) / dd) % 2 == 0) ne++;
        I *ei = mxMalloc((size_t)imax(1, ne) * sizeof(I)), u = 0;
        for (I i = 1; i + dd <= n; i++) if (((i - 1) / dd) % 2 == 0) ei[u++] = i;
        double *A = mxMalloc((size_t)ld * imax(1, k) * sizeof(double)), *B = mxMalloc((size_t)ld * imax(1, k) * sizeof(double)),
               *C = mxMalloc((size_t)ld * imax(1, n) * sizeof(double));
        I s = 2 * k + 1, pairs = s * (s - 1) / 2, t = 0;
        mxArray *R = fprev_out(pairs, ne);
        for (I p = 0; p < s; p++)
            for (I q = p + 1; q < s; q++, t++) {
                for (I x = 0; x < ld * k; x++) { A[x] = 1.0; B[x] = 1.0; }
                for (I x = 0; x < ld * n; x++) C[x] = 1.0;
                for (I e2 = 0; e2 < ne; e2++) {
                    I i = ei[e2] - 1, j = i + dd;
                    I idx[2] = {p, q};
                    for (int h = 0; h < 2; h++) {
                        double v = h == 0 ? M : -M;
                        I sidx = idx[h];
                        if (sidx == 0) C[i + j * ld] = v;
                        else if (sidx <= k) A[i + (sidx - 1) * ld] = v;   /* A(i,l) B(j,l) */
                        else B[i + (sidx - k - 1) * ld] = v;              /* B(i,l) A(j,l) */
                    }
                }
                dsyr2k_("U", "N", &n, &k, &D1, A, &ld, B, &ld, &D1, C, &ld);
                for (I e2 = 0; e2 < ne; e2++) {
                    I i = ei[e2] - 1, j = i + dd;
                    mxGetPr(R)[t + e2 * pairs] = C[i + j * ld];
                }
            }
        mxFree(A); mxFree(B); mxFree(C); mxFree(ei);
        plhs[0] = R;
        return;
    }
    mexErrMsgIdAndTxt("kpeig3:mode", "unknown mode %s", mode);
}
