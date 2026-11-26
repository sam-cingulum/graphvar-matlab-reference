/* kpeig: call MATLAB's own LAPACK (libmwlapack, ILP64) symmetric eigen
 * routines directly, so a probe can find which one eig() uses and model its
 * stages. Build: mex -R2018a kpeig.c -lmwlapack -lmwblas
 *
 *   [w, V]          = kpeig('syev',  A, jobz, uplo)
 *   [w, V]          = kpeig('syevd', A, jobz, uplo)
 *   [w, V]          = kpeig('syevr', A, jobz, uplo)   range 'A', abstol 0
 *   [w, V]          = kpeig('syevx', A, jobz, uplo)   range 'A', abstol 0
 *   [d, e, tau, Ar] = kpeig('sytrd', A, uplo)
 *   Q               = kpeig('orgtr', Ar, tau, uplo)
 *   [w, Z]          = kpeig('stedc', d, e)            compz 'I'
 *   [w, Z]          = kpeig('steqr', d, e)            compz 'I'
 *   w               = kpeig('sterf', d, e)
 * Every call asks for the optimal workspace first (lwork = -1), as LAPACK
 * callers normally do. info ~= 0 raises an error naming the routine.
 */
#include "mex.h"
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
typedef ptrdiff_t I;
extern void dsyev_(const char *, const char *, const I *, double *, const I *, double *, double *, const I *, I *);
extern void dsyevd_(const char *, const char *, const I *, double *, const I *, double *, double *, const I *, I *, const I *, I *);
extern void dsyevr_(const char *, const char *, const char *, const I *, double *, const I *, const double *, const double *,
                    const I *, const I *, const double *, I *, double *, double *, const I *, I *, double *, const I *, I *, const I *, I *);
extern void dsyevx_(const char *, const char *, const char *, const I *, double *, const I *, const double *, const double *,
                    const I *, const I *, const double *, I *, double *, double *, const I *, double *, const I *, I *, I *, I *);
extern void dsytrd_(const char *, const I *, double *, const I *, double *, double *, double *, double *, const I *, I *);
extern void dorgtr_(const char *, const I *, double *, const I *, const double *, double *, const I *, I *);
extern void dstedc_(const char *, const I *, double *, double *, double *, const I *, double *, const I *, I *, const I *, I *);
extern void dsteqr_(const char *, const I *, double *, double *, double *, const I *, double *, I *);
extern void dsterf_(const I *, double *, double *, I *);

static void str(const mxArray *a, char *buf, size_t n, const char *what) {
    if (!mxIsChar(a) || mxGetString(a, buf, n)) mexErrMsgIdAndTxt("kpeig:arg", "%s must be a short char", what);
}
static mxArray *copy_square(const mxArray *a, I *n) {
    if (!mxIsDouble(a) || mxIsComplex(a) || mxGetM(a) != mxGetN(a)) mexErrMsgIdAndTxt("kpeig:arg", "A must be real square double");
    *n = (I)mxGetM(a);
    return mxDuplicateArray(a);
}
static mxArray *col(I n, const double *src) {
    mxArray *o = mxCreateDoubleMatrix((mwSize)n, 1, mxREAL);
    if (n > 0 && src) memcpy(mxGetPr(o), src, (size_t)n * sizeof(double));
    return o;
}
static void chk(I info, const char *who) {
    if (info != 0) mexErrMsgIdAndTxt("kpeig:info", "%s returned info = %lld", who, (long long)info);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    char mode[16], jobz[4] = "V", uplo[4] = "L";
    I n, info = 0, lwork = -1, liwork = -1, iq;
    double wq;
    if (nrhs < 2) mexErrMsgIdAndTxt("kpeig:arg", "kpeig(mode, ...)");
    str(prhs[0], mode, sizeof mode, "mode");

    if (!strcmp(mode, "syev") || !strcmp(mode, "syevd") || !strcmp(mode, "syevr") || !strcmp(mode, "syevx")) {
        if (nrhs >= 3) str(prhs[2], jobz, sizeof jobz, "jobz");
        if (nrhs >= 4) str(prhs[3], uplo, sizeof uplo, "uplo");
        mxArray *A = copy_square(prhs[1], &n);
        double *a = mxGetPr(A), *w = mxCalloc(n > 0 ? n : 1, sizeof(double));
        I lda = n > 0 ? n : 1;
        if (!strcmp(mode, "syev")) {
            dsyev_(jobz, uplo, &n, a, &lda, w, &wq, &lwork, &info); chk(info, "dsyev query");
            lwork = (I)wq; double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
            dsyev_(jobz, uplo, &n, a, &lda, w, work, &lwork, &info); chk(info, "dsyev");
            plhs[0] = col(n, w); if (nlhs > 1) plhs[1] = A; else mxDestroyArray(A);
        } else if (!strcmp(mode, "syevd")) {
            dsyevd_(jobz, uplo, &n, a, &lda, w, &wq, &lwork, &iq, &liwork, &info); chk(info, "dsyevd query");
            lwork = (I)wq; liwork = iq;
            double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
            I *iwork = mxCalloc(liwork > 0 ? liwork : 1, sizeof(I));
            dsyevd_(jobz, uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info); chk(info, "dsyevd");
            plhs[0] = col(n, w); if (nlhs > 1) plhs[1] = A; else mxDestroyArray(A);
        } else {
            const char *range = "A"; double vl = 0, vu = 0, abstol = 0; I il = 1, iu = n, m = 0, ldz = lda;
            mxArray *Z = mxCreateDoubleMatrix((mwSize)n, (mwSize)n, mxREAL);
            double *z = mxGetPr(Z);
            if (!strcmp(mode, "syevr")) {
                I *isuppz = mxCalloc(2 * (n > 0 ? n : 1), sizeof(I));
                dsyevr_(jobz, range, uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, isuppz, &wq, &lwork, &iq, &liwork, &info);
                chk(info, "dsyevr query");
                lwork = (I)wq; liwork = iq;
                double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
                I *iwork = mxCalloc(liwork > 0 ? liwork : 1, sizeof(I));
                dsyevr_(jobz, range, uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
                chk(info, "dsyevr");
            } else {
                I *iwork = mxCalloc(5 * (n > 0 ? n : 1), sizeof(I)), *ifail = mxCalloc(n > 0 ? n : 1, sizeof(I));
                dsyevx_(jobz, range, uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, &wq, &lwork, iwork, ifail, &info);
                chk(info, "dsyevx query");
                lwork = (I)wq; double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
                dsyevx_(jobz, range, uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
                chk(info, "dsyevx");
            }
            plhs[0] = col(n, w); if (nlhs > 1) plhs[1] = Z; else mxDestroyArray(Z);
            mxDestroyArray(A);
        }
        return;
    }
    if (!strcmp(mode, "sytrd")) {
        if (nrhs >= 3) str(prhs[2], uplo, sizeof uplo, "uplo");
        mxArray *A = copy_square(prhs[1], &n);
        I lda = n > 0 ? n : 1;
        double *d = mxCalloc(lda, sizeof(double)), *e = mxCalloc(lda, sizeof(double)), *tau = mxCalloc(lda, sizeof(double));
        dsytrd_(uplo, &n, mxGetPr(A), &lda, d, e, tau, &wq, &lwork, &info); chk(info, "dsytrd query");
        lwork = (I)wq; double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
        dsytrd_(uplo, &n, mxGetPr(A), &lda, d, e, tau, work, &lwork, &info); chk(info, "dsytrd");
        plhs[0] = col(n, d);
        if (nlhs > 1) plhs[1] = col(n > 0 ? n - 1 : 0, e);
        if (nlhs > 2) plhs[2] = col(n > 0 ? n - 1 : 0, tau);
        if (nlhs > 3) plhs[3] = A; else mxDestroyArray(A);
        return;
    }
    if (!strcmp(mode, "orgtr")) {
        if (nrhs < 3) mexErrMsgIdAndTxt("kpeig:arg", "orgtr needs Ar, tau");
        if (nrhs >= 4) str(prhs[3], uplo, sizeof uplo, "uplo");
        mxArray *A = copy_square(prhs[1], &n);
        I lda = n > 0 ? n : 1;
        const double *tau = mxGetPr(prhs[2]);
        dorgtr_(uplo, &n, mxGetPr(A), &lda, tau, &wq, &lwork, &info); chk(info, "dorgtr query");
        lwork = (I)wq; double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
        dorgtr_(uplo, &n, mxGetPr(A), &lda, tau, work, &lwork, &info); chk(info, "dorgtr");
        plhs[0] = A;
        return;
    }
    if (!strcmp(mode, "stedc") || !strcmp(mode, "steqr") || !strcmp(mode, "sterf")) {
        if (nrhs < 3) mexErrMsgIdAndTxt("kpeig:arg", "%s needs d, e", mode);
        n = (I)mxGetNumberOfElements(prhs[1]);
        double *d = mxCalloc(n > 0 ? n : 1, sizeof(double)), *e = mxCalloc(n > 0 ? n : 1, sizeof(double));
        if (n) memcpy(d, mxGetPr(prhs[1]), (size_t)n * sizeof(double));
        if (n > 1) memcpy(e, mxGetPr(prhs[2]), (size_t)(n - 1) * sizeof(double));
        I ldz = n > 0 ? n : 1;
        if (!strcmp(mode, "sterf")) {
            dsterf_(&n, d, e, &info); chk(info, "dsterf");
            plhs[0] = col(n, d);
            return;
        }
        mxArray *Z = mxCreateDoubleMatrix((mwSize)n, (mwSize)n, mxREAL);
        if (!strcmp(mode, "steqr")) {
            double *work = mxCalloc(n > 1 ? 2 * n - 2 : 1, sizeof(double));
            dsteqr_("I", &n, d, e, mxGetPr(Z), &ldz, work, &info); chk(info, "dsteqr");
        } else {
            dstedc_("I", &n, d, e, mxGetPr(Z), &ldz, &wq, &lwork, &iq, &liwork, &info); chk(info, "dstedc query");
            lwork = (I)wq; liwork = iq;
            double *work = mxCalloc(lwork > 0 ? lwork : 1, sizeof(double));
            I *iwork = mxCalloc(liwork > 0 ? liwork : 1, sizeof(I));
            dstedc_("I", &n, d, e, mxGetPr(Z), &ldz, work, &lwork, iwork, &liwork, &info); chk(info, "dstedc");
        }
        plhs[0] = col(n, d);
        if (nlhs > 1) plhs[1] = Z; else mxDestroyArray(Z);
        return;
    }
    mexErrMsgIdAndTxt("kpeig:mode", "unknown mode %s", mode);
}
