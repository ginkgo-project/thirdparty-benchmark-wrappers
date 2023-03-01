#ifndef MA57_HPP_
#define MA57_HPP_

extern "C" {

void MA57ID(double *cntl, int *icntl);

void MA57AD(int *n, int *ne, int *irn, int *jcn, int *lkeep, int *keep,
            int *iwork, int *icntl, int *info, double *rinfo);

void MA57BD(int *n, int *ne, double *a, double *fact, int *lfact, int *ifact,
            int *lifact, int *lkeep, int *keep, int *ppos, int *icntl,
            double *cntl, int *info, double *rinfo);
void MA57CD(int *job, int *n, double *fact, int *lfact, int *ifact, int *lifact,
            int *nrhs, double *rhs, int *lrhs, double *w, int *lw, int iw1[],
            int *icntl, int *info);
void MA57DD(int *job, int *n, int *ne, double *a, int *irn, int *jcn,
            double *fact, int *lfact, int *ifact, int *lifact, double *rhs,
            double *x, double *resid, double *w, int *iw, int *icntl,
            double *cntl, int *info, double *rinfo);
void MA57ED(int *n, int *ic, int *keep, double *fact, int *lfact,
            double *newfac, int *lnew, int *ifact, int *lifact, int *newifc,
            int *linew, int *info);
}

#endif // MA57_HPP_
