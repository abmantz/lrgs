Version | Comments
------- | --------
0.5.5   | Fixed post2dataframe so that column names will always be unique
0.5.4   | Fixed a bug that occurred when p=1 and a starting value of Tau was passed
0.5.3   | Fixed a bug that struck when p+m==1 --> CRAN
0.5.2   | Fixed a bug that would strike when dirichlet=TRUE and *not* tracing G
0.5.1   | Added published paper reference information to the Gibbs.regression help. --> CRAN
0.5.0   | M and M.inv are now passed as 3-dimensional arrays rather than lists of matrices. This should make it more convenient to define M based on columns of a table. **Not backwards-compatible.**
0.4.2   | First public release --> CRAN
