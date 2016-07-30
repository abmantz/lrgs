\name{Gibbs.post2dataframe}
\alias{Gibbs.post2dataframe}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Linear Regression by Gibbs Sampling
}
\description{
Transforms a set of posterior samples produced by Gibbs.regression into a data frame for more straightforward analysis.
}
\usage{
Gibbs.post2dataframe(p)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{p}{
an object returned from Gibbs.regression.
}
}
\value{
A data frame with one column corresponding to each parameter stored in post.
}
\author{
Adam Mantz
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{Gibbs.regression}}
}
