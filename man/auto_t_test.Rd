% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/auto_t_test.R
\name{auto_t_test}
\alias{auto_t_test}
\title{auto t test}
\usage{
auto_t_test(data, col, ..., var_equal = FALSE, abbrv = TRUE)
}
\arguments{
\item{data}{dataframe}

\item{col}{a column with 2 categories representing the 2 populations}

\item{...}{numeric variables to perform t.test on. Default is to select all numeric variables}

\item{var_equal}{default FALSE; t.test parameter}

\item{abbrv}{default TRUE; remove some extra columns from output}
}
\value{
dataframe
}
\description{
Performs a t.test on 2 populations for numeric variables.
}
\examples{


iris \%>\%
 dplyr::filter(Species != "setosa") \%>\%
 auto_t_test(col = Species)
}
