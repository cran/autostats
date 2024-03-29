% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tidy_shap.R
\name{tidy_shap}
\alias{tidy_shap}
\title{tidy shap}
\usage{
tidy_shap(model, newdata, form = NULL, ..., top_n = 12, aggregate = NULL)
}
\arguments{
\item{model}{xgboost model}

\item{newdata}{dataframe similar to model input}

\item{form}{formula used for model}

\item{...}{additional parameters for shapley value}

\item{top_n}{top n features}

\item{aggregate}{a character vector. Predictors containing the string will be aggregated, and renamed to that string.}
}
\value{
list
}
\description{
plot and summarize shapley values from an xgboost model
}
\details{
returns a list with the following entries

\describe{
\item{\emph{shap_tbl}}{: table of shaply values}
\item{\emph{shap_summary}}{: table summarizing shapley values. Includes correlation between shaps and feature values.}
\item{\emph{swarmplot}}{: one plot showing the relation between shaps and features}
\item{\emph{scatterplots}}{: returns the top 9 most important features as determined by sum of absolute shapley values, as a facetted scatterplot of feature vs shap}
}
}
