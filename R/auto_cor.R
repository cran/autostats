#' auto correlation
#'
#' Finds the correlation between numeric variables in a data frame, chosen using tidyselect.
#' Additional parameters for the correlation test can be specified as in \code{\link[stats]{cor.test}}
#'
#' includes the asymmetric correlation coefficient xi from \code{\link[XICOR]{xicor}}
#'
#' @param .data data frame
#' @param ... tidyselect cols
#' @param use method to deal with na. Default is to remove rows with NA
#' @param method correlation method. default is pearson, but also supports xicor.
#' @param include_nominals logicals, default TRUE. Dummify nominal variables?
#' @param max_levels maximum numbers of dummies to be created from nominal variables
#' @param sparse logical, default TRUE. Filters and arranges cor table
#' @param pval_thresh threshold to filter out weak correlations
#'
#' @return data frame of correlations
#' @export
#'
#' @examples
#'
#' iris %>%
#' auto_cor()
#'
#' # don't use sparse if you're interested in only one target variable
#' iris %>%
#' auto_cor(sparse = FALSE) %>%
#' dplyr::filter(x == "Petal.Length")
auto_cor <- function(.data, ...,
                     use = c("pairwise.complete.obs", "all.obs", "complete.obs",
                                         "everything", "na.or.complete"),
                     method =  c("pearson", "kendall", "spearman", "xicor"),
                     include_nominals = TRUE,
                     max_levels = 5L,
                     sparse = TRUE,
                     pval_thresh = .1){

  na.method <- match.arg(use)

  cor.method <- match.arg(method)

  p.value <- statistic <- NULL


  .data %>%
    framecleaner::select_otherwise(..., otherwise = tidyselect::everything(), return_type = "df") -> .data

  .data %>% names -> dnames0

  if(include_nominals){

    .data %>%
      framecleaner::select_otherwise(..., otherwise = tidyselect::everything(), return_type = "df") -> .data

    .data %>%
      framecleaner::create_dummies(max_levels = max_levels,
                                   append_col_name = TRUE) -> .data
  } else {


    .data %>%
      framecleaner::select_otherwise(..., otherwise = where(is.numeric), return_type = "df") -> .data
  }

names(.data) -> dnames


setdiff(dnames, dnames0) -> dummy_names

if(cor.method == "xicor"){
  cor_fun <- calc_xicor
} else{
  cor_fun <- calc_cor
}


cor_list <- list()

if(!sparse){

  for(i in dnames){

    for(j in setdiff(dnames, i)){

      if(check_same_dummy(.data, i, j, dummy_names)) {next}

      cor_fun(.data, i, j, cor_list, na.method, cor.method) -> cor_list
    }
  }

  rlist::list.rbind(cor_list) %>%
    dplyr::mutate(significance = gtools::stars.pval(p.value)) -> corlist
} else{

1:length(dnames) -> ldnames

for(i in ldnames){

  for(j in 1:(i-1)){

    if(j == 0 | i == j) {next}

    n1 <- dnames[i]
    n2 <- dnames[j]

    if(check_same_dummy(.data, n1, n2, dummy_names)) {next}

    cor_fun(.data, n1, n2, cor_list, na.method, cor.method) -> cor_list

  }
}

  rlist::list.rbind(cor_list) %>%
    dplyr::mutate(significance = gtools::stars.pval(p.value)) -> corlist

  cor <- NULL

  corlist %>%
    dplyr::arrange(dplyr::desc(abs(cor))) %>%
    dplyr::filter(p.value < pval_thresh) %>%
    dplyr::mutate(method = cor.method) -> corlist
}


corlist
}


check_same_dummy <- function(db, nm1, nm2, dummynames){

  (nm1 %in% dummynames) &
  (nm2 %in% dummynames) &
  (sum(unique(db[[nm1]] + db[[nm2]])) == 1)

}


calc_cor <- function(.data, n1, n2, cor_list, na.method, cor.method){

  stats::cor.test(.data[[n1]], .data[[n2]], use = na.method, method = cor.method) -> cor1

  cor_list <- cor_list %>%
    rlist::list.append(
      tibble::tibble(x = n1, y = n2, cor = cor1$estimate, p.value = cor1$p.value)
    )

}

calc_xicor <- function(.data, n1, n2, cor_list, ...){

  XICOR::xicor(.data[[n1]], .data[[n2]], pvalue = TRUE) -> cor1

  cor_list <- cor_list %>%
    rlist::list.append(
      tibble::tibble(x = n1, y = n2, cor = cor1$xi, p.value = cor1$pval)
    )
}

