Bootstrap
================

Examples
--------

``` r
# it's very important to set the seed when doing bootstrap analysis
set.seed(1) 
```

``` r
n_samp = 250

sim_df_const = 
  tibble(
    x = rnorm(n_samp, 1, 1),
    error = rnorm(n_samp, 0, 1),
    y = 2 + 3 * x + error
  )

sim_df_nonconst = 
  sim_df_const %>% 
  mutate(error = error * .75 * x, # error is the function of covariate
         y = 2 + 3 * x + error)
# as x increase, error increase
```

``` r
sim_df = 
  bind_rows(const = sim_df_const, nonconst = sim_df_nonconst, .id = "data_source") 

sim_df %>% 
  ggplot(aes(x = x, y = y)) + 
  geom_point(alpha = .5) +
  stat_smooth(method = "lm") +
  facet_grid(~data_source) 
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-4-1.png" width="90%" />

``` r
lm(y ~ x, data = sim_df_const) %>% 
  broom::tidy() %>% 
  knitr::kable(digits = 3)
```

| term        |  estimate|  std.error|  statistic|  p.value|
|:------------|---------:|----------:|----------:|--------:|
| (Intercept) |     1.977|      0.098|     20.157|        0|
| x           |     3.045|      0.070|     43.537|        0|

``` r
lm(y ~ x, data = sim_df_nonconst) %>% 
  broom::tidy() %>% 
  knitr::kable(digits = 3)
```

| term        |  estimate|  std.error|  statistic|  p.value|
|:------------|---------:|----------:|----------:|--------:|
| (Intercept) |     1.934|      0.105|     18.456|        0|
| x           |     3.112|      0.075|     41.661|        0|

Bootstrape
----------

``` r
# ?sample_frac

# create a bootstrap sample (for bootstrap, we always take a sample with the same number of population)
boot_sample = function(df) {
  sample_frac(df, replace = TRUE)
}
```

``` r
boot_sample(sim_df_nonconst) %>% 
  ggplot(aes(x = x, y = y)) + 
  geom_point(alpha = .5) +
  stat_smooth(method = "lm")
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-8-1.png" width="90%" />

``` r
# rerun the bootstrap sample for 1000 times (resample the data 1000 time to get 1000 dataset)
boot_straps = data_frame(
  strap_number = 1:1000,
  strap_sample = rerun(1000, boot_sample(sim_df_nonconst))
)

boot_straps 
## # A tibble: 1,000 x 2
##    strap_number strap_sample      
##           <int> <list>            
##  1            1 <tibble [250 x 3]>
##  2            2 <tibble [250 x 3]>
##  3            3 <tibble [250 x 3]>
##  4            4 <tibble [250 x 3]>
##  5            5 <tibble [250 x 3]>
##  6            6 <tibble [250 x 3]>
##  7            7 <tibble [250 x 3]>
##  8            8 <tibble [250 x 3]>
##  9            9 <tibble [250 x 3]>
## 10           10 <tibble [250 x 3]>
## # ... with 990 more rows
```

``` r
# some sample something get several times overrepresented and some people are underrepresented

# some values are repeated, some don’t appear in both datasets.
# in this case, sampleing with replacement allow us to understand variability get different results of analysis
# a way to have more sample, but not a way for more data(not increase the number of sample size, but increase the number of repeating samples)
boot_straps %>% 
  filter(strap_number %in% 1:2) %>% 
  # arrage the data according to x
  mutate(strap_sample = map(strap_sample, ~arrange(.x, x))) %>% 
  pull(strap_sample)
## [[1]]
## # A tibble: 250 x 3
##         x   error       y
##     <dbl>   <dbl>   <dbl>
##  1 -1.21  -0.781  -2.43  
##  2 -0.914 -0.908  -1.65  
##  3 -0.914 -0.908  -1.65  
##  4 -0.733  0.447   0.248 
##  5 -0.733  0.447   0.248 
##  6 -0.733  0.447   0.248 
##  7 -0.733  0.447   0.248 
##  8 -0.641 -0.416  -0.338 
##  9 -0.606 -0.106   0.0774
## 10 -0.536  0.0227  0.413 
## # ... with 240 more rows
## 
## [[2]]
## # A tibble: 250 x 3
##         x   error      y
##     <dbl>   <dbl>  <dbl>
##  1 -1.29   1.40   -0.454
##  2 -0.989 -1.97   -2.93 
##  3 -0.914 -0.908  -1.65 
##  4 -0.805  0.292  -0.123
##  5 -0.733  0.447   0.248
##  6 -0.733  0.447   0.248
##  7 -0.733  0.447   0.248
##  8 -0.665 -0.544  -0.539
##  9 -0.536  0.0227  0.413
## 10 -0.536  0.0227  0.413
## # ... with 240 more rows
```

``` r
# the samples are different, but the linear regression looks the same
boot_straps %>% 
  filter(strap_number %in% 1:3) %>% 
  unnest() %>% 
  ggplot(aes(x = x, y = y)) + 
  geom_point(alpha = .5) +
  stat_smooth(method = "lm", se = FALSE) +
  facet_grid(~strap_number) 
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-11-1.png" width="90%" />

``` r
bootstrap_results = 
  boot_straps %>% 
  mutate(models = map(strap_sample, ~lm(y ~ x, data = .x) ),
         results = map(models, broom::tidy)) %>% 
  select(-strap_sample, -models) %>% 
  unnest() %>% 
  group_by(term) %>% 
  summarize(boot_se = sd(estimate))

bootstrap_results %>% 
  knitr::kable(digits = 3)
```

| term        |  boot\_se|
|:------------|---------:|
| (Intercept) |     0.077|
| x           |     0.106|

``` r
lm(y ~ x, data = sim_df_nonconst) %>% 
  broom::tidy() %>% 
  select(term, std.error) %>% 
  knitr::kable(digits = 3)
```

| term        |  std.error|
|:------------|----------:|
| (Intercept) |      0.105|
| x           |      0.075|

``` r
# the variability 
boot_straps %>% 
  unnest() %>% 
  ggplot(aes(x = x, y = y)) + 
  geom_line(aes(group = strap_number), stat = "smooth", method = "lm", se = FALSE, alpha = .1, color = "blue") +
  geom_point(data = sim_df_nonconst, alpha = .5)
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-14-1.png" width="90%" />

``` r
# this doing the quite same thing
boot_straps = 
  sim_df_nonconst %>% 
  # use modelr to do the bootstrap for  1000 times
  modelr::bootstrap(n = 1000)

boot_straps$strap[[1]]
## <resample [250 x 3]> 228, 50, 145, 2, 208, 160, 25, 179, 149, 11, ...
## <resample [250 x 3]> 228, 50, 145, 2, 208, 160, 25, 179, 149, 11, ...
as_data_frame(boot_straps$strap[[1]])
## # A tibble: 250 x 3
##         x    error       y
##     <dbl>    <dbl>   <dbl>
##  1 -0.606 -0.106    0.0774
##  2  1.88  -0.431    7.21  
##  3 -0.116  0.00958  1.66  
##  4  1.18   0.361    5.91  
##  5  1.54  -2.43     4.20  
##  6  2.87  -1.50     9.11  
##  7  1.62   0.190    7.05  
##  8  2.03  -0.975    7.11  
##  9 -0.286  0.154    1.29  
## 10  2.51   1.79    11.3   
## # ... with 240 more rows
```

``` r
sim_df_nonconst %>% 
  modelr::bootstrap(n = 1000) %>% 
  mutate(models = map(strap, ~lm(y ~ x, data = .x) ),
         results = map(models, broom::tidy)) %>% 
  select(-strap, -models) %>% 
  unnest() %>% 
  group_by(term) %>% 
  summarize(boot_se = sd(estimate))
## # A tibble: 2 x 2
##   term        boot_se
##   <chr>         <dbl>
## 1 (Intercept)  0.0744
## 2 x            0.101
# we get different answers, since bootstrap sample is different every time you resampling
```

``` r
lm(y ~ x, data = sim_df_const) %>% 
  broom::tidy() %>% 
  knitr::kable(digits = 3)
```

| term        |  estimate|  std.error|  statistic|  p.value|
|:------------|---------:|----------:|----------:|--------:|
| (Intercept) |     1.977|      0.098|     20.157|        0|
| x           |     3.045|      0.070|     43.537|        0|

``` r

# the bootstrap is quite the same thing as the correct model
# bootstrap to check and get something really similar
sim_df_const %>% 
  modelr::bootstrap(n = 1000) %>% 
  mutate(models = map(strap, ~lm(y ~ x, data = .x) ),
         results = map(models, broom::tidy)) %>% 
  select(-strap, -models) %>% 
  unnest() %>% 
  group_by(term) %>% 
  summarize(boot_se = sd(estimate))
## # A tibble: 2 x 2
##   term        boot_se
##   <chr>         <dbl>
## 1 (Intercept)  0.0968
## 2 x            0.0674
```

``` r
sim_df = 
  tibble(
    x = rnorm(25, 1, 1),
    error = rnorm(25, 0, 1),
    y = 2 + 3 * x + error
  )
```

``` r
sim_df %>% 
  modelr::bootstrap(n = 1000) %>% 
  mutate(models = map(strap, ~lm(y ~ x, data = .x) ),
         results = map(models, broom::glance)) %>% 
  select(-strap, -models) %>% 
  unnest() %>% 
  ggplot(aes(x = r.squared)) + geom_density()
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-19-1.png" width="90%" />

We could take the 2.5% and 97.5% quantiles of the estimates across bootstrap samples, because the shape isn’t symmetric, using the mean +/- 1.96 times the standard error probably wouldn’t work well

``` r
sim_df %>% 
  modelr::bootstrap(n = 1000) %>% 
  mutate(models = map(strap, ~lm(y ~ x, data = .x) ),
         results = map(models, broom::tidy)) %>% 
  select(-strap, -models) %>% 
  unnest() %>% 
  select(id = `.id`, term, estimate) %>% 
  spread(key = term, value = estimate) %>% 
  rename(beta0 = `(Intercept)`, beta1 = x) %>% 
  mutate(log_b0b1 = log(beta0 * beta1)) %>% 
  ggplot(aes(x = log_b0b1)) + geom_density()
## Warning in log(beta0 * beta1): NaNs produced
## Warning: Removed 5 rows containing non-finite values (stat_density).
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-20-1.png" width="90%" />

``` r
sim_df %>% 
  modelr::bootstrap(n = 1000) %>% 
  mutate(models = map(strap, ~lm(y ~ x, data = .x) ),
         results = map(models, broom::tidy)) %>% 
  select(-strap, -models) %>% 
  unnest() %>% 
  select(id = `.id`, term, estimate) %>% 
  spread(key = term, value = estimate) %>% 
  rename(beta0 = `(Intercept)`, beta1 = x) %>% 
  mutate(log_b0b1 = log(beta0 * beta1)) #%>% 
## Warning in log(beta0 * beta1): NaNs produced
## # A tibble: 1,000 x 4
##    id    beta0 beta1 log_b0b1
##    <chr> <dbl> <dbl>    <dbl>
##  1 0001  1.94   3.17     1.81
##  2 0002  1.93   3.30     1.85
##  3 0003  2.05   3.43     1.95
##  4 0004  1.11   3.66     1.40
##  5 0005  2.33   2.82     1.88
##  6 0006  1.49   3.71     1.71
##  7 0007  1.60   3.25     1.65
##  8 0008  1.73   3.11     1.69
##  9 0009  0.825  3.85     1.16
## 10 0010  2.00   3.15     1.84
## # ... with 990 more rows
  #na.omit() %>% 
  #summarise(quantile(log_b0b1, c(0.025, 0.975)))
```

``` r
data("nyc_airbnb")

nyc_airbnb = 
  nyc_airbnb %>% 
  mutate(stars = review_scores_location / 2) %>% 
  rename(boro = neighbourhood_group,
         neighborhood = neighbourhood) %>% 
  filter(boro != "Staten Island") %>% 
  select(price, stars, boro, neighborhood, room_type)
```

``` r
nyc_airbnb %>% 
  filter(boro == "Manhattan") %>% 
  modelr::bootstrap(n = 1000) %>% 
  mutate(models = map(strap, ~ lm(price ~ stars + room_type, data = .x)),
         results = map(models, broom::tidy)) %>% 
  select(results) %>% 
  unnest() %>% 
  filter(term == "stars") %>% 
  ggplot(aes(x = estimate)) + geom_density()
```

<img src="bootstrap_files/figure-markdown_github/unnamed-chunk-23-1.png" width="90%" />
