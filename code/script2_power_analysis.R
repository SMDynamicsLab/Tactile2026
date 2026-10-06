##
## Tactile2026
## (Correa Morales, García Lodi Fe, Silva, Versaci, Spiousas, Laje)
##
##
## =====================================================================
## A-priori power analysis by simulation for a linear mixed model
##
## H0:  dSEC/dIPI  ==  -1   at ISI = 550
## H1:  dSEC/dIPI  ==  -1 + delta
##
## ISI is a two-level factor ("400", "550"). Releveling to 550 as the
## reference makes beta["IPIc"] the IPI slope at ISI = 550 directly.
##
## Design and covariance structure are taken from the pilot data;
## only the target slope is replaced by the hypothesised value.
##
## MIGHT TAKE A LONG TIME
## Start with n_simul = 200 to gauge runtime, then raise to 1000-2000.
##
## Clone repository and set working directory to 'Tactile2026/'
## =====================================================================

library(lme4)
library(lmerTest)   # masks lmer(); needed for Satterthwaite df
library(dplyr)
library(readr)
library(tidyr)
library(here)
library(future)
library(future.apply)

set.seed(20260909)

NULL_SLOPE <- -1       # value of the IPI slope under H0
ISI_REF    <- "550"    # factor level at which the slope is evaluated
ALPHA      <- 0.05     # two-sided
n_subj_full <- 47      # sample size in the full experiment
n_simul <- 1000

## Effect-size scale for `delta`. See section 2b for what each denominator
## means and when to prefer it.
##   "tau"      delta is in units of the between-subject SD of the IPI slope
##   "beta_std" delta is a fully standardized regression coefficient
##   "sigma"    delta is in units of residual SD per SD of IPI
##   "raw"      delta is in the native slope units (the original behaviour)
ES_SCALE <- "raw"

## Parallel backend. multisession works everywhere; on Linux/macOS outside
## RStudio, plan(multicore) forks instead and starts up much faster.
N_WORKERS <- round(max(1, parallel::detectCores())/2)
plan(multisession, workers = N_WORKERS)
## plan(sequential)                        # to debug

## ---------------------------------------------------------------------
## 0. Load and clean the pilot data
## ---------------------------------------------------------------------

DMC_pilot.raw <- read_csv(here('raw data','pilot_data.csv'), show_col_types = FALSE) %>%
# DMC_pilot.raw <- read_csv('pilot_data.csv', show_col_types = FALSE) %>%
	mutate(ISI = as.factor(ISI)) %>%
  drop_na(firstPeak) %>%          # los 214 taps sin picos identificados
  mutate(IPI = secPeak - firstPeak,
         SEP = SEC + secPeak) %>%
  filter(IPI > 0)                 # el segundo pico llega después del primero

## Outlier removal on SEC, per subject, 3 MAD
DMC_pilot.power <- DMC_pilot.raw %>%
  group_by(Subject) %>%
  filter(abs(SEC - median(SEC, na.rm = TRUE)) <= 3 * mad(SEC, na.rm = TRUE)) %>%
  ungroup()

summary(DMC_pilot.power)
summary(DMC_pilot.raw)

## Which dataset the power analysis is built on. It should be the one your
## real analysis pipeline will produce, i.e. the cleaned one: sigma and the
## random-effect SDs estimated from uncleaned data are inflated by the very
## outliers you intend to discard, which would understate power.
PILOT_DATA <- DMC_pilot.power

cat("\nRows: raw =", nrow(DMC_pilot.raw),
    " cleaned =", nrow(DMC_pilot.power),
    sprintf(" (%.1f%% removed)\n",
            100 * (1 - nrow(DMC_pilot.power) / nrow(DMC_pilot.raw))))

## ---------------------------------------------------------------------
## 1. Reparameterise the pilot model
##
## With treatment contrasts, beta["IPIc"] is the IPI slope at the REFERENCE
## level of ISI. Releveling to "550" therefore puts the coefficient you want
## to test on a single row of the fixed-effect table, with the right SE and
## df. The interaction becomes the (400 - 550) difference in slope.
##
## Centering IPI does NOT change beta["IPIc"] or its SE; it only makes the
## random intercept refer to the middle of the IPI range, which conditions
## the fit better and makes the intercept/slope correlation interpretable.
## ---------------------------------------------------------------------

prep <- function(dat, isi_ref = ISI_REF, ipi_center = NULL) {
  if (is.null(ipi_center)) ipi_center <- mean(dat$IPI, na.rm = TRUE)
  dat$ISI     <- relevel(factor(dat$ISI), ref = isi_ref)
  dat$IPIc    <- dat$IPI - ipi_center
  dat$Subject <- factor(dat$Subject)
  attr(dat, "ipi_center") <- ipi_center
  dat
}

d <- prep(PILOT_DATA)
stopifnot(levels(d$ISI)[1] == ISI_REF)   # reference must be 550

## Same model specification as in full data, except for Feedback (absent in pilot data)
form     <- SEC ~ IPIc * ISI + (1 + IPIc | Subject)
sim_form <-     ~ IPIc * ISI + (1 + IPIc | Subject)   # RHS only, for simulate()

ctrl <- lmerControl(optimizer = "Nelder_Mead", calc.derivs = FALSE)

m_pilot <- lmer(form, data = d, control = ctrl)
print(summary(m_pilot))

## Sanity check: this is the pilot's dSEC/dIPI at ISI = 550, i.e. the row
## you will be testing against -1. The other rows should now read as
## "intercept at ISI=550", "ISI400 - ISI550 offset", "slope difference".
cat("\nPilot IPI slope at ISI =", ISI_REF, ":",
    round(fixef(m_pilot)["IPIc"], 4),
    " SE:", round(sqrt(vcov(m_pilot)["IPIc", "IPIc"]), 4), "\n\n")

## ---------------------------------------------------------------------
## 2. Data-generating parameters
##
## theta is lme4's *relative* Cholesky factor (scaled by sigma), so carrying
## theta and sigma over together reproduces the pilot's random-effect SDs.
## ---------------------------------------------------------------------

pars <- list(beta  = fixef(m_pilot),
             theta = getME(m_pilot, "theta"),
             sigma = sigma(m_pilot))

## ---------------------------------------------------------------------
## 2b. Standardized effect sizes
##
## `delta` is a deviation of a SLOPE from -1, so "standardized" needs an
## explicit denominator. The three on offer answer different questions:
##
##   tau      = SD of the by-subject IPI slope. delta = 0.3 then means the
##              population slope sits 0.3 between-subject SDs away from -1.
##              This is the closest analogue to a one-sample Cohen's d, and
##              power tracks a one-sample t-test on subject-level slopes
##              (attenuated by the trial-level noise in each subject's slope).
##              Recommended default here.
##
##   beta_std = SD(SEC)/SD(IPI), i.e. delta is a change in the fully
##              standardized regression coefficient. Comparable across
##              studies that report standardized betas, but it mixes
##              between- and within-subject variance into SD(SEC), so it is
##              only loosely related to power.
##
##   sigma    = residual SD per SD of IPI. Sensible if your reference point
##              is trial-level noise rather than individual differences.
##
## Note that under "tau", theta_scale rescales the denominator too, so a
## sensitivity run at theta_scale = 1.5 also enlarges the raw slope
## deviation. That is deliberate: it holds the standardized effect fixed.
## Under the other scales the denominator stays at its pilot value.
## ---------------------------------------------------------------------

vc <- as.data.frame(VarCorr(m_pilot))
tau_slope <- vc$sdcor[vc$grp == "Subject" & vc$var1 == "IPIc" & is.na(vc$var2)]
sd_SEC    <- sd(d$SEC,  na.rm = TRUE)
sd_IPI    <- sd(d$IPIc, na.rm = TRUE)

cat("Effect-size denominators from the pilot:\n",
    "  tau_slope (by-subject SD of IPI slope):", round(tau_slope, 4), "\n",
    "  SD(SEC)/SD(IPI):                       ", round(sd_SEC / sd_IPI, 4), "\n",
    "  sigma/SD(IPI):                         ", round(sigma(m_pilot) / sd_IPI, 4), "\n\n")

es_unit <- function(scale = ES_SCALE, theta_scale = 1) {
  switch(scale,
         tau      = tau_slope * theta_scale,
         beta_std = sd_SEC / sd_IPI,
         sigma    = sigma(m_pilot) / sd_IPI,
         raw      = 1,
         stop("unknown ES_SCALE: ", scale))
}

## Per-subject design templates. Resampling whole subjects (rather than
## building a balanced grid) preserves the pilot's trial counts and any
## imbalance across the IPI x ISI cells. ISI keeps its releveled factor
## coding, which simulate() needs in order to build the right model matrix.
templates <- d %>%
  select(Subject, IPIc, ISI) %>%
  group_split(Subject)

make_design <- function(n, templates) {
  idx <- sample(seq_along(templates), n, replace = TRUE)
  out <- lapply(seq_len(n), function(i) {
    tmp <- templates[[idx[i]]]
    tmp$Subject <- sprintf("S%03d", i)   # relabel so resampled subjects are distinct
    tmp
  })
  out <- bind_rows(out)
  out$Subject <- factor(out$Subject)
  out$ISI     <- factor(out$ISI, levels = levels(d$ISI))   # preserve reference
  out
}

## ---------------------------------------------------------------------
## 3. One simulated experiment
##
## theta_scale multiplies the whole random-effect covariance factor; use it
## for sensitivity analysis, since with few pilot subjects the random-slope
## SD is the least well estimated quantity and the one power is most
## sensitive to.
## ---------------------------------------------------------------------

one_rep <- function(n, delta, templates, pars,
                    theta_scale = 1, alpha = ALPHA, scale = ES_SCALE) {
  
  dsim <- make_design(n, templates)
  
  delta_raw <- delta * es_unit(scale, theta_scale)   # standardized -> slope units
  
  beta <- pars$beta
  beta["IPIc"] <- NULL_SLOPE + delta_raw      # the only parameter we move
  
  y <- simulate(sim_form, nsim = 1, newdata = dsim, family = gaussian,
                newparams = list(beta  = beta,
                                 theta = pars$theta * theta_scale,
                                 sigma = pars$sigma))[[1]]
  dsim$SEC <- y
  
  fit <- try(suppressMessages(suppressWarnings(
    lmer(form, data = dsim, control = ctrl))), silent = TRUE)
  
  if (inherits(fit, "try-error"))
    return(c(est = NA, se = NA, df = NA, t = NA, p = NA, sing = NA, fail = 1))
  
  cf  <- coef(summary(fit))                    # lmerTest -> Satterthwaite df
  est <- cf["IPIc", "Estimate"]
  se  <- cf["IPIc", "Std. Error"]
  df  <- cf["IPIc", "df"]
  
  tstat <- (est - NULL_SLOPE) / se             # shifted null, same SE and df
  c(est  = est, se = se, df = df, t = tstat,
    p    = 2 * pt(-abs(tstat), df),
    sing = as.numeric(isSingular(fit)),
    fail = 0)
}

## ---------------------------------------------------------------------
## 4. Sweep over n and delta
## ---------------------------------------------------------------------

power_cell <- function(n, delta, nsim = n_simul, theta_scale = 1,
                       alpha = ALPHA, scale = ES_SCALE) {
  
  ## future.seed = TRUE switches to L'Ecuyer-CMRG streams, so results are
  ## reproducible from the top-level set.seed() and independent of how many
  ## workers you happen to have. Never use set.seed() inside the workers.
  ## Chunking amortises the per-task overhead: one lmer fit is fast enough
  ## that sending 1000 individual tasks would cost more in dispatch than in
  ## fitting.
  reps <- future_lapply(
    seq_len(nsim),
    function(i) one_rep(n, delta, templates, pars, theta_scale, alpha, scale),
    future.seed    = TRUE,
    future.packages = c("lme4", "lmerTest", "dplyr"),
    future.chunk.size = max(1, ceiling(nsim / (nbrOfWorkers() * 4)))
  )
  
  reps <- as.data.frame(do.call(rbind, reps))
  ok   <- reps$fail == 0 & !is.na(reps$p)
  pw   <- mean(reps$p[ok] < alpha)
  tibble(n           = n,
         delta       = delta,
         es_scale    = scale,
         delta_raw   = delta * es_unit(scale, theta_scale),
         true_slope  = NULL_SLOPE + delta * es_unit(scale, theta_scale),
         theta_scale = theta_scale,
         power       = pw,
         mcse        = sqrt(pw * (1 - pw) / sum(ok)),
         mean_est    = mean(reps$est[ok]),
         mean_se     = mean(reps$se[ok]),
         median_df   = median(reps$df[ok]),
         p_singular  = mean(reps$sing[ok]),
         n_fail      = sum(!ok))
}

power_grid <- function(n_vec, delta_vec, nsim = n_simul, theta_scale = 1,
                       scale = ES_SCALE) {
  grid <- expand.grid(n = n_vec, delta = delta_vec)
  res  <- vector("list", nrow(grid))
  for (i in seq_len(nrow(grid))) {
    t0 <- Sys.time()
    res[[i]] <- power_cell(grid$n[i], grid$delta[i], nsim, theta_scale, ALPHA, scale)
    cat(sprintf("n=%3d  delta=%+0.2f [%s]  slope=%+0.4f  power=%.3f (+/-%.3f)  [%.0fs]\n",
                res[[i]]$n, res[[i]]$delta, scale,
                res[[i]]$true_slope, res[[i]]$power, res[[i]]$mcse,
                as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
  bind_rows(res)
}

## Analytic ceiling for ES_SCALE = "tau": if each subject's slope were known
## without error, the test reduces to a one-sample t on n subject slopes with
## Cohen's d = delta. Simulated power should sit at or below this. A large gap
## means trial-level noise, not subject count, is your binding constraint.
power_ceiling <- function(n, delta, alpha = ALPHA) {
  crit <- qt(1 - alpha/2, n - 1)
  ncp  <- delta * sqrt(n)
  pt(-crit, n - 1, ncp) + pt(crit, n - 1, ncp, lower.tail = FALSE)
}

## ---- the run you asked for -------------------------------------------
## delta = 0 is the type-I error check: it should land near ALPHA.
## MIGHT TAKE A LONG TIME
## Start with nsim = 200 to gauge runtime, then raise to 1000-2000.

deltas <- c(0, 0.1, 0.2, 0.3, 0.4)

res_47 <- power_grid(n_vec = n_subj_full, delta_vec = deltas, nsim = n_simul)

res_47 <- res_47 %>% mutate(ceiling = power_ceiling(n, delta))
print(as.data.frame(res_47), digits = 3)

## If you are on a threaded BLAS (OpenBLAS, MKL), each worker may itself try
## to use every core, which oversubscribes the machine and can make the
## parallel version SLOWER than sequential. Pin the workers to one thread:
##   RhpcBLASctl::blas_set_num_threads(1)   # in the main session, before plan()
## or set OMP_NUM_THREADS=1 in .Renviron. lmer on data this size is unlikely
## to benefit from multithreaded BLAS anyway.

## Release the workers when done:
## plan(sequential)

## Power curve across n, at whichever delta you consider the smallest
## effect of interest:
## res_n <- power_grid(n_vec = c(20, 30, 40, n_subj_full, 60, 80),
##                     delta_vec = 0.3, nsim = n_simul)

## Sensitivity to the random-slope SD. Under ES_SCALE = "tau" this holds the
## standardized effect fixed while changing both tau and the raw slope, so it
## isolates how much the trial-level attenuation depends on tau:
## res_sens <- bind_rows(lapply(c(0.75, 1, 1.5),
##   function(s) power_grid(n_subj_full, 0.3, nsim = n_simul, theta_scale = s)))

## Same grid on a different denominator, if a reviewer asks for standardized
## betas instead:
## res_std <- power_grid(n_subj_full, deltas, nsim = n_simul, scale = "beta_std")

## ---------------------------------------------------------------------
## 5. Notes
##
## - delta is standardized according to ES_SCALE; the `delta_raw` and
##   `true_slope` columns show what each value means in native slope units.
##   Report both, since "delta = 0.3" is uninterpretable without the
##   denominator, and the denominator is itself a pilot estimate.
##
## - The random structure (1 + IPIc | Subject) gives each subject ONE IPI
##   slope shared across both ISI levels. The by-subject variability of the
##   ISI=550 slope specifically is therefore assumed equal to the overall
##   slope variance. If you expect the two ISI conditions to differ in that
##   respect, the generating model should be (1 + IPIc * ISI | Subject), and
##   with 2 levels and 47 subjects that is estimable but will likely be
##   singular in the pilot; check before adopting it.
##
## - Adding trials per subject buys much less than adding subjects: the
##   sampling variance of beta["IPIc"] has a floor of tau_slope^2 / n that
##   trials cannot touch. If power is poor, check whether tau_slope (the
##   IPIc random-effect SD in summary(m_pilot)) is what is driving it before
##   concluding you need a different design.
##
## - If isSingular is frequent in the simulations, the pilot's RE covariance
##   is near the boundary; report p_singular alongside power, and consider
##   whether the (1 + IPIc | Subject) structure is supportable.
##
## - If your prediction is that the slope equals -1 at BOTH ISI levels, that
##   is a joint test of beta["IPIc"] == -1 AND beta["IPIc:ISI400"] == 0, not
##   this single-coefficient test. Different (and lower-powered) question.
## ---------------------------------------------------------------------