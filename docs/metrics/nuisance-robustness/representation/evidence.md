# Representation Robustness evidence

## M1: equivalent customer relabeling

This document records the four evidence conclusions approved by the user on
2026-09-07. The research scope and approved IQR direction are recorded in
[scope.md](scope.md).

## 1. Equivalent permutations can affect traditional solvers

Koch et al. (2011), *MIPLIB 2010*, Mathematical Programming Computation 3:103–163,
provides direct experimental evidence for traditional MIP solvers.

- DOI: <https://doi.org/10.1007/s12532-011-0025-9>
- Full text: <https://opus4.kobv.de/opus4-mpc/files/25/Koch2011_Article_MIPLIB2010.pdf>
- Verification: the original text of Sections 5.2–5.4 was inspected.

Sections 5.2–5.3 describe solving 100 row and column permutations of each of 67
models with SCIP/SoPlex and observing changes in solution time. Section 5.4
discusses the consequences of performance variability for comparisons between
solvers, parameter settings, and algorithm changes.

This is evidence that equivalent input permutations can affect a traditional
solver. It does not establish the magnitude of customer-relabeling effects in
PyVRP.

## 2. Dispersion across equivalent representations has a precedent

Section 5.2 summarizes performance across permutations using the coefficient of
variation of solution time. The relevant precedent for PitBench is the
experimental structure: run multiple equivalent representations of the same
problem and summarize the dispersion of the resulting performance values.

PitBench's approved direction instead uses the IQR of gap values at a fixed time
budget, conditional on the instance and solver seed. The paper's runtime
statistic is not adopted as the PitBench estimator.

## 3. Fixed solver seed plus IQR is a project protocol choice

The sources reviewed for this item do not directly prescribe this CVRP protocol
or establish that a particular number of permutations is sufficient. The
fixed-seed IQR direction is the user's approved project choice, not a protocol
claimed to have been established by the cited MIP experiment.

The 100 permutations used in MIPLIB 2010 are an experimental setting, not a
required sample size for PitBench.

## 4. Decisions reserved for M2

The following protocol choices remain open:

- How customer permutations are generated and how many are used.
- Whether the original representation participates in the IQR calculation.
- How per-instance, per-seed IQR values are aggregated across solver seeds and
  instances.

This evidence document does not select those settings or change the frozen Seed
Robustness protocol.
