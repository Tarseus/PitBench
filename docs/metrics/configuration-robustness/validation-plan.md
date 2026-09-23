# Configuration Robustness & Tunability validation plan

## M3: approved multi-fidelity falsification cases

The 2026-09-23 M2 revision is falsified by the following cases before a real
M5 panel is accepted:

1. A higher fidelity panel must contain every paired instance/seed position
   from each preceding level and no position outside the declared full panel.
2. A candidate that has a missing, invalid, rejected, crashed, or otherwise
   unavailable feedback position at its current fidelity level must not receive
   numeric feedback or promotion.
3. Promotion must retain exactly the declared count of configurations with the
   largest complete paired degradation; tied values retain first evaluation
   order.
4. Only the two configurations completing the declared full-panel level may
   compete for final selection; lower-fidelity results alone cannot select the
   retest configuration.
5. Previously evaluated nested positions must be reused when a configuration is
   promoted, rather than executed a second time.
6. Choco and VROOM must enumerate their legal non-default finite spaces on
   complete panels and must not be eliminated by the multi-fidelity schedule.
7. The independent default/selected retest remains a complete panel and never
   changes the selected configuration.
