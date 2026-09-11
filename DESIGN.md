# Implementation design: adaptive CP for LTE C-V2X sidelink

## 1. Scope and standards mapping

This project evaluates a **proposed, non-standard adaptive-CP PHY extension** on a
3GPP-based LTE C-V2X PC5 sidelink model. It does not claim that adaptive CP is a
Release-14 Mode-4 feature.

| Item | Primary setting | Implementation mapping |
|---|---:|---|
| Carrier | 5.9 GHz | `radio.carrier_hz` |
| Bandwidth | 10 MHz | 50-RB system assumption; occupied RBs are fixed across policies |
| Native sample rate | 15.36 MHz | `radio.sample_rate_hz` |
| FFT size / SCS | 1024 / 15 kHz | `radio.fft_size`, `radio.subcarrier_spacing_hz` |
| Waveform | SC-FDMA (DFT-spread OFDM) | waveform validation path in `phy.py` |
| Vehicle speed | 60 km/h | maximum Doppler computed as `v fc / c` |
| Control step | 100 ms | 10 packets at a fixed 10 ms packet interval |
| Episode | 20 s, 200 steps | LOS 0-4 s, UrbanCanyon 4-16 s, LOS 16-20 s |

The direct CP reference is 3GPP TS 36.211 V14.2.0, section 9.2.1. The ETSI
publication explicitly states that PSSCH, PSCCH, PSBCH, and synchronization
signals configured for sidelink transmission Mode 3/4 support only normal CP.
Section 9.3.4 also maps PSSCH to transform precoding, while section 9.2.5 reserves
the final SC-FDMA symbol as a guard. Extended CP is consequently excluded as a
fixed C-V2X Mode-4 baseline even though the generic sidelink resource-block table
also lists an extended-CP configuration for other sidelink cases.

## 2. Fixed normal CP

At the 30.72 MHz LTE reference clock, normal CP is 160 Ts for the first symbol of
each slot and 144 Ts for each remaining symbol. At 15.36 MHz these are 80 and 72
samples, or 5.208333 us and 4.687500 us. A 1 ms subframe therefore contains:

- 2 first-slot symbols with 80-sample CP;
- 12 other symbols with 72-sample CP;
- 14 useful 1024-sample intervals;
- 1024 CP samples and 14336 useful samples (6.6667% time overhead).

`FixedNormalCPPolicy` preserves the symbol-dependent vector; it is not represented
by a validation-tuned scalar.

## 3. Rule-based policy and continuous CP state

The rule is `T_required = rho * estimated_RMS_DS`, followed by sample-level
clipping to the adaptive CP bounds. `rho=4.0` is a conservative research
coefficient motivated by OFDM/V2X design guidance that CP duration can be chosen
on the order of two to four times RMS delay spread; it is not a 3GPP value.
The Rule-based policy label is kept unchanged for figure readability.

The adaptive CP state is an integer native-rate sample count bounded by
`cp.min_samples` and `cp.max_samples`. DDQN does not select among pre-registered
CP-duration candidates. It selects only the direction of change, while the actual
change magnitude is computed from the current delay-spread tracking error.
All values other than the standard normal-CP symbol pattern are non-standard
extension candidates.

## 4. Timing convention

The system-level resource grid remains exactly 1 ms (15360 native samples). The
extension changes the guard/useful allocation within that fixed subframe budget;
unused or displaced resource elements are treated as unavailable payload and CP
overhead is `sum(CP samples)/15360`. The waveform gate evaluates
individual 1024-point DFT-spread-OFDM blocks with the selected guard. This is a
research abstraction: a deployable design would require transmitter/receiver
signalling, reference-signal redesign, and standard changes.

## 5. State, action, feasible reliability, and reward

The receiver-observable state is normalized estimated RMS delay spread, estimated
maximum excess delay, post-equalization SINR, RSRP, recent PRR, previous action,
normalized previous CP change, and normalized Doppler. True DS and LOS/NLOS labels
are logged only and never exposed to the agent.

The DDQN action is not an absolute CP value.  It is an incremental controller
over an integer CP state:

- decrease CP index;
- hold CP index;
- increase CP index.

The CP sample count is part of the environment state and persists across control
steps. The update is
`K_CP(t+1)=clip(K_CP(t)+direction*DeltaK(t), K_min, K_max)`, where
`DeltaK(t)=clip(ceil(delta_gain*abs(K_target(t)-K_CP(t))), delta_min, delta_max)`
and `K_target(t)=ceil(target_rho*estimated_DS/Ts)`. Feasible PRR is the PRR at
`K_max` for the current snapshot. The precheck still sweeps CP samples only for
PHY/oracle validation; those sweep values are not DDQN actions.

`gap=max(0, feasible_PRR - reliability_tolerance - selected_PRR)` and
`target_gap=max(0, min(target_PRR, feasible_PRR) - selected_PRR)`. Avoidable
outage is logged when the selected CP misses the PRR target while the longest CP
could meet it; unavoidable outage is logged when even the longest CP cannot meet
the target. Reward terms penalize feasible-reliability gap, target gap, avoidable
outage, CP overhead, CP switching, and boundary actions. Reliability penalties
dominate overhead and switching terms.

## 6. PHY and BLER model

Channel taps are generated from correlated lognormal RMS-DS targets and an
exponential power-delay profile. Energy beyond the CP produces residual
inter-symbol/inter-block interference. The fast training abstraction applies
one-tap frequency-domain MMSE equalization and a QPSK BER-to-packet-success
mapping. The independent waveform gate implements DFT spreading, subcarrier
mapping, IFFT, CP insertion, causal multipath convolution, FFT equalization, and
inverse DFT spreading.

Calibration and validation trace IDs are disjoint. Gate outputs include PRR MAE,
rank correlation/agreement, and the CP-sensitive snapshot proportion. Gate
thresholds live in configuration and are not adjusted after test evaluation.

## 7. Channel model

The controlled trace is LOS-UrbanCanyon-LOS. Path loss and
log-delay-spread parameter mappings are isolated in `standards.py` and labelled
as TR 37.885-based. UrbanCanyon uses the building-NLOS baseline plus explicit
delay-tail stress parameters so the experiment contains snapshots where normal CP
is insufficient but longer CP can recover PRR. It is a stress case, not a claim
that 3GPP specifies adaptive CP for that state. Shadowing, delay spread,
interference, and estimation errors are temporally correlated. Each time step
includes independent fast-fading tap phases while its large-scale state remains
correlated. Exact document versions/table numbers must be checked against the
cited originals before publication.

Primary documents checked during implementation:

- ETSI TS 136 211 V14.2.0 (3GPP TS 36.211 Release 14):
  https://www.etsi.org/deliver/etsi_ts/136200_136299/136211/14.02.00_60/ts_136211v140200p.pdf
- 3GPP specification record for TR 37.885:
  https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3209

## 8. Training and statistics

Five seeds train independently for exactly 2000 episodes. Networks, optimizers,
and replay buffers are never shared. Validation uses separate traces; the best
validation checkpoint and final checkpoint are both saved. Every best checkpoint
is evaluated on the same unseen trace IDs with common packet uniforms for paired
policy comparison.

Reports contain raw episode results, seed means, 95% CIs over the five independent
seed means, paired bootstrap differences
for DDQN-Fixed and DDQN-Rule, and a pre-registered PRR non-inferiority margin.
Representative time plots select the episode nearest median DDQN NLOS PRR, then
nearest median goodput.

## 9. Paper outputs

Six figures are generated: train reward, delay spread, CP duration, packet
reception ratio, useful RF energy efficiency, and a summary metric table. Every
figure owns an upper-right internal legend with 25% headroom. Fixed, Rule, and
DDQN use consistent vivid colors and solid lines; differing line widths/z-orders
separate overlapping policies.

The table includes PRR, goodput, mean CP, CP overhead, useful RF energy efficiency,
LOS/UrbanCanyon PRR, avoidable/unavoidable outage, PIR95, maximum
failure burst, NLOS feasible-reliability gap, and CP switch rate.
PIR is the elapsed time between successful packet receptions at one intended
receiver (the receiver-level/Type-2-style metric); it is not a network-wide
inter-reception statistic over multiple receivers.

## 10. Limitations

This code is a reproducible research simulator, not a conformance implementation.
SB-SPS interference is represented through a correlated interference process, not
a full network-wide MAC simulator. The link abstraction must pass the waveform
gate before training. Adaptive timing/signalling is intentionally modelled as a
PHY extension and is not interoperable with unmodified Mode-4 receivers.
